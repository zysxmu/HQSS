from __future__ import print_function
import os
import tqdm
import time
import random
import IPython
import argparse
import itertools
import numpy as np

from skimage import io, color
from skimage.transform import resize

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import ReplayBuffer
from utils import weights_init_normal, vis, load_online_test_data
from model import Generator_S2F, Generator_F2S
from datasets.datasets import ImageDataset
from test_online import test_online, test_online_best, test_online_SRD, test_online_best_SRD, test_online_SRD, test_online_best_SRD


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
parser.add_argument('--seed', type=int, default=628)
parser.add_argument('--savename', type=str, default='')
parser.add_argument("--unaligned", type=int, default=1)
# parser.add_argument("--preload", type=int, default=0)
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opt.dataroot = '/home/zys/G2R-SRD-dataset-new'
# opt.dataroot = '/home/zys/G2R-dataset-sup'
# opt.dataroot = '/home/zys/G2R-SRD-dataset-sup'

if 'new' in opt.dataroot:
    opt.savename = opt.savename + '-new'

if opt.unaligned:
    opt.savename = opt.savename + '-unaligned'
else:
    opt.savename = opt.savename + '-Nounaligned'

if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
opt.log_path = os.path.join('ckpt/'+opt.savename, 'seed%dep%d' % (opt.seed, opt.n_epochs) + '.txt')

if not os.path.exists('ckpt/'+opt.savename):
    os.mkdir('ckpt/'+opt.savename)

if torch.cuda.is_available():
    opt.cuda = True

print(opt)
open(opt.log_path, 'a').write(str(opt) + '\n')


###### Definition of variables ######
# Networks
netG_1 = Generator_S2F()  # shadow to shadow_free
netG_2 = Generator_F2S()  # shadow to shadow_free

if opt.cuda:
    netG_1.cuda()
    netG_2.cuda()

netG_1.apply(weights_init_normal)
netG_2.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.MSELoss()  # lsgan
# criterion_GAN = torch.nn.BCEWithLogitsLoss() # vanilla
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer = torch.optim.Adam(itertools.chain(netG_1.parameters(), netG_2.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_C = Tensor(opt.batchSize, 1, opt.size, opt.size)
input_D = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_E = Tensor(opt.batchSize, 3, opt.size, opt.size)
input_F = Tensor(opt.batchSize, 1, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
fake_B_buffer = ReplayBuffer()

dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=opt.unaligned),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

curr_iter = 0
G_losses_temp = 0
D_B_losses_temp = 0
G_losses = []
D_B_losses = []

print('len dataloader', len(dataloader))
open(opt.log_path, 'w').write(str(opt) + '\n\n')

# load online test data
rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list = load_online_test_data(opt)

best_shadow_rmse = 100000000
best_shadow_rmse_epoch = -1
best_shadow_rmse_PI = 100000000
best_shadow_rmse_PI_epoch = -1

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    time_start = time.time()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Set model input
        real_nsr = Variable(input_A.copy_(batch['A']))  # non shadow region:input: step1-gt
        random_sr = Variable(input_B.copy_(batch['B']))  # random real shadow region: gan training
        mask = Variable(input_C.copy_(batch['C']))  # nonshadow region mask: step2-input
        real_ns = Variable(input_D.copy_(batch['D']))  # without real shadow region: step2-gt
        real_nsrs = Variable(input_E.copy_(batch['E']))  # without nonshadow region and real shadow region: step2-input
        mask_dil = Variable(input_F.copy_(batch['F']))  # without nonshadow region and real shadow region: step2-input

        # vis('real_nsr', real_nsr)
        # vis('random_sr', random_sr)
        # vis('mask', mask, True)
        # vis('real_ns', real_ns)
        # vis('real_nsrs', real_nsrs)
        # vis('mask_dil', mask_dil, True)

        # GAN loss, netG_1:# shadow to shadow_free, fake_B-shadow, fake_nsr-non-shadow
        fake_nsr = netG_1(random_sr)
        loss_cycle = criterion_cycle(fake_nsr, real_nsr)

        # netG_2:# shadow to shadow_free, netG_2-finetune-network,
        output = netG_2(fake_nsr+real_nsrs, mask*2.0-1.0)
        loss_sr = criterion_identity(output, real_ns)

        loss_shadow = criterion_cycle(torch.cat( ((output[:, 0]+1.0)*mask_dil-1.0, output[:, 1:]*mask_dil), 1),
                                    torch.cat( ((real_ns[:, 0]+1.0)*mask_dil-1.0, real_ns[:, 1:]*mask_dil), 1))

        loss_G = loss_cycle + loss_sr + loss_shadow
        loss_G.backward()

        G_losses_temp += loss_G.item()
        curr_iter += 1
        optimizer.step()

        if (i+1) % opt.iter_loss == 0:
            time_end = time.time() 
            time_sum = time_end - time_start
            time_log = '%d iteration time: %.3f' % (opt.iter_loss, time_sum)
            print(time_log)
            open(opt.log_path, 'a').write(time_log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_B_losses.append(D_B_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            D_B_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f], [D_B_losses %.5f],' \
                      % (opt.iter_loss, G_losses[G_losses.__len__()-1], \
                         D_B_losses[D_B_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

            time_start = time.time()

    if epoch > -1:
        # test time
        time_start = time.time()
        if 'SRD' in opt.dataroot:
            shadow_rmse, shadow_rmse_PI = \
                test_online_SRD(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list,
                            gt_list, epoch, 0)
        else:
            shadow_rmse, shadow_rmse_PI = \
                test_online(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list,
                            gt_list, epoch, 0)

        time_end = time.time()
        time_sum = time_end - time_start
        print('test time:  %.3f' % (time_sum))
        open(opt.log_path, 'a').write('test time:  %.3f' % (time_sum) + '\n')

        if shadow_rmse < best_shadow_rmse:
            best_shadow_rmse = shadow_rmse
            best_shadow_rmse_epoch = epoch+1
            print('save!', ('ckpt/' + opt.savename +
                            '/seed%dep%d_netG_1_best_shadow_rmse.pth' % (opt.seed, opt.n_epochs)))
            open(opt.log_path, 'a').write('save!' + ('ckpt/' + opt.savename +
                            '/seed%dep%d_netG_1_best_shadow_rmse.pth' % (opt.seed, opt.n_epochs)) + '\n')
            torch.save(netG_1.state_dict(),
                       ('ckpt/' + opt.savename +
                        '/seed%dep%d_netG_1_best_shadow_rmse.pth' % (opt.seed, opt.n_epochs)))
            torch.save(netG_2.state_dict(),
                       ('ckpt/' + opt.savename +
                        '/seed%dep%d_netG_2_best_shadow_rmse.pth' % (opt.seed, opt.n_epochs)))

        if shadow_rmse_PI < best_shadow_rmse_PI:
            best_shadow_rmse_PI = shadow_rmse_PI
            best_shadow_rmse_PI_epoch = epoch+1
            print('save!', ('ckpt/' + opt.savename +
                            '/seed%dep%d_netG_1_best_shadow_rmse_PI.pth' % (opt.seed, opt.n_epochs)))
            open(opt.log_path, 'a').write('save!' + ('ckpt/' + opt.savename +
                            '/seed%dep%d_netG_1_best_shadow_rmse_PI.pth' % (opt.seed, opt.n_epochs)) + '\n')
            torch.save(netG_1.state_dict(),
                       ('ckpt/' + opt.savename +
                        '/seed%dep%d_netG_1_best_shadow_rmse_PI.pth' % (opt.seed, opt.n_epochs)))
            torch.save(netG_2.state_dict(),
                       ('ckpt/' + opt.savename +
                        '/seed%dep%d_netG_2_best_shadow_rmse_PI.pth' % (opt.seed, opt.n_epochs)))

    netG_1.train()
    netG_2.train()

    if epoch > opt.n_epochs-10:
        print('save!', ('ckpt/'+opt.savename+'/seed%dep%d_netG_1_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netG_1.state_dict(), ('ckpt/'+opt.savename+'/seed%dep%d_netG_1_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netG_2.state_dict(), ('ckpt/'+opt.savename+'/seed%dep%d_netG_2_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))

    print('best_shadow_rmse_epoch', best_shadow_rmse_epoch, 'best_shadow_rmse', best_shadow_rmse)
    print('best_shadow_rmse_PI_epoch', best_shadow_rmse_PI_epoch, 'best_shadow_rmse_PI', best_shadow_rmse_PI)
    open(opt.log_path, 'a').write('best_shadow_rmse_epoch' + str(best_shadow_rmse_epoch) +
                                  'best_shadow_rmse' + str(best_shadow_rmse) + '\n')
    open(opt.log_path, 'a').write('best_shadow_rmse_PI_epoch' + str(best_shadow_rmse_PI_epoch)
                                  + 'best_shadow_rmse_PI' + str(best_shadow_rmse_PI) + '\n')

if 'SRD' in opt.dataroot:
    test_online_best_SRD(opt.savename,
                     rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, 0)
else:
    test_online_best(opt.savename,
                     rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, 0)
