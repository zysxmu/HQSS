from __future__ import print_function
import os
import abc
import time
import tqdm
import random
import IPython
import argparse
import itertools
import numpy as np
from skimage import io, color
from skimage.transform import resize

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import ReplayBuffer, LambdaLR
from utils import weights_init_normal, vis, load_online_test_data, savename
from model import define_models
from datasets.datasets_decouple import ImageDataset_decouple
from test_online import test_online, test_online_best, test_online_SRD, test_online_best_SRD


def get_dataloader(opt):
    if True:
        dataloader = DataLoader(ImageDataset_decouple(opt.dataroot),
                                    batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)
    return dataloader

def get_losses():
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return criterion_GAN, criterion_cycle, criterion_identity, kl_loss

def get_optimizer_LRsch(opt, netG_A2B, netEn_sf, netG_1, netG_2, netD_S, netD_NS):
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netEn_sf.parameters(),
                                                   netG_1.parameters(), netG_2.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(itertools.chain(netD_S.parameters(), netD_NS.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    
    return optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D

def get_tensors_buffer(opt):
    Tensor = torch.cuda.FloatTensor
    input_S1 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_S2 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_NS1 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_NS2 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_M_S1 = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_M_S2 = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_M_NS1 = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_M_NS2 = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_real_NS1_nsrs = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_real_NS2_nsrs = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_real_ns_1 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_real_ns_2 = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_M_NS1_dil = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_M_NS2_dil = Tensor(opt.batchSize, 1, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_S_buffer = ReplayBuffer(max_size=200)
    fake_NS_buffer = ReplayBuffer(max_size=200)

    return input_S1, input_S2, input_NS1, input_NS2, input_M_S1, \
           input_M_S2, input_M_NS1, input_M_NS2, target_real, target_fake, input_real_NS1_nsrs, input_real_NS2_nsrs, \
           input_real_ns_1, input_real_ns_2, input_M_NS1_dil, input_M_NS2_dil, \
           fake_S_buffer, fake_NS_buffer

def shadow_removal_loss(S, NS, real_NS_nsrs, srloss_scale, M_NS, M_NS_dil,
                        real_ns, cycleloss_scale, dialoss_scale, is_clear):

    with torch.no_grad():
        S_shadow_feature = netEn_sf(S)
        fake_S_NS = netG_A2B(S_shadow_feature, NS)

    fake_S_NS = fake_S_NS.detach()

    ### clear fake image
    fake_S_NS = fake_S_NS * M_NS + NS * (1 - M_NS)
    
    fake_S_NS_nsr = netG_1(fake_S_NS)

    loss_cycle = criterion_cycle(fake_S_NS_nsr, NS) * cycleloss_scale

    # netG_2:# shadow to shadow_free, netG_2-finetune-network,
    if not is_clear:
        output_fake_S_NS_nsr = netG_2(fake_S_NS_nsr + real_NS_nsrs, M_NS * 2.0 - 1.0)
    else:
        ### clear further
        output_fake_S_NS_nsr = netG_2(fake_S_NS_nsr * M_NS + real_NS_nsrs * (1 - M_NS), M_NS * 2.0 - 1.0)


    loss_sr = criterion_identity(output_fake_S_NS_nsr, real_ns) * srloss_scale

    loss_dia = criterion_cycle(torch.cat(((output_fake_S_NS_nsr[:, 0] + 1.0) * M_NS_dil - 1.0,
                                            output_fake_S_NS_nsr[:, 1:] * M_NS_dil), 1),
                                 torch.cat(((real_ns[:, 0] + 1.0) * M_NS_dil + - 1.0,
                                            real_ns[:, 1:] * M_NS_dil), 1)) * dialoss_scale

    return loss_cycle, loss_sr, loss_dia


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
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opt.savename = '' # directory for saving
opt.dataroot = './ISTD_Dataset' # path for dataset
opt.clear = 0
savename(opt)

print(opt)
open(opt.log_path, 'a').write(str(opt) + '\n')

###### Definition of variables ######
# Networks
netG_A2B, netEn_sf, netD_S, netD_NS, netG_1, netG_2 = define_models()

# Losses
criterion_GAN, criterion_cycle, criterion_identity, kl_loss = get_losses()

# Optimizers & LR schedulers
optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D = \
    get_optimizer_LRsch(opt, netG_A2B, netEn_sf, netG_1, netG_2, netD_S, netD_NS)

# Inputs & targets memory allocation
input_S1, input_S2, input_NS1, input_NS2, input_M_S1, \
           input_M_S2, input_M_NS1, input_M_NS2, target_real, target_fake, input_real_NS1_nsrs, input_real_NS2_nsrs, \
           input_real_ns_1, input_real_ns_2, input_M_NS1_dil, input_M_NS2_dil, \
            fake_S_buffer, fake_NS_buffer = get_tensors_buffer(opt)

# Dataset loader
dataloader = get_dataloader(opt)

curr_iter = 0
G_losses_temp = 0
G_GAN_losses_temp = 0
G_cycle_losses_temp = 0
G_sr_losses_temp = 0
G_dia_losses_temp = 0
G_rec_losses_temp = 0
D_losses_temp = 0
G_losses = []
G_GAN_losses = []
G_cycle_losses = []
G_sr_losses = []
G_dia_losses = []
G_rec_losses = []
D_losses = []

print('len dataloader', len(dataloader))
open(opt.log_path, 'w').write(str(opt) + '\n\n')

# load online test data
rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list = load_online_test_data(opt)

best_shadow_rmse = 100000000
best_shadow_rmse_epoch = -1
best_shadow_rmse_PI = 100000000
best_shadow_rmse_PI_epoch = -1

recloss_scale = 1
scale_gan = 0.1
cycleloss_scale = 1
dialoss_scale = 1
srloss_scale = 1
scale_D_S = 1
print('recloss_scale', recloss_scale, 'scale_gan',
      scale_gan, 'cycleloss_scale', cycleloss_scale, 'dialoss_scale', dialoss_scale,
      'dialoss_scale', dialoss_scale, 'srloss_scale', srloss_scale, 'scale_D_S', scale_D_S)

name = ""


print('start training!')

import shutil
shutil.copy('train_shadow_removal_network.py', opt.savename)

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    time_start = time.time()

    checkpoints = ['100']

    print(checkpoints)
    open(opt.log_path, 'a').write('checkpoints: ' + str(checkpoints) + '\n')
    index = random.randint(0, len(checkpoints)-1)

    path = 'netG_A2B.pth'
    print('load: ' + path)
    open(opt.log_path, 'a').write('load: ' + path + '\n')
    netG_A2B.load_state_dict(torch.load(path))

    path = 'netEn_sf.pth'
    print('load: ' + path)
    open(opt.log_path, 'a').write('load: ' + path + '\n')
    netEn_sf.load_state_dict(torch.load(path))

    netEn_sf.eval()
    netG_A2B.eval()

    for i, batch in enumerate(dataloader):
        curr_iter += 1

        # Set model input
        S1 = Variable(input_S1.copy_(batch['S1']))
        NS1 = Variable(input_NS1.copy_(batch['NS1']))
        M_S1 = Variable(input_M_S1.copy_(batch['M_S1']))

        ##
        M_NS1 = Variable(input_M_NS1.copy_(batch['M_NS1']))
        real_NS1_nsrs = Variable(input_real_NS1_nsrs.copy_(batch['real_NS1_nsrs']))


        real_ns_1 = Variable(input_real_ns_1.copy_(batch['real_ns_1']))

        M_NS1_dil = Variable(input_M_NS1_dil.copy_(batch['M_NS1_dil']))

        S1_name = batch['S1_name']

        if torch.sum(M_S1) < 10 or torch.sum(M_NS1) < 10:
            print('###')
            print('skip!!! ', batch['S1_name'], torch.sum(M_S1), torch.sum(M_NS1))
            print('###')
            continue

        ####

        ind = 1
        if ind == 1:
            loss_cycle_1, loss_sr_1, loss_dia_1 = \
                shadow_removal_loss(S1, NS1, real_NS1_nsrs, srloss_scale, M_NS1, M_NS1_dil, real_ns_1, cycleloss_scale,
                                    dialoss_scale, opt.clear)

            loss_cycle_2 = loss_sr_2 = loss_dia_2 = torch.tensor(0.).cuda()
            loss_cycle_3 = loss_sr_3 = loss_dia_3 = torch.tensor(0.).cuda()
            loss_cycle_4 = loss_sr_4 = loss_dia_4 = torch.tensor(0.).cuda()


        loss_S1_rec = loss_S2_rec = loss_S1_rec_other_shadow_feature = \
            loss_S2_rec_other_shadow_feature = 0

        loss_NS1_rec = loss_NS2_rec = loss_NS1_rec_other_shadow_feature = \
            loss_NS2_rec_other_shadow_feature = 0

        loss_GAN_netD_S_1 = loss_GAN_netD_S_2 = loss_GAN_netD_S_3 = \
            loss_GAN_netD_S_4 = 0


        loss_G = loss_S1_rec + loss_S2_rec + loss_NS1_rec + loss_NS2_rec \
                 + loss_S1_rec_other_shadow_feature + loss_S2_rec_other_shadow_feature +\
                 loss_NS1_rec_other_shadow_feature + loss_NS2_rec_other_shadow_feature + \
                 loss_GAN_netD_S_1 + loss_GAN_netD_S_2 + loss_GAN_netD_S_3 + loss_GAN_netD_S_4 +\
                 loss_cycle_1 + loss_cycle_2 + loss_cycle_3 + loss_cycle_4 + \
                 loss_sr_1 + loss_sr_2 + loss_sr_3 + loss_sr_4 + \
                 loss_dia_1 + loss_dia_2 + loss_dia_3 + loss_dia_4


        G_losses_temp += loss_G.item()
        G_rec_losses_temp += 0
        G_GAN_losses_temp += 0
        G_cycle_losses_temp += loss_cycle_1.item() + loss_cycle_2.item() + loss_cycle_3.item() + loss_cycle_4.item()
        G_sr_losses_temp += loss_sr_1.item() + loss_sr_2.item() + loss_sr_3.item() + loss_sr_4.item()
        G_dia_losses_temp += loss_dia_1.item() + loss_dia_2.item() + loss_dia_3.item() + loss_dia_4.item()

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()


        #####
        ##### Discriminator
        #####

        if epoch < 100:
            scale_D_S = scale_D_S
        else:
            scale_D_S = 1
        # Real loss
        loss_D_S_real_1 = loss_D_S_real_2 = 0

        # Fake loss
        loss_fake_D_S_1 = loss_fake_D_S_2 = loss_fake_D_S_3 = loss_fake_D_S_4 = 0

        # Total loss
        loss_D = 0
        D_losses_temp += 0

        if (i+1) % opt.iter_loss == 0:
            # function()   执行的程序
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            time_log = '%d iteration time: %.3f' % (opt.iter_loss, time_sum)
            print(time_log)
            open(opt.log_path, 'a').write(time_log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_S1_rec %.5f], [loss_S2_rec %.5f],' \
                  '[loss_NS1_rec %.5f], [loss_NS2_rec %.5f], [loss_S1_rec_other_shadow_feature %.5f],' \
                  '[loss_S2_rec_other_shadow_feature %.5f], [loss_NS1_rec_other_shadow_feature %.5f], ' \
                  '[loss_NS2_rec_other_shadow_feature %.5f],' \
                  '[loss_GAN_netD_S_1 %.5f], [loss_GAN_netD_S_2 %.5f], [loss_GAN_netD_S_3 %.5f],' \
                  '[loss_GAN_netD_S_4 %.5f], [loss_cycle_1 %.5f], [loss_cycle_2 %.5f], ' \
                  '[loss_cycle_3 %.5f], [loss_cycle_4 %.5f], [loss_sr_1 %.5f], [loss_sr_2 %.5f],' \
                  '[loss_sr_3 %.5f], [loss_sr_4 %.5f],' \
                  '[loss_dia_1 %.5f], [loss_dia_2 %.5f], [loss_dia_3 %.5f], [loss_dia_4 %.5f],' % \
                  (epoch + 1, curr_iter, loss_G, loss_S1_rec, loss_S2_rec, loss_NS1_rec, loss_NS2_rec,
                   loss_S1_rec_other_shadow_feature, loss_S2_rec_other_shadow_feature,
                   loss_NS1_rec_other_shadow_feature, loss_NS2_rec_other_shadow_feature,
                   loss_GAN_netD_S_1, loss_GAN_netD_S_2, loss_GAN_netD_S_3, loss_GAN_netD_S_4,
                   loss_cycle_1, loss_cycle_2, loss_cycle_3, loss_cycle_4,
                   loss_sr_1, loss_sr_2, loss_sr_3, loss_sr_4,
                   loss_dia_1, loss_dia_2, loss_dia_3, loss_dia_4
                   )
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_D %.5f], [loss_D_S_real_1 %.5f], [loss_D_S_real_2 %.5f],' \
                  '[loss_fake_D_S_1 %.5f], [loss_fake_D_S_2 %.5f], [loss_fake_D_S_3 %.5f],' \
                  '[loss_fake_D_S_4 %.5f]' % \
                  (epoch + 1, curr_iter, loss_D, loss_D_S_real_1, loss_D_S_real_2, \
                 loss_fake_D_S_1, loss_fake_D_S_2, loss_fake_D_S_3, loss_fake_D_S_4
                   )
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_losses.append(D_losses_temp / opt.iter_loss)
            G_GAN_losses.append(G_GAN_losses_temp / opt.iter_loss)
            G_cycle_losses.append(G_cycle_losses_temp / opt.iter_loss)
            G_sr_losses.append(G_sr_losses_temp / opt.iter_loss)
            G_dia_losses.append(G_dia_losses_temp / opt.iter_loss)
            G_rec_losses.append(G_rec_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            D_losses_temp = 0
            G_GAN_losses_temp = 0
            G_cycle_losses_temp = 0
            G_sr_losses_temp = 0
            G_dia_losses_temp = 0
            G_rec_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f] [loss_rec %.5f]' \
                      '[G_GAN_losses %.5f], [G_cycle_losses %.5f], [G_sr_losses %.5f], [G_dia_losses %.5f],' \
                      ' [D_losses %.5f], ' \
                      % (opt.iter_loss, G_losses[G_losses.__len__()-1], \
                         G_rec_losses[G_rec_losses.__len__() - 1],
                         G_GAN_losses[G_GAN_losses.__len__()-1],
                         G_cycle_losses[G_cycle_losses.__len__()-1],
                         G_sr_losses[G_sr_losses.__len__()-1],
                         G_dia_losses[G_dia_losses.__len__()-1],
                        D_losses[D_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

            time_start = time.time()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    if (epoch > opt.n_epochs - 10) or (epoch % 10 == 0):

        print('save!', ('ckpt/' + opt.savename + '/seed%dep%d_netG_1_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netG_A2B.state_dict(),
                   ('ckpt/' + opt.savename + '/seed%dep%d_netG_A2B_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netEn_sf.state_dict(),
                   ('ckpt/' + opt.savename + '/seed%dep%d_netEn_sf_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))

    if epoch > -1:
        # test time
        time_start = time.time()
        if 'SRD' in opt.dataroot:
            shadow_rmse, shadow_rmse_PI = \
                test_online_SRD(netG_1, netG_2, rgbimage_list, labimage_list, mask_list,
                            gtimage_list, truemask_list, gt_list, epoch, opt.clear)
        else:
            shadow_rmse, shadow_rmse_PI = \
                test_online(netG_1, netG_2, rgbimage_list, labimage_list, mask_list,
                            gtimage_list, truemask_list, gt_list, epoch, opt.clear)
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
        torch.save(netG_A2B.state_dict(),
                   ('ckpt/' + opt.savename + '/seed%dep%d_netG_A2B_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netG_1.state_dict(), ('ckpt/'+opt.savename+'/seed%dep%d_netG_1_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))
        torch.save(netG_2.state_dict(), ('ckpt/'+opt.savename+'/seed%dep%d_netG_2_%d.pth' % (opt.seed, opt.n_epochs, epoch + 1)))


    # print('Epoch:{}'.format(epoch))
    print('best_shadow_rmse_epoch', best_shadow_rmse_epoch, 'best_shadow_rmse', best_shadow_rmse)
    print('best_shadow_rmse_PI_epoch', best_shadow_rmse_PI_epoch, 'best_shadow_rmse_PI', best_shadow_rmse_PI)
    open(opt.log_path, 'a').write('best_shadow_rmse_epoch' + str(best_shadow_rmse_epoch) +
                                  'best_shadow_rmse' + str(best_shadow_rmse) + '\n')
    open(opt.log_path, 'a').write('best_shadow_rmse_PI_epoch' + str(best_shadow_rmse_PI_epoch)
                                  + 'best_shadow_rmse_PI' + str(best_shadow_rmse_PI) + '\n')

if 'SRD' in opt.dataroot:\
    test_online_best_SRD(opt.savename,
                 rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, opt.clear)
else:
    test_online_best(opt.savename,
                     rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, opt.clear)
