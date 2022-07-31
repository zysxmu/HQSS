from __future__ import print_function
import os
import abc
import copy
import tqdm
import time
import math
import random
import IPython
import argparse
import itertools

import numpy as np
from skimage import io, color
from focal_frequency_loss import FocalFrequencyLoss as FFL

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal, vis, savename
from test_online import test_online, test_online_best
from datasets.datasets_decouple import ImageDataset_decouple
from model import Generator_Encoder_SC, Generator_decoder_SC, define_models


def get_dataloader(opt):
    if True:
        dataloader = DataLoader(ImageDataset_decouple(opt.dataroot),
                                    batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    return dataloader

def get_losses():
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ffl_loss = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class
    return criterion_GAN, criterion_cycle, criterion_identity, kl_loss, ffl_loss

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
parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
parser.add_argument('--seed', type=int, default=628)
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

opt.savename = '' # directory for saving
opt.dataroot = './ISTD_Dataset' # path for dataset
savename(opt)

print(opt)
open(opt.log_path, 'a').write(str(opt) + '\n')

import shutil
shutil.copy('train_shadow_generation_network.py', opt.savename)


###### Definition of variables ######
# Networks
netG_A2B, netEn_sf, netD_S, netD_NS, netG_1, netG_2 = define_models()

# Losses
criterion_GAN, criterion_cycle, criterion_identity, kl_loss, ffl_loss = get_losses()

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
G_GAN_losses_netD_S_temp = 0
G_GAN_losses_netD_NS_temp = 0
G_rec_losses_temp = 0
G_ffl_losses_temp = 0
G_cycle_rec_losses_temp = 0
G_ffl_cycle_rec_losses_temp = 0
G_mean_light_losses_temp = 0
G_ffl_cycle_feat_rec_losses_temp = 0
G_loss_lab_losses_temp = 0
D_losses_temp = 0
G_losses = []
G_GAN_netD_S_losses = []
G_GAN_losses_netD_NS_losses = []
G_rec_losses = []
D_losses = []
G_ffl_losses = []
G_cycle_rec_losses = []
G_ffl_cycle_rec_losses = []
G_mean_light_losses = []
G_ffl_cycle_feat_rec_losses = []
G_loss_lab_losses = []

print('len dataloader', len(dataloader))
open(opt.log_path, 'w').write(str(opt) + '\n\n')

best_shadow_rmse = 100000000
best_shadow_rmse_epoch = -1
best_shadow_rmse_PI = 100000000
best_shadow_rmse_PI_epoch = -1

recloss_scale = 1
scale_gan = 0.05
cycleloss_scale = 0
dialoss_scale = 0
srloss_scale = 0
scale_D_S = 1
scale_D_NS = 1
scale_ffl = 1
warm_e = 0

print('recloss_scale', recloss_scale, 'scale_gan',
      scale_gan, 'cycleloss_scale', cycleloss_scale, 'dialoss_scale', dialoss_scale,
      'dialoss_scale', dialoss_scale, 'srloss_scale', srloss_scale, 'scale_D_S', scale_D_S, 'scale_D_NS', scale_D_NS,
      'scale_ffl', scale_ffl)

netG_A2B_copy = Generator_decoder_SC()
netG_A2B_copy.cuda()
netG_A2B_copy.eval()
netEn_sf_copy = Generator_Encoder_SC()
netEn_sf_copy.cuda()
netEn_sf_copy.eval()

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    time_start = time.time()
    for i, batch in enumerate(dataloader):
        curr_iter += 1

        # Set model input
        S1 = Variable(input_S1.copy_(batch['S1']))
        NS1 = Variable(input_NS1.copy_(batch['NS1']))

        M_S1 = Variable(input_M_S1.copy_(batch['M_S1']))
        M_NS1 = Variable(input_M_NS1.copy_(batch['M_NS1']))

        real_NS1_nsrs = Variable(input_real_NS1_nsrs.copy_(batch['real_NS1_nsrs']))
        real_ns_1 = Variable(input_real_ns_1.copy_(batch['real_ns_1']))
        M_NS1_dil = Variable(input_M_NS1_dil.copy_(batch['M_NS1_dil']))

        if torch.sum(M_S1) < 10 or torch.sum(M_NS1) < 10:
            print('###')
            print('skip!!! ', batch['S1_name'], torch.sum(M_S1), torch.sum(M_NS1))
            print('###')
            continue

        ####
        #### generator
        ####

        #### feature
        S1_shadow_feature = netEn_sf(S1)
        NS1_shadow_feature = netEn_sf(NS1)

        #### rec
        S1_rec = netG_A2B(S1_shadow_feature, S1)
        NS1_rec = netG_A2B(NS1_shadow_feature, NS1)
        fake_S1_NS1 = netG_A2B(S1_shadow_feature, NS1)
        fake_NS1_S1 = netG_A2B(NS1_shadow_feature, S1)

        ### loss light
        loss_mean_light = torch.tensor(0.)

        loss_lab = (criterion_identity(fake_S1_NS1[0, 1][M_NS1[0, 0] != 0], NS1[0, 1][M_NS1[0, 0] != 0]) +
                    criterion_identity(fake_S1_NS1[0, 2][M_NS1[0, 0] != 0], NS1[0, 2][M_NS1[0, 0] != 0]) +
                    criterion_identity(fake_NS1_S1[0, 1][M_S1[0, 0] != 0], S1[0, 1][M_S1[0, 0] != 0]) +
                    criterion_identity(fake_NS1_S1[0, 2][M_S1[0, 0] != 0], S1[0, 2][M_S1[0, 0] != 0])) * 0.01

        if epoch < 100:
            recloss_scale = recloss_scale
        else:
            recloss_scale = 1

        ##### identify loss
        loss_S1_rec = criterion_identity(S1_rec, S1) * recloss_scale
        loss_NS1_rec = criterion_identity(NS1_rec, NS1) * recloss_scale

        ### ffl loss
        loss_ffl_1 = ffl_loss(S1_rec, S1)
        loss_ffl_5 = ffl_loss(NS1_rec, NS1)

        ####
        if epoch < 100:
            scale_gan = scale_gan
        else:
            scale_gan = 1
        ### fake shadow image gan loss
        pred_fake_S1_NS1 = netD_S(fake_S1_NS1)
        loss_GAN_netD_S_1 = criterion_GAN(pred_fake_S1_NS1, target_real) * scale_gan

        ###
        pred_fake_NS1_S1 = netD_NS(fake_NS1_S1)
        loss_GAN_netD_NS_1 = criterion_GAN(pred_fake_NS1_S1, target_real) * scale_gan

        #####
        netG_A2B_copy = copy.deepcopy(netG_A2B)
        netEn_sf_copy = copy.deepcopy(netEn_sf)
        ### loss cycle rec
        if epoch < -1:
            loss_cycle_fake_S1_NS1_rec = torch.tensor(0.)
            loss_cycle_fake_NS1_S1_rec = torch.tensor(0.)
            loss_ffl_cycle_1 = torch.tensor(0.)
            loss_ffl_cycle_2 = torch.tensor(0.)
        else:
            ## cycle
            cycle_fake_S1_NS1 = netG_A2B_copy(NS1_shadow_feature, fake_S1_NS1)
            cycle_fake_NS1_S1 = netG_A2B_copy(S1_shadow_feature, fake_NS1_S1)
            loss_cycle_fake_S1_NS1_rec = criterion_identity(cycle_fake_S1_NS1, NS1) * 0.1
            loss_cycle_fake_NS1_S1_rec = criterion_identity(cycle_fake_NS1_S1, S1) * 0.1
            loss_ffl_cycle_1 = ffl_loss(cycle_fake_S1_NS1, NS1) * 0.1
            loss_ffl_cycle_2 = ffl_loss(cycle_fake_NS1_S1, S1) * 0.1

            cycle_fake_S1_NS1_shadowfeat = netEn_sf_copy(fake_S1_NS1)
            cycle_fake_NS1_S1_shadowfeat = netEn_sf_copy(fake_NS1_S1)
            loss_ffl_cycle_feat_1 = criterion_identity(cycle_fake_S1_NS1_shadowfeat, S1_shadow_feature) * 0.01
            loss_ffl_cycle_feat_2 = criterion_identity(cycle_fake_NS1_S1_shadowfeat, NS1_shadow_feature) * 0.01

        loss_G = loss_S1_rec + loss_NS1_rec + \
                 loss_GAN_netD_S_1 + loss_GAN_netD_NS_1 + \
                 loss_ffl_1 + loss_ffl_5 + \
                 loss_mean_light + \
                 loss_cycle_fake_S1_NS1_rec + loss_cycle_fake_NS1_S1_rec + \
                 loss_ffl_cycle_feat_1 + loss_ffl_cycle_feat_2 + \
                 loss_ffl_cycle_1 + loss_ffl_cycle_2 + loss_lab


        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        G_losses_temp += loss_G.item()
        G_rec_losses_temp += loss_S1_rec.item() + loss_NS1_rec.item()
        G_GAN_losses_netD_S_temp += loss_GAN_netD_S_1.item()
        G_GAN_losses_netD_NS_temp += loss_GAN_netD_NS_1.item()
        G_ffl_losses_temp += loss_ffl_1.item() + loss_ffl_5.item()
        G_cycle_rec_losses_temp += loss_cycle_fake_S1_NS1_rec.item() + loss_cycle_fake_NS1_S1_rec.item()
        G_ffl_cycle_rec_losses_temp += loss_ffl_cycle_1.item() + loss_ffl_cycle_2.item()
        G_mean_light_losses_temp += loss_mean_light.item()
        G_loss_lab_losses_temp += loss_lab.item()
        G_ffl_cycle_feat_rec_losses_temp += loss_ffl_cycle_feat_1.item() + loss_ffl_cycle_feat_2.item()

        #####
        ##### Discriminator
        #####

        if epoch < 100:
            scale_D_S = scale_D_S
        else:
            scale_D_S = 1
        # Real loss
        pred_real_D_S_1 = netD_S(S1)
        loss_D_S_real_1 = criterion_GAN(pred_real_D_S_1, target_real) * scale_D_S

        # Fake loss
        fake_S_1 = fake_S_buffer.push_and_pop(fake_S1_NS1.detach())
        pred_fake_D_S_1 = netD_S(fake_S_1)
        loss_fake_D_S_1 = criterion_GAN(pred_fake_D_S_1, target_fake) * scale_D_S

        ###
        if epoch < 100:
            scale_D_NS = scale_D_NS
        else:
            scale_D_NS = 1
        # Real loss
        pred_real_D_NS_1 = netD_NS(NS1)
        loss_D_NS_real_1 = criterion_GAN(pred_real_D_NS_1, target_real) * scale_D_NS

        # Fake loss
        fake_NS_1 = fake_NS_buffer.push_and_pop(fake_NS1_S1.detach())
        pred_fake_D_NS_1 = netD_NS(fake_NS_1)
        loss_fake_D_NS_1 = criterion_GAN(pred_fake_D_NS_1, target_fake) * scale_D_NS

        # Total loss
        loss_D = loss_D_S_real_1 + \
                 loss_fake_D_S_1 + \
                 loss_D_NS_real_1 + \
                 loss_fake_D_NS_1
        D_losses_temp += loss_D.item()

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        if (i+1) % opt.iter_loss == 0:
            time_end = time.time()
            time_sum = time_end - time_start
            time_log = '%d iteration time: %.3f' % (opt.iter_loss, time_sum)
            print(time_log)
            open(opt.log_path, 'a').write(time_log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_S1_rec %.5f], ' \
                  '[loss_NS1_rec %.5f],' \
                  '[loss_GAN_netD_S_1 %.5f]' \
                  '[loss_GAN_netD_NS_1 %.5f],' \
                  '[loss_ffl_1 %.5f], ' \
                  '[loss_ffl_5 %.5f]' % \
                  (epoch + 1, curr_iter, loss_G, loss_S1_rec, loss_NS1_rec,
                   loss_GAN_netD_S_1,
                   loss_GAN_netD_NS_1,
                   loss_ffl_1, loss_ffl_5
                   )
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            log = 'Epoch: %d, [iter %d], [loss_D %.5f], [loss_D_S_real_1 %.5f]' \
                  '[loss_fake_D_S_1 %.5f], [loss_D_NS_real_1 %.5f], ' \
                  '[loss_fake_D_NS_1 %.5f]' % \
                  (epoch + 1, curr_iter, loss_D, loss_D_S_real_1, loss_fake_D_S_1, loss_D_NS_real_1, loss_fake_D_NS_1
                   )

            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_losses.append(D_losses_temp / opt.iter_loss)
            G_GAN_netD_S_losses.append(G_GAN_losses_netD_S_temp / opt.iter_loss)
            G_GAN_losses_netD_NS_losses.append(G_GAN_losses_netD_NS_temp / opt.iter_loss)
            G_rec_losses.append(G_rec_losses_temp / opt.iter_loss)
            G_ffl_losses.append(G_ffl_losses_temp / opt.iter_loss)
            G_cycle_rec_losses.append(G_cycle_rec_losses_temp / opt.iter_loss)
            G_ffl_cycle_rec_losses.append(G_ffl_cycle_rec_losses_temp / opt.iter_loss)
            G_mean_light_losses.append(G_mean_light_losses_temp / opt.iter_loss)
            G_ffl_cycle_feat_rec_losses.append(G_ffl_cycle_feat_rec_losses_temp / opt.iter_loss)
            G_loss_lab_losses.append(G_loss_lab_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            D_losses_temp = 0
            G_GAN_losses_netD_S_temp = 0
            G_rec_losses_temp = 0
            G_GAN_losses_netD_NS_temp = 0
            G_cycle_rec_losses_temp = 0
            G_ffl_losses_temp = 0
            G_cycle_rec_temp = 0
            G_ffl_cycle_rec_losses_temp = 0
            G_mean_light_losses_temp = 0
            G_ffl_cycle_feat_rec_losses_temp = 0
            G_loss_lab_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f] [loss_rec %.5f]' \
                      '[G_GAN_netD_S_losses %.5f], [G_GAN_losses_netD_NS_losses %.5f], ' \
                      '[G_ffl_losses %.5f], [G_cycle_rec_losses %.5f] [G_ffl_cycle_rec_losses %.5f]' \
                      '[G_mean_light_losses %.5f] [G_ffl_cycle_feat_rec_losses %.5f]' \
                      '[G_loss_lab_losses %.5f]' \
                      '[D_losses %.5f], ' \
                      % (opt.iter_loss, G_losses[G_losses.__len__() - 1], \
                         G_rec_losses[G_rec_losses.__len__() - 1],
                         G_GAN_netD_S_losses[G_GAN_netD_S_losses.__len__() - 1],
                         G_GAN_losses_netD_NS_losses[G_GAN_losses_netD_NS_losses.__len__() - 1],
                         G_ffl_losses[G_ffl_losses.__len__() - 1],
                         G_cycle_rec_losses[G_cycle_rec_losses.__len__() - 1],
                         G_ffl_cycle_rec_losses[G_ffl_cycle_rec_losses.__len__() - 1],
                         G_mean_light_losses[G_mean_light_losses.__len__() - 1],
                         G_ffl_cycle_feat_rec_losses[G_ffl_cycle_feat_rec_losses.__len__() - 1],
                         G_loss_lab_losses[G_loss_lab_losses.__len__()-1],
                         D_losses[D_losses.__len__() - 1])
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
