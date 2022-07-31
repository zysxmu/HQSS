import os
import glob
import time
import copy
import tqdm
import IPython
import argparse
import numpy as np
from skimage import io, color
from skimage.color import rgb2lab
from skimage.transform import resize
from skimage.util import img_as_ubyte
from os.path import exists, join as join_paths

import torch
import torchvision.transforms as transforms

from model import Generator_S2F,Generator_F2S
from utils import load_online_test_data

import warnings
warnings.filterwarnings("ignore")

def calculate_rmse(recovered, gt, mask):
    # Transform into lab color space
    recovered_lab = recovered
    gt_lab = gt
    
    return abs((gt_lab-recovered_lab) * mask).sum(), mask.sum()

def test_online(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, epoch, is_clear):
    netG_1.eval()
    netG_2.eval()

    idx = 0
    non_shadow_mse = 0
    shadow_mse = 0
    total_mse = 0
    total = 0
    total_shadow = 0
    total_non_shadow =  0

    total_mse_PI = 0
    total_shadow_mse_PI = 0
    total_non_shadow_mse_PI = 0

    # print('1')
    with torch.no_grad():

        for rgbimage_o, labimage_o, mask_o, gtimage_o, truemask_o, img_name in\
                zip(rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list):
            rgbimage = copy.deepcopy(rgbimage_o)
            labimage = copy.deepcopy(labimage_o)
            mask = copy.deepcopy(mask_o)
            gtimage = copy.deepcopy(gtimage_o)
            truemask = copy.deepcopy(truemask_o)

            # labimage480=resize(labimage,(480,640,3))
            labimage480 = copy.deepcopy(labimage)
            labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
            labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
            labimage480=torch.from_numpy(labimage480).float()
            labimage480=labimage480.view(480,640,3)
            labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
            labimage480=labimage480.unsqueeze(0).cuda()


            # mask480=resize(mask,(480,640))
            mask480 = copy.deepcopy(mask)
            mask480 = mask480/255.0
            mask480=torch.from_numpy(mask480).float()
            mask480=mask480.view(480,640,1)
            mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
            mask480=mask480.unsqueeze(0).cuda()
            zero = torch.zeros_like(mask480)
            one = torch.ones_like(mask480)
            mask480=torch.where(mask480 > 0.5, one, zero)

            real_s480=labimage480.clone()
            real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
            real_s480[:,1:]=real_s480[:,1:]*mask480

            real_ns480=labimage480.clone()
            real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
            real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)

            if not is_clear:
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480+real_ns480,mask480*2.0-1.0)
            else:
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480 * mask480 + real_ns480 * (1 - mask480), mask480 * 2.0 - 1.0)

            fake_B480 = temp_B480.data
            fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
            fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
            fake_B480=fake_B480.data.squeeze(0).cpu()
            fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
            # fake_B480=resize(fake_B480,(480,640,3))

            # print('1.5')
            fake_B480=color.lab2rgb(fake_B480)
            # print('2')

            #replace
            mask[mask>0.5]=1
            mask[mask<=0.5]=0
            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate((mask, mask, mask), axis=-1)
            outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0

            outputimage = img_as_ubyte(outputimage)

            ###
            ### RMSE
            ###
            pred = outputimage
            pred = color.rgb2lab(pred)

            truemask[truemask > 0.5] = 1
            truemask[truemask <= 0.5] = 0
            truemask = np.expand_dims(truemask, axis=2)
            truemask = np.concatenate((truemask, truemask, truemask), axis=-1)

            curr_non_shadow_mse, curr_non_shadow = calculate_rmse(pred, gtimage, ~truemask)
            curr_shadow_mse, curr_shadow = calculate_rmse(pred, gtimage, truemask)
            curr_mse, curr = calculate_rmse(pred, gtimage, np.ones_like(truemask))

            non_shadow_mse += curr_non_shadow_mse
            shadow_mse += curr_shadow_mse
            total_mse += curr_mse

            total_mse_PI += curr_mse/(curr/3)
            total_shadow_mse_PI += curr_shadow_mse/(curr_shadow/3)
            total_non_shadow_mse_PI += curr_non_shadow_mse/(curr_non_shadow/3)

            total_non_shadow += curr_non_shadow
            total_shadow += curr_shadow
            total += curr

        print('test result of epoch', epoch+1)
        print('S: {shadow_rmse:.4f}, NS: {nonshadow_rmse:.4f}, RMSE: {whole_rmse:.4f}'.
              format(shadow_rmse=shadow_mse/(total_shadow/3), nonshadow_rmse=non_shadow_mse/(total_non_shadow/3),
                     whole_rmse=total_mse/(total/3)))
        print('PI, S: {shadow_rmse_PI:.4f}, NS: {nonshadow_rmse_PI:.4f}, RMSE: {whole_rmse_PI:.4f}'.
              format(shadow_rmse_PI=total_shadow_mse_PI/len(gtimage_list),
                     nonshadow_rmse_PI=total_non_shadow_mse_PI/len(gtimage_list),
                     whole_rmse_PI=total_mse_PI/len(gtimage_list)))
        return shadow_mse/(total_shadow/3), total_shadow_mse_PI/len(gtimage_list)

def test_online_best(savename, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, is_clear):
    for key in ['*100*netG_*_best_shadow_rmse.pth', '*100*netG_*_best_shadow_rmse_PI.pth',
                '*150*netG_*_best_shadow_rmse.pth', '*150*netG_*_best_shadow_rmse_PI.pth']:
        paths = glob.glob(os.path.join('ckpt', savename, key))
        if len(paths) == 0:
            continue
        if 'netG_1' in paths[0]:
            g1ckpt = paths[0]
            g2ckpt = paths[1]
        else:
            g1ckpt = paths[1]
            g2ckpt = paths[0]
        print(g1ckpt, g2ckpt)

        netG_1 = Generator_S2F()
        netG_2 = Generator_F2S()
        netG_1.cuda()
        netG_2.cuda()

        netG_1.load_state_dict(torch.load(g1ckpt))
        netG_1.eval()
        netG_2.load_state_dict(torch.load(g2ckpt))
        netG_2.eval()

        test_online(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list,
                    0, is_clear)


def test_online_SRD(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, epoch, is_clear):
    netG_1.eval()
    netG_2.eval()

    idx = 0
    non_shadow_mse = 0
    shadow_mse = 0
    total_mse = 0
    total = 0
    total_shadow = 0
    total_non_shadow =  0

    total_mse_PI = 0
    total_shadow_mse_PI = 0
    total_non_shadow_mse_PI = 0

    # print('1')
    with torch.no_grad():

        for rgbimage_o, labimage_o, mask_o, gtimage_o, truemask_o, img_name in\
                zip(rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list):
            rgbimage = copy.deepcopy(rgbimage_o)
            labimage = copy.deepcopy(labimage_o)
            mask = copy.deepcopy(mask_o)
            gtimage = copy.deepcopy(gtimage_o)
            truemask = copy.deepcopy(truemask_o)

            # labimage480=resize(labimage,(480,640,3))
            labimage480 = copy.deepcopy(labimage)
            labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
            labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
            labimage480=torch.from_numpy(labimage480).float()
            labimage480=labimage480.view(640,840,3)
            labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
            labimage480=labimage480.unsqueeze(0).cuda()


            # mask480=resize(mask,(480,640))
            mask480 = copy.deepcopy(mask)
            mask480 = mask480/255.0
            mask480=torch.from_numpy(mask480).float()
            mask480=mask480.view(640, 840,1)
            mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
            mask480=mask480.unsqueeze(0).cuda()
            zero = torch.zeros_like(mask480)
            one = torch.ones_like(mask480)
            mask480=torch.where(mask480 > 0.5, one, zero)

            real_s480=labimage480.clone()
            real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
            real_s480[:,1:]=real_s480[:,1:]*mask480

            real_ns480=labimage480.clone()
            real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
            real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)

            if not is_clear:
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480+real_ns480,mask480*2.0-1.0)
            else:
                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480 * mask480 + real_ns480 * (1 - mask480), mask480 * 2.0 - 1.0)

            fake_B480 = temp_B480.data
            fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
            fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
            fake_B480=fake_B480.data.squeeze(0).cpu()
            fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
            # fake_B480=resize(fake_B480,(480,640,3))

            # print('1.5')
            fake_B480=color.lab2rgb(fake_B480)
            # print('2')

            #replace

            mask[mask>0.5]=1
            mask[mask<=0.5]=0
            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate((mask, mask, mask), axis=-1)
            outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0

            outputimage = img_as_ubyte(outputimage)

            ###
            ### RMSE
            ###
            pred = outputimage
            pred = color.rgb2lab(pred)

            truemask[truemask > 0.5] = 1
            truemask[truemask <= 0.5] = 0
            truemask = np.expand_dims(truemask, axis=2)
            truemask = np.concatenate((truemask, truemask, truemask), axis=-1)

            curr_non_shadow_mse, curr_non_shadow = calculate_rmse(pred, gtimage, ~truemask)
            curr_shadow_mse, curr_shadow = calculate_rmse(pred, gtimage, truemask)
            curr_mse, curr = calculate_rmse(pred, gtimage, np.ones_like(truemask))

            non_shadow_mse += curr_non_shadow_mse
            shadow_mse += curr_shadow_mse
            total_mse += curr_mse

            total_mse_PI += curr_mse/(curr/3)
            total_shadow_mse_PI += curr_shadow_mse/(curr_shadow/3)
            total_non_shadow_mse_PI += curr_non_shadow_mse/(curr_non_shadow/3)

            total_non_shadow += curr_non_shadow
            total_shadow += curr_shadow
            total += curr

        print('test result of epoch', epoch+1)
        print('S: {shadow_rmse:.4f}, NS: {nonshadow_rmse:.4f}, RMSE: {whole_rmse:.4f}'.
              format(shadow_rmse=shadow_mse/(total_shadow/3), nonshadow_rmse=non_shadow_mse/(total_non_shadow/3),
                     whole_rmse=total_mse/(total/3)))
        print('PI, S: {shadow_rmse_PI:.4f}, NS: {nonshadow_rmse_PI:.4f}, RMSE: {whole_rmse_PI:.4f}'.
              format(shadow_rmse_PI=total_shadow_mse_PI/len(gtimage_list),
                     nonshadow_rmse_PI=total_non_shadow_mse_PI/len(gtimage_list),
                     whole_rmse_PI=total_mse_PI/len(gtimage_list)))
        return shadow_mse/(total_shadow/3), total_shadow_mse_PI/len(gtimage_list)

def test_online_best_SRD(savename, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, is_clear):
    for key in ['*100*netG_*_best_shadow_rmse.pth', '*100*netG_*_best_shadow_rmse_PI.pth',
                '*150*netG_*_best_shadow_rmse.pth', '*150*netG_*_best_shadow_rmse_PI.pth']:
        paths = glob.glob(os.path.join('ckpt', savename, key))
        if len(paths) == 0:
            continue
        if 'netG_1' in paths[0]:
            g1ckpt = paths[0]
            g2ckpt = paths[1]
        else:
            g1ckpt = paths[1]
            g2ckpt = paths[0]
        print(g1ckpt, g2ckpt)

        netG_1 = Generator_S2F()
        netG_2 = Generator_F2S()
        netG_1.cuda()
        netG_2.cuda()

        netG_1.load_state_dict(torch.load(g1ckpt))
        netG_1.eval()
        netG_2.load_state_dict(torch.load(g2ckpt))
        netG_2.eval()

        test_online_SRD(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list,
                    0, is_clear)


def test_online_inCo(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list,
                     truemask_list, gt_list, netEn_sf,
                     epoch):

    netEn_sf.eval()
    netG_1.eval()
    netG_2.eval()

    idx = 0
    non_shadow_mse = 0
    shadow_mse = 0
    total_mse = 0
    total = 0
    total_shadow = 0
    total_non_shadow =  0

    total_mse_PI = 0
    total_shadow_mse_PI = 0
    total_non_shadow_mse_PI = 0

    with torch.no_grad():

        for rgbimage_o, labimage_o, mask_o, gtimage_o, truemask_o, img_name in\
                zip(rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list):
            rgbimage = copy.deepcopy(rgbimage_o)
            labimage = copy.deepcopy(labimage_o)
            mask = copy.deepcopy(mask_o)
            gtimage = copy.deepcopy(gtimage_o)
            truemask = copy.deepcopy(truemask_o)

            # labimage480=resize(labimage,(480,640,3))
            labimage480 = copy.deepcopy(labimage)
            labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
            labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
            labimage480=torch.from_numpy(labimage480).float()
            labimage480=labimage480.view(480,640,3)
            labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
            labimage480=labimage480.unsqueeze(0).cuda()


            # mask480=resize(mask,(480,640))
            mask480 = copy.deepcopy(mask)
            mask480 = mask480/255.0
            mask480=torch.from_numpy(mask480).float()
            mask480=mask480.view(480,640,1)
            mask480=mask480.transpose(0, 1).transpose(0, 2).contiguous()
            mask480=mask480.unsqueeze(0).cuda()
            zero = torch.zeros_like(mask480)
            one = torch.ones_like(mask480)
            mask480=torch.where(mask480 > 0.5, one, zero)

            real_s480=labimage480.clone()
            real_s480[:,0]=(real_s480[:,0]+1.0)*mask480-1.0
            real_s480[:,1:]=real_s480[:,1:]*mask480

            real_ns480=labimage480.clone()
            real_ns480[:,0]=(real_ns480[:,0]+1.0)*(mask480-1.0)*(-1.0)-1.0
            real_ns480[:,1:]=real_ns480[:,1:]*(mask480-1.0)*(-1.0)

            NS_shadow_feature = netEn_sf(real_ns480)
            temp_B480 = netG_1(NS_shadow_feature, real_s480)
            temp_B480 = netG_2(NS_shadow_feature, temp_B480+real_ns480, mask480*2.0-1.0)

            fake_B480 = temp_B480.data
            fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
            fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
            fake_B480=fake_B480.data.squeeze(0).cpu()
            fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
            # fake_B480=resize(fake_B480,(480,640,3))

            fake_B480=color.lab2rgb(fake_B480)

            #replace
            mask[mask>0.5]=1
            mask[mask<=0.5]=0
            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate((mask, mask, mask), axis=-1)
            outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0

            outputimage = img_as_ubyte(outputimage)

            ###
            ### RMSE
            ###
            pred = outputimage
            pred = color.rgb2lab(pred)

            truemask[truemask > 0.5] = 1
            truemask[truemask <= 0.5] = 0
            truemask = np.expand_dims(truemask, axis=2)
            truemask = np.concatenate((truemask, truemask, truemask), axis=-1)

            curr_non_shadow_mse, curr_non_shadow = calculate_rmse(pred, gtimage, ~truemask)
            curr_shadow_mse, curr_shadow = calculate_rmse(pred, gtimage, truemask)
            curr_mse, curr = calculate_rmse(pred, gtimage, np.ones_like(truemask))

            non_shadow_mse += curr_non_shadow_mse
            shadow_mse += curr_shadow_mse
            total_mse += curr_mse

            total_mse_PI += curr_mse/(curr/3)
            total_shadow_mse_PI += curr_shadow_mse/(curr_shadow/3)
            total_non_shadow_mse_PI += curr_non_shadow_mse/(curr_non_shadow/3)

            total_non_shadow += curr_non_shadow
            total_shadow += curr_shadow
            total += curr

        print('test result of epoch', epoch+1)
        print('S: {shadow_rmse:.4f}, NS: {nonshadow_rmse:.4f}, RMSE: {whole_rmse:.4f}'.
              format(shadow_rmse=shadow_mse/(total_shadow/3), nonshadow_rmse=non_shadow_mse/(total_non_shadow/3),
                     whole_rmse=total_mse/(total/3)))
        print('PI, S: {shadow_rmse_PI:.4f}, NS: {nonshadow_rmse_PI:.4f}, RMSE: {whole_rmse_PI:.4f}'.
              format(shadow_rmse_PI=total_shadow_mse_PI/len(gtimage_list),
                     nonshadow_rmse_PI=total_non_shadow_mse_PI/len(gtimage_list),
                     whole_rmse_PI=total_mse_PI/len(gtimage_list)))
        return shadow_mse/(total_shadow/3), total_shadow_mse_PI/len(gtimage_list)

def test_online_best_inCO(savename, rgbimage_list, labimage_list, mask_list,
                          gtimage_list, truemask_list, netEn_sf, gt_list):
    for key in ['*100*netG_*_best_shadow_rmse.pth', '*100*netG_*_best_shadow_rmse_PI.pth',
                '*150*netG_*_best_shadow_rmse.pth', '*150*netG_*_best_shadow_rmse_PI.pth']:
        paths = glob.glob(os.path.join('ckpt', savename, key))
        if len(paths) == 0:
            continue
        if 'netG_1' in paths[0]:
            g1ckpt = paths[0]
            g2ckpt = paths[1]
        else:
            g1ckpt = paths[1]
            g2ckpt = paths[0]
        print(g1ckpt, g2ckpt)

        netG_1 = Generator_S2F()
        netG_2 = Generator_F2S()
        netG_1.cuda()
        netG_2.cuda()

        netG_1.load_state_dict(torch.load(g1ckpt))
        netG_1.eval()
        netG_2.load_state_dict(torch.load(g2ckpt))
        netG_2.eval()

        test_online_inCo(netG_1, netG_2, rgbimage_list, labimage_list, mask_list,
                         gtimage_list, truemask_list, gt_list, netEn_sf, 0)


if __name__ == "__main__":
    rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list = load_online_test_data('SRD')

    savename = '0514-11-Nounaligned'
    # for key in ['*100*netG_*_best_shadow_rmse.pth', '*100*netG_*_best_shadow_rmse_PI.pth',
    #               '*150*netG_*_best_shadow_rmse.pth', '*150*netG_*_best_shadow_rmse_PI.pth']:
    for key in ['*150*netG_*_best_shadow_rmse.pth']:
        paths = glob.glob(os.path.join('ckpt', savename, key))
        if len(paths) == 0:
            continue
        if 'netG_1' in paths[0]:
            g1ckpt = paths[0]
            g2ckpt = paths[1]
        else:
            g1ckpt = paths[1]
            g2ckpt = paths[0]
        print(g1ckpt, g2ckpt)

        netG_1 = Generator_S2F()
        netG_2 = Generator_F2S()
        netG_1.cuda()
        netG_2.cuda()

        netG_1.load_state_dict(torch.load(g1ckpt))
        netG_1.eval()
        netG_2.load_state_dict(torch.load(g2ckpt))
        netG_2.eval()

        test_online_SRD(netG_1, netG_2, rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list, 0, 0)
