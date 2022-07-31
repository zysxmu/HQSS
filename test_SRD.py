import os
import IPython
import argparse
import numpy as np
from skimage import io, color
from skimage.transform import resize
from os.path import exists, join as join_paths

import torch
import torchvision.transforms as transforms

from model import Generator_S2F,Generator_F2S

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_1', type=str, default='ckpt/bs4netG_1_100.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_2', type=str, default='ckpt/bs4netG_2_100.pth', help='A2B generator checkpoint file')
parser.add_argument('--seed', type=int, default=628)
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--savename', type=str, default='')
opt = parser.parse_args()

## ISTD
opt.dataroot_A = './SRD/test/test_A'
opt.im_suf_A = '.jpg'
opt.dataroot_B = './SRD/test/test_B'
opt.im_suf_B = '.jpg'
opt.savename = ''

if torch.cuda.is_available():
    opt.cuda = True

print(opt)

test_448480 = 0
test_480 = 1

netG_1 = Generator_S2F()
netG_2 = Generator_F2S()

if opt.cuda:
    netG_1.cuda()
    netG_2.cuda()

gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

for ee in [1]:
    g1ckpt = './seed628ep150_netG_1_best_shadow_rmse.pth'
    g2ckpt = './seed628ep150_netG_2_best_shadow_rmse.pth'
    savepath = opt.savename + '/best'

    netG_1.load_state_dict(torch.load(g1ckpt))
    netG_1.eval()
    netG_2.load_state_dict(torch.load(g2ckpt))
    netG_2.eval()

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for idx, img_name in enumerate(gt_list):
        # Set model input
        with torch.no_grad():

            if test_480:
                rgbimage=io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))
                rgbimage = resize(rgbimage, (640, 840, 3))
                rgbimage = (rgbimage * 255).astype("uint8")

                labimage = color.rgb2lab(rgbimage)

                labimage480=labimage
                labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
                labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
                labimage480=torch.from_numpy(labimage480).float()
                labimage480=labimage480.view(640,840,3)
                labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
                labimage480=labimage480.unsqueeze(0).cuda()


                mask=io.imread(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))
                mask = resize(mask, (640, 840))
                mask[mask <= 0.5] = 0
                mask[mask > 0.5] = 255
                mask = mask.astype('uint8')


                mask480=mask
                mask480=torch.from_numpy(mask480).float()
                mask480=mask480.view(640,840,1)
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


                temp_B480 = netG_1(real_s480)
                temp_B480 = netG_2(temp_B480+real_ns480,mask480*2.0-1.0)

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
                print(fake_B480.shape, rgbimage.shape)
                outputimage=fake_B480*mask+rgbimage*(mask-1.0)*(-1.0)/255.0
                save_result = join_paths(savepath+'/%s'% (img_name + opt.im_suf_A))
                io.imsave(save_result, outputimage)

            print('Generated images %04d of %04d' % (idx+1, len(gt_list)))
