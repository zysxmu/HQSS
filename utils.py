import os
import sys
import tqdm
import time
import random
import datetime
import numpy as np

from skimage import io, color
from skimage.transform import resize
from skimage.filters import threshold_otsu

import torch
import torch.nn as nn
from torch.autograd import Variable
# from visdom import Visdom
import torchvision.transforms as transforms


to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)


class QueueMask_llab():
    def __init__(self, length):
        self.max_length = length
        self.queue = []
        self.queue_L = []

    def insert(self, mask,mask_L):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)
        if self.queue_L.__len__() >= self.max_length:
            self.queue_L.pop(0)

        self.queue.append(mask)
        self.queue_L.append(mask_L)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        temp=np.random.randint(0, self.queue.__len__())
        return self.queue[temp],self.queue_L[temp]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1],self.queue_L[self.queue.__len__()-1]


class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]


def mask_generator(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask
    
def cyclemask_generator(shadow, shadow_free):
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu()))

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    L=L*0.1
    mask = torch.tensor(np.float32(diff <= L)).unsqueeze(0).unsqueeze(0).cuda() #0:shadow, 1.0:non-shadow
    mask.requires_grad=False
    return mask

def mask_generator_lab_lab(shadow, shadow_free):
    im_f=shadow_free.data
    im_f[:,0]=50.0*(im_f[:,0]+1.0)
    im_f[:,1:]=255.0*(im_f[:,1:]+1.0)/2.0-128.0
    im_f=im_f.data.squeeze(0).cpu()
    im_f=im_f.transpose(0, 2).transpose(0, 1).contiguous().numpy()
    im_f=color.lab2rgb(im_f.astype('int8'))
    im_f=color.rgb2gray(im_f)
    
    im_s=shadow.data
    im_s[:,0]=50.0*(im_s[:,0]+1.0)
    im_s[:,1:]=255.0*(im_s[:,1:]+1.0)/2.0-128.0
    im_s=im_s.data.squeeze(0).cpu()
    im_s=im_s.transpose(0, 2).transpose(0, 1).contiguous().numpy()
    im_s=color.lab2rgb(im_s.astype('int8'))
    im_s=color.rgb2gray(im_s)

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask

def mask_generator_lab(shadow, shadow_free):
    im_f = to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5).cpu())
    im_s = to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5).cpu())

    diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0).cuda() #-1.0:non-shadow, 1.0:shadow
    mask.requires_grad = False

    return mask

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)

        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class cyclemaskloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,fake_B,real_A,mask):
        mask=(1.0-mask)/2.0
        mask=mask.repeat(1,3,1,1)
        mask.requires_grad=False
        return torch.mean(torch.pow((torch.mul(fake_B,mask)-torch.mul(real_A,mask)), 2))


def vis(name, data, ismask=False):
    tmp = data.detach().cpu()

    if not ismask:
        tmp[:, 0] = 50.0 * (tmp[:, 0] + 1.0)
        tmp[:, 1:] = 255.0 * (tmp[:, 1:] + 1.0) / 2.0 - 128.0
        tmp = tmp.data.squeeze(0).cpu()
        tmp = tmp.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        tmp = color.lab2rgb(tmp)
    else:
        tmp = tmp.data.squeeze(0).cpu()
        tmp = tmp.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        
    io.imsave(name + '.png', tmp)

def load_online_test_data(opt):
    '''
    for test
    '''
    if 'SRD' in opt.dataroot:
        print('load online test images!')
        ## SRD
        dataroot_A = './SRD/test/test_A'
        im_suf_A = '.jpg'
        dataroot_B = './SRD/test/test_B'
        dataroot_gtimage = './SRD/test/test_C'
        dataroot_truemask = './SRD/test/test_B'
        im_suf_B = '.jpg'
    else:
        print('load online test images!')
        ## ISTD
        dataroot_A = './ISTD_Dataset/test/test_A'
        im_suf_A = '.png'
        dataroot_B = './G2R-dataset/BDRAR/test_A_mask_istd_6/'
        dataroot_gtimage = './ISTD_adjusted/test_C_fixed_official'
        dataroot_truemask = './ISTD_Dataset/test/test_B'
        im_suf_B = '.png'

    rgbimage_list = []
    labimage_list = []
    mask_list = []
    gtimage_list = []
    truemask_list = []

    gt_list = [os.path.splitext(f)[0] for f in os.listdir(dataroot_A) if f.endswith(im_suf_A)]

    for img_name in tqdm.tqdm(gt_list):
        rgbimage = io.imread(os.path.join(dataroot_A, img_name + im_suf_A))
        labimage = color.rgb2lab(io.imread(os.path.join(dataroot_A, img_name + im_suf_A)))
        mask = io.imread(os.path.join(dataroot_B, img_name + im_suf_B))

        gtimage = io.imread(os.path.join(dataroot_gtimage, img_name + im_suf_A))
        gtimage = color.rgb2lab(gtimage)

        truemask = io.imread(os.path.join(dataroot_truemask, img_name + im_suf_A))
        if 'SRD' in opt.dataroot:
            rgbimage = resize(rgbimage, (640, 840))
            labimage = resize(labimage, (640, 840))
            gtimage = resize(gtimage, (640, 840))

            rgbimage = (rgbimage * 255).astype("uint8")

            mask = resize(mask, (640, 840))
            mask[mask <= 0.5] = 0
            mask[mask > 0.5] = 255
            mask = mask.astype('uint8')

            truemask = resize(truemask, (640, 840))
            truemask[truemask <= 0.5] = 0
            truemask[truemask > 0.5] = 255
            truemask = truemask.astype('uint8')

        rgbimage_list.append(rgbimage)
        labimage_list.append(labimage)
        mask_list.append(mask)
        gtimage_list.append(gtimage)
        truemask_list.append(truemask)

    print('load online test images end!')
    return rgbimage_list, labimage_list, mask_list, gtimage_list, truemask_list, gt_list

def savename(opt):

    opt.savename = opt.savename
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    opt.log_path = os.path.join('' + opt.savename, 'seed%dep%d' % (opt.seed, opt.n_epochs) + '.txt')

    if not os.path.exists('' + opt.savename):
        os.mkdir('' + opt.savename)

    if torch.cuda.is_available():
        opt.cuda = True
