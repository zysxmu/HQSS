import glob
import random
import os
import sys

import IPython
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch
import scipy
import pickle


# 这个默认是用膨胀核大小为50的
class ImageDataset_decouple():
    def __init__(self, root='', mode='train'):
        self.root = root
        self.mode = mode

        if 'SRD' in self.root:
            self.shadowdir = os.path.join(root, '%s/train_A' % mode) + '/'
            self.maskdir = os.path.join(root, '%s/train_B' % mode) + '/'
            self.SD = sorted(glob.glob(self.shadowdir + '*jpg'))
            self.MD = sorted(glob.glob(self.maskdir + '*jpg'))
        else:
            self.shadowdir = os.path.join(root, '%s/train_A' % mode) + '/'
            self.maskdir = os.path.join(root, '%s/train_B' % mode) + '/'

            self.SD = sorted(glob.glob(self.shadowdir + '*png'))
            self.MD = sorted(glob.glob(self.maskdir + '*png'))


        lower = 0.8
        upper = 1.2
        if 'SRD' in self.root:
            if os.path.exists('pickles/SRD_map0.81.2.pickle'):
                with open('pickles/SRD_map0.81.2.pickle', 'rb') as f:
                    self.map = pickle.load(f)
                assert len(self.map) == 2680
            else:
                print('wrong!')
                print('wrong!')
                print('wrong!')
                sys.exit()
        else:
            if os.path.exists('pickles/map0.81.2.pickle'):
                with open('pickles/map0.81.2.pickle', 'rb') as f:
                    self.map = pickle.load(f)
                assert len(self.map) == 1330
            else:
                print('wrong!')
                print('wrong!')
                print('wrong!')
                sys.exit()

    def find_suit_mask(self, mname, smask):

        if 'SRD' in self.shadowdir and 'train_B' in self.maskdir:
            mname = os.path.basename(mname)
            mname = os.path.join(self.root, '%s/o_mask' % self.mode, mname)
            self.map[mname] = []

        if len(self.map[mname]) > 0:
            newmname = self.map[mname][random.randint(0, len(self.map[mname]) - 1)]
            newm = io.imread(newmname)

            if 'SRD' in self.root:
                newm = resize(newm, (640, 840))
                newm[newm <= 0.5] = 0
                newm[newm > 0.5] = 255
                newm = newm.astype('uint8')

            newnmask = ~newm
            newsmask = ~newnmask
            newmask = newsmask - (newsmask & smask)

            assert newmname != mname
            assert (((np.sum(newmask) / 255) / (np.sum(smask) / 255))) >= 0.8 and \
                   (((np.sum(newmask) / 255) / (np.sum(smask) / 255))) <= 1.2

        else:
            i = 0
            # print(mname)
            newmname = os.path.join(self.maskdir, self.MD[random.randint(0, len(self.SD) - 1)])
            newm = io.imread(newmname)

            if 'SRD' in self.root:
                newm = resize(newm, (640, 840))
                newm[newm <= 0.5] = 0
                newm[newm > 0.5] = 255
                newm = newm.astype('uint8')

            newnmask = ~newm
            newsmask = ~newnmask
            newmask = newsmask - (newsmask & smask)

            while newmname == mname or (((np.sum(newmask) / 255) / (np.sum(smask) / 255))) < 0.05:
                newmname = os.path.join(self.maskdir, self.MD[random.randint(0, len(self.SD) - 1)])
                newm = io.imread(newmname)

                if 'SRD' in self.root:
                    newm = resize(newm, (640, 840))
                    newm[newm <= 0.5] = 0
                    newm[newm > 0.5] = 255
                    newm = newm.astype('uint8')


                newnmask = ~newm
                newsmask = ~newnmask
                newmask = newsmask - (newsmask & smask)

                i += 1
                if i == 100:
                    break

            assert newmname != mname
        return newmask

    def __getitem__(self, index):
        # generate non-shadow region
        #         ISTD dataset image size 480*640

        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k = random.randint(0, 100)


        S1_sname = self.SD[index]
        S1_mname = self.MD[index]

        S1 = io.imread(S1_sname)
        M_S1 = io.imread(S1_mname)

        if 'SRD' in self.root:
            S1 = resize(S1, (640, 840))

            M_S1 = resize(M_S1, (640, 840))
            M_S1[M_S1 <= 0.5] = 0
            M_S1[M_S1 > 0.5] = 255
            M_S1 = M_S1.astype('uint8')

        M_NS1 = self.find_suit_mask(S1_mname, M_S1)

        ###
        imdilate = 50
        M_NS1_dil = scipy.ndimage.binary_dilation(M_NS1, iterations=imdilate).astype(np.uint8) * 255

        M_S1 = np.tile(M_S1[..., None], [1, 1, 3])
        M_NS1 = np.tile(M_NS1[..., None], [1, 1, 3])
        M_NS1_dil = np.tile(M_NS1_dil[..., None], [1, 1, 3])

        NS1 = color.rgb2lab(S1 * M_NS1.astype('bool'))
        NS1 = resize(NS1, (448, 448, 3))
        NS1 = NS1[i:i + 400, j:j + 400, :]
        if k > 50:
            NS1 = np.fliplr(NS1)
        NS1[:, :, 0] = np.asarray(NS1[:, :, 0]) / 50.0 - 1.0
        NS1[:, :, 1:] = 2.0 * (np.asarray(NS1[:, :, 1:]) + 128.0) / 255.0 - 1.0
        NS1 = torch.from_numpy(NS1.copy()).float()
        NS1 = NS1.view(400, 400, 3)
        NS1 = NS1.transpose(0, 1).transpose(0, 2).contiguous()

        ###

        real_NS1_nsrs = color.rgb2lab(S1 * (~M_NS1).astype('bool'))
        real_NS1_nsrs = resize(real_NS1_nsrs, (448, 448, 3))
        real_NS1_nsrs = real_NS1_nsrs[i:i + 400, j:j + 400, :]
        if k > 50:
            real_NS1_nsrs = np.fliplr(real_NS1_nsrs)
        real_NS1_nsrs[:, :, 0] = np.asarray(real_NS1_nsrs[:, :, 0]) / 50.0 - 1.0
        real_NS1_nsrs[:, :, 1:] = 2.0 * (np.asarray(real_NS1_nsrs[:, :, 1:]) + 128.0) / 255.0 - 1.0
        real_NS1_nsrs = torch.from_numpy(real_NS1_nsrs.copy()).float()
        real_NS1_nsrs = real_NS1_nsrs.view(400, 400, 3)
        real_NS1_nsrs = real_NS1_nsrs.transpose(0, 1).transpose(0, 2).contiguous()
        ###

        real_ns_1 = color.rgb2lab(S1)
        real_ns_1 = resize(real_ns_1, (448, 448, 3))
        real_ns_1 = real_ns_1[i:i + 400, j:j + 400, :]
        if k > 50:
            real_ns_1 = np.fliplr(real_ns_1)
        real_ns_1[:, :, 0] = np.asarray(real_ns_1[:, :, 0]) / 50.0 - 1.0
        real_ns_1[:, :, 1:] = 2.0 * (np.asarray(real_ns_1[:, :, 1:]) + 128.0) / 255.0 - 1.0
        real_ns_1 = torch.from_numpy(real_ns_1.copy()).float()
        real_ns_1 = real_ns_1.view(400, 400, 3)
        real_ns_1 = real_ns_1.transpose(0, 1).transpose(0, 2).contiguous()

        ###
        M_NS1_dil = resize(M_NS1_dil[:, :, 0], (448, 448))
        M_NS1_dil[M_NS1_dil > 0.5] = 1
        M_NS1_dil[M_NS1_dil <= 0.5] = 0
        M_NS1_dil = M_NS1_dil[i:i + 400, j:j + 400]
        if k > 50:
            M_NS1_dil = np.fliplr(M_NS1_dil)
        M_NS1_dil = torch.from_numpy(M_NS1_dil.copy()).float()
        M_NS1_dil = M_NS1_dil.view(400, 400, 1)
        M_NS1_dil = M_NS1_dil.transpose(0, 1).transpose(0, 2).contiguous()


        M_NS1 = resize(M_NS1[:, :, 0], (448, 448))
        M_NS1[M_NS1 > 0.5] = 1
        M_NS1[M_NS1 <= 0.5] = 0
        M_NS1 = M_NS1[i:i + 400, j:j + 400]
        if k > 50:
            M_NS1 = np.fliplr(M_NS1)
        M_NS1 = torch.from_numpy(M_NS1.copy()).float()
        M_NS1 = M_NS1.view(400, 400, 1)
        M_NS1 = M_NS1.transpose(0, 1).transpose(0, 2).contiguous()
        ###

        S1 = color.rgb2lab(S1 * M_S1.astype('bool'))
        S1 = resize(S1, (448, 448, 3))
        S1 = S1[i:i + 400, j:j + 400, :]
        if k > 50:
            S1 = np.fliplr(S1)
        S1[:, :, 0] = np.asarray(S1[:, :, 0]) / 50.0 - 1.0
        S1[:, :, 1:] = 2.0 * (np.asarray(S1[:, :, 1:]) + 128.0) / 255.0 - 1.0
        S1 = torch.from_numpy(S1.copy()).float()
        S1 = S1.view(400, 400, 3)
        S1 = S1.transpose(0, 1).transpose(0, 2).contiguous()

        M_S1 = resize(M_S1[:, :, 0], (448, 448))
        M_S1[M_S1 > 0.5] = 1
        M_S1[M_S1 <= 0.5] = 0
        M_S1 = M_S1[i:i + 400, j:j + 400]
        if k > 50:
            M_S1 = np.fliplr(M_S1)
        M_S1 = torch.from_numpy(M_S1.copy()).float()
        M_S1 = M_S1.view(400, 400, 1)
        M_S1 = M_S1.transpose(0, 1).transpose(0, 2).contiguous()

        return {'S1': S1, 'M_S1': M_S1,
                'M_NS1': M_NS1, 'NS1': NS1,
                'S1_name': S1_sname, 'real_NS1_nsrs': real_NS1_nsrs,
                'real_ns_1': real_ns_1,
                'M_NS1_dil': M_NS1_dil}

    def __len__(self):
        return len(self.SD)

