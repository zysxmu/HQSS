import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch
import scipy
import pickle
import tqdm
from multiprocessing import Process


def fun1(index, lower, upper):

    # root = './ISTD_Dataset'
    root = './SRD'
    mode = 'train'
    shadowdir = os.path.join(root, '%s/train_A' % mode) + '/'
    maskdir = os.path.join(root, '%s/o_mask' % mode) + '/'

    MD = os.listdir(maskdir)
    length = len(MD) // 16

    left, right = index * length, min((index + 1) * length, len(MD))
    if index == 15:
        right = len(MD)
    print('start', index, 'left', left, 'right', right)



    map_dict = {}
    for i in tqdm.tqdm(range(left, right)):
        mname = os.path.join(maskdir, MD[i])
        m = io.imread(mname)

        # for SRD
        m = resize(m, (640, 840))
        m[m <= 0.5] = 0
        m[m > 0.5] = 255
        m = m.astype('uint8')


        nmask = ~m
        smask = ~nmask
        map_dict[mname] = []

        for j in range(len(MD)):
            if j == i:
                continue
            newmname = os.path.join(maskdir, MD[j])
            newm = io.imread(newmname)

            # for SRD
            newm = resize(newm, (640, 840))
            newm[newm <= 0.5] = 0
            newm[newm > 0.5] = 255
            newm = newm.astype('uint8')

            newnmask = ~newm
            newsmask = ~newnmask
            newmask = newsmask - (newsmask & smask)
            if newmname == mname or \
                    (((np.sum(newmask) / 255) / (np.sum(smask) / 255))) < lower or \
                    (((np.sum(newmask) / 255) / (np.sum(smask) / 255))) > upper:
                continue
            map_dict[mname].append(newmname)
        if len(map_dict[mname]) == 0:
            print(mname)
        print(mname, len(map_dict[mname]))
    with open('SRD_map' + str(lower) + str(upper) + str(index) + '.pickle', 'wb') as f:
        pickle.dump(map_dict, f)


lower = 0.8
upper = 1.2
process_list = []
for i in range(16):
    p = Process(target=fun1, args=(i, lower, upper, ))
    p.start()
    process_list.append(p)

for i in process_list:
    p.join()


with open('SRD_map' + str(lower) + str(upper) + str(0) + '.pickle', 'rb') as f:
    a = pickle.load(f)
for i in range(1, 16):
    with open('SRD_map' + str(lower) + str(upper) + str(i) + '.pickle', 'rb') as f:
        b = pickle.load(f)
    a.update(b)
with open('SRD_map' + str(lower) + str(upper) + '.pickle', 'wb') as f:
    pickle.dump(a, f)

print('end')
