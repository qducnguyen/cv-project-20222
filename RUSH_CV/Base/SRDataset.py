import random
import logging

import cv2
import numpy as np
import skimage.color as sc


import torch
from torch.utils.data import Dataset



class SRDataset(Dataset):
    """
    Base class of super-resolution dataset.
    """

    def __init__(self, 
                 data_np_path, 
                 patch_size, 
                 is_train, 
                 is_pre_scale, 
                 scale):

        self.IMG_EXTENSIONS = ['.png']
        self.data_npy_path = data_np_path
        self.patch_size = patch_size
        self.is_train = is_train
        self.scale = scale
        self.is_pre_scale = is_pre_scale

        self.init_dataset()

    def init_dataset(self):
        """
        Running when starting fitting the model
        """
        self.data_npy = np.load(self.data_npy_path)

    def pre_scale_img(self, img, img_tar, scale):
        if img_tar is None:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.resize(img, img_tar.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

    def get_patch(self, *args, patch_size, scale):
        ih, iw = args[0].shape[:2]

        
        tp = patch_size  # target patch (HR)
        ip = tp // scale  # input patch (LR)

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        lr_result = args[0][iy:iy + ip, ix:ix + ip, :]

        if self.is_pre_scale:
            lr_result = self.pre_scale_img(lr_result, None, self.scale)
              
        ret = [
            lr_result,
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]  # results

        return ret

    def augment(self, *args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

            return img

        return [_augment(a) for a in args]
    
    def _get_patch(self, img_in, img_tar):
        patch_size = self.patch_size
        scale = self.scale

        if self.is_train:
            img_in, img_tar = self.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            
            img_in, img_tar = self.augment(img_in, img_tar)

        else:
            if self.is_pre_scale:
                img_in = self.pre_scale_img(img_in, img_tar, self.scale)

            else:
                ih, iw = img_in.shape[:2]
                img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
            
        return img_in, img_tar


    def default_loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR) 
    
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
    

    def set_channel(self, *args, n_channels=3):
    
        def _set_channel(img):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if n_channels == 1 and c == 3:
                img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
            elif n_channels == 3 and c == 1:
                img = np.concatenate([img] * n_channels, 2)

            return img

        return [_set_channel(a) for a in args]


    def np2Tensor(self, *args, rgb_range=1):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255)

            return tensor

        return [_np2Tensor(a) for a in args]


    def __getitem__(self, idx):
        Y_path, X_path = self.data_npy[idx]

        X = self.default_loader(X_path)
        Y = self.default_loader(Y_path) 

        X_patch, Y_patch = self._get_patch(X, Y)
        X_channel, Y_channel = self.set_channel(X_patch, Y_patch)
        X, Y = self.np2Tensor(X_channel, Y_channel)

        return idx, X, Y
    

    def __len__(self):
        return self.data_npy.shape[0]

  