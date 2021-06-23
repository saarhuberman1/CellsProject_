import os
# from random import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import json
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from scipy import ndimage
from imgaug import augmenters as iaa


class AECellDataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 dataset_path: str,
                 is_train: bool,
                 target_size: int = 256,
                 is_bm=False
                 ):
        super(AECellDataset, self).__init__()
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.target_size = target_size
        self.images = []
        self.true_num_images = 0
        self.is_bm = is_bm
        print('globbing ', self.dataset_path)
        self.images = glob.glob(r'{}/*.jpg'.format(self.dataset_path))
        self.true_num_images += len(self.images)

        for i in [0, 1, 2, 3]:
            imgs = glob.glob(r'{}/{}/*.jpg'.format(self.dataset_path, i))
            self.true_num_images += len(imgs)
            self.images += imgs
            if self.is_train:
                if i == 0:
                    self.images += imgs

        if self.is_train:
            self.images = self.images * 10
        random.shuffle(self.images)
        print('found {} images for {}'.format(self.true_num_images, 'training' if self.is_train else 'testing'))

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __getitem__(self, index: int):
        im_path = self.images[index]
        im = cv2.imread(im_path)
        orig_im = im.copy()
        h, w, c = im.shape

        if not self.is_bm:
            w_margin = (w - 1440) // 2
            im = im[:, w_margin:-w_margin]
            orig_im = orig_im[:, w_margin:-w_margin]

        # random crop
        if self.is_train:
            if random.random() < 0.5:
                angle = random.uniform(-25, 25)
                im = ndimage.rotate(im, angle, reshape=False)
                orig_im = ndimage.rotate(orig_im, angle, reshape=False)

            max_w, max_h, _ = im.shape
            crop_x = random.randint(0, max_w - self.target_size - 1)
            crop_y = random.randint(0, max_h - self.target_size - 1)
            im = im[crop_x:crop_x + self.target_size, crop_y:crop_y + self.target_size]
            orig_im = orig_im[crop_x:crop_x + self.target_size, crop_y:crop_y + self.target_size]
            # augment texture image:
            im, orig_im = self.augment(im, orig_im)

        im = im / 127.5 - 1  # normalize to [-1,1]
        orig_im = orig_im / 127.5 - 1
        if not self.is_bm:
            return TF.to_tensor(im.copy()).float(), TF.to_tensor(orig_im.copy()).float()
        else:
            return TF.to_tensor(im.copy()).float(), TF.to_tensor(orig_im.copy()).float(), im_path


    def augment(self, im, orig_im):
        # add blur
        im = random_blur(im)

        # add noise
        im = random_noise(im, scale=255, noise_std=3)

        gamma_scale = random.uniform(0.7, 1.7)
        if np.random.uniform(0, 1) > 0.5:
            im = gamma_correction(im, correction=gamma_scale, is_normalized=False)

        # # permute colors channels:
        # if np.random.uniform(0, 1) > 0.5:
        #     im = im[...,::-1]

        # flip image
        if random.random() < 0.5:  # vertical flip
            im = np.flip(im, axis=0)
            orig_im = np.flip(orig_im, axis=0)
        if random.random() < 0.5:  # horizontal flip
            im = np.flip(im, axis=1)
            orig_im = np.flip(orig_im, axis=1)

        # temperature
        if random.uniform(0, 1) > 0.0:
            aug = iaa.ChangeColorTemperature((4500, 8000), from_colorspace=iaa.CSPACE_BGR)

            im = aug.augment_image(im.astype('uint8'))

        #        # rotate image
        #        if random.random() < 0.5:
        #            angle = random.uniform(-10, 10)
        #            im = ndimage.rotate(im, angle, reshape=False)
        #            orig_im = ndimage.rotate(orig_im, angle, reshape=False)

        return im, orig_im


def random_noise(image, scale, noise_std=25, type='gauss'):
    def add_noise(noise_type):
        if noise_type == "gauss":
            row, col, ch = image.shape
            std = random.randint(0, noise_std)
            gauss = np.random.normal(0, std, (row, col, ch))
            noisy = image + gauss
            noisy = np.clip(noisy, 0, scale)
            noisy = noisy.astype(image.dtype)
            return noisy
        elif noise_type == "s&p":
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape[:2]]
            out[coords] = scale
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape[:2]]
            out[coords] = 0
            return out
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_type == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape((row, col, ch))
            noisy = image + image * gauss
            return noisy

    if random.uniform(0, 1) < 0.5 and image is not None:
        image = add_noise(type)
    return image


def random_blur(image, blur=5, chance=None, rand_kernel_w=None, rand_kernel_h=None):
    if image is None:
        return image

    def rand_kernel(mode):
        if mode == 'w':
            return random.choice(list(range(3, blur + 1, 2))) if rand_kernel_w is None else rand_kernel_w
        else:
            return random.choice(list(range(3, blur + 1, 2))) if rand_kernel_h is None else rand_kernel_h

    if chance is None:
        chance = random.uniform(0, 1)

    if chance < 0.7:
        return image
    elif chance < 0.85:
        return cv2.blur(image, (rand_kernel('w'), rand_kernel('h')))
    else:
        return cv2.GaussianBlur(image, (rand_kernel('w'), rand_kernel('h')), 0)


def gamma_correction(image, correction, is_normalized=True, scale=255):
    if image is not None:
        if not is_normalized:
            image = image / scale

        if correction is None:
            gamma_scale_red = random.uniform(0.7, 1.7)
            gamma_scale_green = random.uniform(0.85, 1.15)
            gamma_scale_blue = random.uniform(0.7, 1.7)
            image[..., 0] = image[..., 0] ** gamma_scale_blue
            image[..., 1] = image[..., 1] ** gamma_scale_green
            image[..., 2] = image[..., 2] ** gamma_scale_red
        else:
            image = image ** correction

        if not is_normalized:
            image *= scale
    return image


if __name__ == '__main__':
    root_path = r'D:\MSc\cancer\Data\data_from_web\bliss\02-008\HE'
    dataset = AECellDataset(root_path, is_train=True)
    # for im,label in dataset:
    for epoch in range(20):
        for im, label in dataset:
            im = np.transpose(im.cpu().numpy())
            label = np.transpose(label.cpu().numpy())

            im = (127.5 * (im + 1)).astype('uint8')
            label = (127.5 * (label + 1)).astype('uint8')

            cv2.imshow('im', im)
            cv2.imshow('orig', label)
            cv2.waitKey()

        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(im[..., ::-1])
        # plt.subplot(122)
        # plt.title(np.sum(label))
        # plt.imshow(label[...,0])
        # plt.show()
        # flag = 0
