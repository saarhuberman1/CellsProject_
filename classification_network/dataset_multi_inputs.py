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
import logging
import torchvision.transforms as transforms


normalize = transforms.Compose([
    transforms.ToTensor(),
    #    transforms.Normalize((0.8998,0.8253,0.9357), (0.1125,0.1751,0.0787)),
    transforms.Normalize((0.9357, 0.8253, 0.8998), (0.0787, 0.1751, 0.1125)),
])


def parse_train_list(list_path, root_images_dir):
    with open(list_path, 'r') as f:
        lines = f.readlines()
    images = []
    labels = []
    ids = []
    for line in lines:
        line = line.strip().split()
        images.append(root_images_dir + line[0])
        labels.append(int(line[1]))
        if len(line) == 3:
            ids.append(int(line[2]))
        else:
            ids.append(-1)
    return images, labels, ids


def white_balance(image):
    if image is not None:
        min_s = 0.65
        max_s = 1.35
        if image.ndim == 2:
            scale = random.uniform(min_s, max_s)
            image *= scale
        else:
            h, w, c = image.shape
            for i in range(c):
                scale = random.uniform(min_s, max_s)
                image[..., i] = image[..., i] * scale
    return image


class CellDataset(Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 train_list_path: str,
                 images_root_dir: str,
                 is_train: bool,
                 target_size: int,
                 full_im=False,
                 gray=False,
                 is_bm=False,
                 args=None,
                 multi_input=False
                 ):
        super(CellDataset, self).__init__()
        self.train_list_path = train_list_path
        self.images_root_dir = images_root_dir
        self.is_train = is_train
        self.target_size = target_size
        self.full_im = full_im
        self.gray = gray
        self.images = []
        self.true_num_images = 0
        self.is_bm = is_bm
        self.args = args
        self.multi_input = multi_input

        self.images, self.labels, self.ids = parse_train_list(self.train_list_path, self.images_root_dir)

        label1 = [i for i in self.labels if i >= 1]
        label0 = [i for i in self.labels if i == 0]

        logging.info('found {} images for {} (label 0: {}, label 1: {})'.format(len(self.images),
                                                                                'training' if self.is_train else 'testing',
                                                                                len(label0), len(label1)))
        logging.info('total of {} ids, {} images per id')


    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __getitem__(self, index: int):
        im_path = self.images[index]
        label = self.labels[index]
        sid = self.ids[index]
        with open(im_path, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            intensity = int(np.random.uniform(0,255,1))
            im = intensity * np.ones((1440,2000,3))
            label = 0
        else:
            try:
                im = cv2.imread(im_path)
                if im is None:
                    print(im_path)
                if self.gray:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
                    im = np.stack([im, im, im], axis=2)
            except:
                intensity = int(np.random.uniform(0,255,1))
                im = intensity * np.ones((1440,2000,3))
                label = 0
        h, w, c = im.shape

        w_margin = (w - 1440) // 2
        im = im[:, w_margin:-w_margin]
        label = 0 if label == 0 else 1

        if self.gray:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
            im = np.stack([im, im, im], axis=2)

        # random crop
        if self.full_im:
            if self.is_train:
                im = cv2.resize(im, dsize=(self.target_size, self.target_size),
                                interpolation=cv2.INTER_AREA)
            else:
                im = cv2.resize(im, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        if self.is_train:
            max_w, max_h, _ = im.shape
            try:
                crop_x = random.randint(0, max_w - self.target_size - 1)
                crop_y = random.randint(0, max_h - self.target_size - 1)
            except:
                crop_x, crop_y = 0,0
            im = im[crop_x:crop_x + self.target_size, crop_y:crop_y + self.target_size]

            # augment texture image:
            im = self.augment(im)

        im = normalize(im / 255)

        if not self.is_bm:
            if self.is_train and not self.multi_input:
                return TF.to_tensor(im.copy()).float(), label
            else:
                return TF.to_tensor(im.copy()).float(), label, sid, im_path
        else:
            return TF.to_tensor(im.copy()).float(), label, sid, im_path

    def augment(self, im):
        if random.uniform(0, 1) < self.args.hue_aug_prob:
            aug = iaa.WithHueAndSaturation([
                iaa.WithChannels(0, iaa.Add((-7, 7))),
                iaa.WithChannels(1, [
                    iaa.Multiply((0.7, 1.3)),
                    iaa.LinearContrast((0.8, 1.2))
                ])
            ], from_colorspace=iaa.CSPACE_BGR)

            im = aug.augment_image(im.astype('uint8'))

        if random.uniform(0, 1) < self.args.temp_aug_prob:
            if random.uniform(0, 1) > 0.5:
                aug = iaa.ChangeColorTemperature((5000, 7000), from_colorspace=iaa.CSPACE_BGR)
            else:
                aug = iaa.ChangeColorTemperature((8000, 1200), from_colorspace=iaa.CSPACE_BGR)
            im = aug.augment_image(im.astype('uint8'))

        # add blur
        im = random_blur(im)

        # add noise
        im = random_noise(im, scale=255, noise_std=3)

        gamma_scale = random.uniform(0.7, 1.7)
        if np.random.uniform(0, 1) < 0.5:
            im = gamma_correction(im, correction=gamma_scale, is_normalized=False)

        if np.random.uniform(0, 1) < 0.0:
            im = white_balance(im)

        # flip image
        if np.random.uniform(0, 1) < 0.5:  # vertical flip
            im = np.flip(im, axis=0)
        if np.random.uniform(0, 1) < 0.5:  # horizontal flip
            im = np.flip(im, axis=1)

        # rotate image
        if np.random.uniform(0, 1) < 0.5:
            angle = random.uniform(-10, 10)
            im = ndimage.rotate(im, angle, reshape=False)

        # crop and resize
        if np.random.uniform(0, 1) < 0.0:
            offset_x = np.random.randint(0, int(self.target_size * 0.2))
            offset_y = np.random.randint(0, int(self.target_size * 0.2))
            crop_im = im[offset_x:, offset_y:]
            im = cv2.resize(crop_im, dsize=(self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        return im.astype('float')


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


