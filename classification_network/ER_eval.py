from tqdm import tqdm
import torchvision.transforms.functional as TF
import cv2
import random
import torch
import time
import numpy as np
import torch.optim as optim
from torchvision import utils
from torch.autograd import Variable
from classification_network.dataset import CellDataset
from classification_network.models.sphere_res import StereoSphereRes
import glob
import copy
import os
import argparse
import logging
import torch.functional as F
import matplotlib.pyplot as plt
import json

# general preferences
use_cuda = torch.cuda.is_available()


def tensor_to_gpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor

def tensor_to_cpu(tensor, is_cuda):
    if is_cuda:
        return tensor.cpu()
    else:
        return tensor

def eval_dev_set(args, model, imgs):
    """"""
    model.eval()
    res_dict = {}
    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for im_path in tqdm(imgs,desc='processing receptors images'):
            # print(im_path)
            im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

            h, w, c = im.shape
            w_margin = (w - 1440) // 2
            im = im[:, w_margin:-w_margin]

            im = im / 127.5 - 1
            im = TF.to_tensor(im.copy()).float()
            im = torch.unsqueeze(im,0)
            outputs = model(im)
            outputs = sm(outputs)

            im_avg_score = outputs[:, 1].float().mean((1, 2)).cpu().numpy()[0]
            json_path = im_path.replace('.jpg', '.json')
            key = 'fold1_score'
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    res_dict = json.load(f)
                if key in res_dict.keys():
                    print(f'key {key} already exists!')
                    continue
                else:
                    res_dict[key] = float(im_avg_score)
            else:
                res_dict = {key : float(im_avg_score)}

            with open(json_path, 'w') as f:
                json.dump(res_dict, f)

    model.train()
    print('\n')
    # with open()


def main(args):
    imgs = glob.glob(r'{}\**\*.jpg'.format(args.data_root_dir), recursive=True)
    imgs = [i for i in imgs if '02-008' not in i and '01-011' not in i and 'HE' not in i]
    print(f'found {len(imgs)} images to process')
    net = StereoSphereRes()
    target_size = 512
    print('target patch size: {}'.format(target_size))
    # test_dataset = CellDataset(args.train_list_path, args.images_root_dir, is_train=False, target_size=target_size,
    #                             is_bm=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
    #                                           drop_last=False)
    #
    print('loading model')
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)


    net = tensor_to_gpu(net, use_cuda)
    eval_dev_set(args, net, imgs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=r"D:\MSc\cancer\Data\data_from_web\bliss"),
    parser.add_argument('--model_path', type=str,
                        default=r"D:\MSc\cancer\Models\models\REC\fold1_sphere_512_12_RAdam\20_0.98_nan.pt"),
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
