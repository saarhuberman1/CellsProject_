import numpy as np
from sklearn import metrics
import random
import torch
import time
import numpy as np
import torch.optim as optim
from torchvision import utils
from torch.autograd import Variable
from classification_network.dataset import CellDataset
# from classification_network.dataset import CellDataset
from classification_network.efficient_fcnn import EfficientNetFCNN
import glob
import copy
import os
import argparse
import logging
import torch.functional as F
import matplotlib.pyplot as plt

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


def eval_dev_set(args, model, data_loader):
    """"""
    model.eval()
    num_correct = 0
    num_samples = 0
    num_correct0 = 0
    num_correct1 = 0
    num_samples0 = 0
    num_samples1 = 0

    im_correct = 0
    im_correct_avg = 0
    num_im = 0
    im_correct0 = 0
    im_correct1 = 0
    im_correct0_avg = 0
    im_correct1_avg = 0
    num_im0 = 0
    num_im1 = 0

    sm = torch.nn.Softmax(dim=1)
    results = []
    results_avg = []
    gt = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(Variable(batch[1]), use_cuda).long()

            outputs = model(inputs)

            b, c, h, w = outputs.size()

            outputs = sm(outputs)
            _, pred = outputs.max(1)
            #            print('outputs shape = ', outputs[:,1].shape)
            if args.full_im:
                pred = torch.unsqueeze(pred, -1)
                flag = 0

            # image-level pred
            num_pixels = pred.size(1) * pred.size(2)
            im_pred = pred.sum((1, 2)) > (num_pixels - pred.sum((1, 2)))
            im_pred_avg = outputs[:, 1].float().mean((1, 2)) > 0.5
            im_correct += im_pred.long().eq(targets).sum().item()
            im_correct_avg += im_pred_avg.long().eq(targets).sum().item()
            num_im += im_pred.nelement()

            im_correct1 += (targets * im_pred.long().eq(targets).long()).sum().item()
            im_correct1_avg += (targets * im_pred_avg.long().eq(targets).long()).sum().item()
            num_im1 += targets.sum().item()

            im_correct0 += ((1 - targets) * im_pred.long().eq(targets).long()).sum().item()
            im_correct0_avg += ((1 - targets) * im_pred_avg.long().eq(targets).long()).sum().item()
            num_im0 += im_pred.nelement() - targets.sum().item()

            results.append(im_pred.detach().cpu().numpy())
            results_avg.append(im_pred.detach().cpu().numpy())

            gt.append(targets.detach().cpu().numpy())

            # patch-level pred
            targets_per_pixel = torch.transpose(targets.expand((h, w, b)), 0, -1)
            num_correct += pred.eq(targets_per_pixel).sum().item()
            num_samples += targets_per_pixel.nelement()

            num_samples1 += targets_per_pixel.sum().item()
            num_samples0 += targets_per_pixel.nelement() - targets_per_pixel.sum().item()

            num_correct1 += (targets_per_pixel * pred.eq(targets_per_pixel).long()).sum().item()
            num_correct0 += ((1 - targets_per_pixel) * pred.eq(targets_per_pixel).long()).sum().item()

    model.train()
    print('\n')

    print('patch-level accuracy: {:.3f}'.format(num_correct / num_samples))
    print('patch-level 0 accuracy: {:.3f}'.format(num_correct0 / num_samples0))
    print('patch-level 1 accuracy: {:.3f}'.format(num_correct1 / num_samples1))
    print('\n')
    print('image-level accuracy: {:.3f}'.format(im_correct / num_im))
    print('image-level avg accuracy: {:.3f}'.format(im_correct_avg / num_im))
    print('image-level 0 accuracy: {:.3f}'.format(im_correct0 / num_im0))
    print('image-level 1 accuracy: {:.3f}'.format(im_correct1 / num_im1))
    print('image-level 0 avg accuracy: {:.3f}'.format(im_correct0_avg / num_im0))
    print('image-level 1 avg accuracy: {:.3f}'.format(im_correct1_avg / num_im1))
    print('\n')
    y = np.squeeze(np.array(gt))
    print(y.shape)
    pred = np.array(results_avg)
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred)
    print('AUC avg score: ', metrics.auc(fpr, tpr))
    print('\n\n\n')


def main(args):
    net = EfficientNetFCNN(name=args.model_type)
    target_size = net.get_im_size()
    print('target patch size: {}'.format(target_size))
    test_path = args.data_root_dir + '/test'
    test_dataset = CellDataset(test_path, is_train=False, target_size=target_size, full_im=args.full_im,
                               gray=args.gray)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                              drop_last=False)
    print('loading model')
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    net = tensor_to_gpu(net, use_cuda)
    net.eval()
    res = eval_dev_set(args, net, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=r"/home/cdsw/images/ER"),
    parser.add_argument('--model_path', type=str,
                        #                        default=r"/home/cdsw/models/efficient-b5-full_im_gray/13_28.58.pt"),
                        #                        default=r"/home/cdsw/models/efficient-b5-full_im/6_29.35.pt"),
                        default=r"/home/cdsw/models/efficient-b6-patch/7_0.95.pt"),
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--full_im', action='store_true')
    parser.add_argument('--gray', action='store_true')
    parser.add_argument('--model_type', type=str,
                        #                        default='efficientnet-b5')
                        default='efficientnet-b6')
    args = parser.parse_args()

    args.full_im = True
    args.full_im = False
    args.gray = True
    args.gray = False

    main(args)
