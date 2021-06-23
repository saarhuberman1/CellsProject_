import pickle
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
# from classification_network.efficient_fcnn import EfficientNetFCNN
import glob
import copy
import os
import argparse
import logging
import torch.functional as F
import matplotlib.pyplot as plt
import json
from classification_network.models import resnet_v2
from sklearn import metrics

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
    res_dict = {}
    sm = torch.nn.Softmax(dim=1)
    score_list = []
    gt_list = []
    is_correct_list = []
    res_dict = {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='processing receptors images'):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            inputs = inputs.float()
            targets = batch[1]
            sid = batch[2]
            im_path = batch[3][0]
            tma = im_path.split('/')[-3]
            rec = im_path.split('/')[-2]
            outputs, features = model(inputs)
            features = features.cpu().numpy().flatten()
            outputs_sm = sm(outputs)
            im_avg_score = outputs[:, :].float().mean((2, 3)).cpu().numpy()
            im_sm_avg_score = outputs_sm[:, 1].float().mean((1, 2)).cpu().numpy()

            score_list.append(im_sm_avg_score[0])
            gt_list.append(targets.cpu().numpy()[0])

            #            outputs = outputs.cpu().numpy()
            #            outputs_sm = outputs_sm.cpu().numpy()
            is_correct = int(int(im_sm_avg_score > 0.5) == targets.cpu().numpy()[0])
            is_correct_list.append(is_correct)
            # logging.info('{},{},{},{},{},{},{},{}'.format(im_path, tma, rec,
            #                                               sid.cpu().numpy()[0],
            #                                               targets.cpu().numpy()[0],
            #                                               #                                                 outputs[0,:,0,0],
            #                                               #                                                 outputs_sm[0,1,0,0],is_correct))
            #                                               im_avg_score[0],
            #                                               im_sm_avg_score[0], is_correct))
            log_str = '{},{},{},{},{},{},{},{},'.format(im_path, tma, rec,
                                                        sid.cpu().numpy()[0],
                                                        targets.cpu().numpy()[0],
                                                        #                                                 outputs[0,:,0,0],
                                                        #                                                 outputs_sm[0,1,0,0],is_correct))
                                                        im_avg_score[0],
                                                        im_sm_avg_score[0], is_correct)
            log_str += ','.join([str(f) for f in list(features)])
            logging.info(log_str)

            res_dict[im_path] = {'im_name': im_path, 'features': np.squeeze(features), 'output': im_avg_score[0],
                                 'id': sid.cpu().numpy()[0], 'gt_label': targets.cpu().numpy()[0]}

    # calculate AUC:
    y = np.array(gt_list)
    pred = np.array(score_list)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc_per_image = metrics.auc(fpr, tpr)
    print('AUC per image: {}'.format(auc_per_image))
    print('ACC: {:.2f}'.format(np.average(is_correct_list)))

    print('dumping results to file..')
    picklefile = os.path.join(os.path.dirname(args.model_path),
                              'score_per_image_{}_{}.pkl'.format(args.log_name, os.path.basename(args.model_path)))
    with open(picklefile, 'wb') as f:
        pickle.dump(res_dict, f)

    model.train()


def start_log(args):
    logfile = os.path.join(os.path.dirname(args.model_path),
                           'score_per_image_{}_{}.csv'.format(args.log_name, os.path.basename(args.model_path)))
    print(logfile)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        handlers=[stream_handler,
                                  logging.FileHandler(filename=logfile)])
    logging.info('im,tma,receptor,id,gt label,score (before softmax),score (after softmax),is_correct')


def main(args):
    start_log(args)
    target_size = 512
    if 'resnet' in args.model_path:
        net = resnet_v2.PreActResNet50()
    elif 'sphere' in args.model_path:
        net = StereoSphereRes(input_size=target_size, input_channels=3, sphereface_size=12, bm=True)
    else:
        raise Exception
        # net = EfficientNetFCNN('efficientnet-b5')
        # target_size = net.get_im_size()
    print('target patch size: {}'.format(target_size))
    test_dataset = CellDataset(args.test_list_path, args.images_root_dir, is_train=False, target_size=target_size,
                               is_bm=True, log=False, args=args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                              drop_last=False)

    print('loading model')
    #    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    net = tensor_to_gpu(net, use_cuda)
    net.eval()
    eval_dev_set(args, net, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root_dir', type=str, default=r"\\ger\ec\proj\ha\RSG\PersonDataCollection1\users\amir\cancer\data\data_from_web\bliss"),
    parser.add_argument('--model_path', type=str,
                        default=r"D:\MSc\cancer\Data\data_from_web\backup\5.4\models\HE_PDL1\new_folds"
                                r"\fold1_sphere_512_RAdam\110_0.89_0.94.pt"),

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--test_list_path', type=str,
                        default=r"D:\MSc\cancer\Data\data_from_web\lists\PDL1_labeled\new_folds_balanced"
                                r"\test_set.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/PDL1/new_folds_balanced/test_set.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/PDL1/new_folds_balanced/fold1_test.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/PDL1/fold1_test.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/PD-1/fold3_test.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/labeled_receptors/fold1_test.txt"),
    #    parser.add_argument('--test_list_path', type=str, default=r"/home/cdsw/lists/unlabeled_receptors/all_hic.txt"),
    parser.add_argument('--log_name', type=str, default=r"fold1_test-set"),
    args = parser.parse_args()
    args.full_im = True
    main(args)
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--images_root_dir', type=str, default=r"D:\MSc\cancer\Data\data_from_web\bliss"),
#     parser.add_argument('--model_path', type=str,
#                         default=r"D:\MSc\cancer\Models\models\REC\fold1_sphere_512_12_RAdam\20_0.98_nan.pt"),
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--test_list_path', type=str, default=r"D:\MSc\cancer\Data\data_from_web\lists\PDL1_labeled\new_folds_balanced\fold1_test.txt")
#     parser.add_argument('--log_name', type=str, default=r"debug"),
#
#     args = parser.parse_args()
#     main(args)
