import cv2
import torch
import numpy as np
from torch.autograd import Variable
from classification_network.dataset import CellDataset
from classification_network.auto_encoder.ae_dataset import AECellDataset
from classification_network.auto_encoder.auto_encoder import ERAutoEncoder

import os
import argparse

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

def eval_dev_set(args, model, data_loader: CellDataset, target_root):
    """"""
    # model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            paths = batch[2]
            outputs = model(inputs)
            b, c, h, w = outputs.size()
            for i in range(b):
                ae_im = outputs[i]
                ae_im = ae_im.detach().permute(1, 2, 0).cpu().numpy()
                name = os.path.basename(paths[i])
                path = r'{}/{}'.format(target_root, name)
                cv2.imwrite(path, (np.clip(127.5 * (ae_im + 1)), 0, 255).astype('uint8'))


def main(args):
    print('processing original images: ')
    net = ERAutoEncoder()
    test_path = args.data_root_dir + '/02_008_PDL1'
    target_path = args.data_root_dir + '/PDL1_AE'
    os.makedirs(target_path, exist_ok=True)
    test_dataset = AECellDataset(test_path, is_train=False, is_bm=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8,
                                              drop_last=False)

    print('loading model')
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)


    net = tensor_to_gpu(net, use_cuda)
    net.eval()
    eval_dev_set(args, net, test_loader, target_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=r"D:\MSc\cancer\Data"),
    parser.add_argument('--ae_model_path', type=str,
                        default=r"D:\MSc\cancer\CellsProject\AE_model\models\auto_encoder\02-008_HE_auto_encoder_fold1_no_rot\75_0.01.pt"),
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
