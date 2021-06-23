import torch
from torch.autograd import Variable
from classification_network.auto_encoder.ae_dataset import AECellDataset
from classification_network.auto_encoder.auto_encoder import ERAutoEncoder
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

def eval_dev_set(args, model, data_loader: CellDataset, criterion=None, epoch=None):
    """"""
    # model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(Variable(batch[1]), use_cuda).long()
            outputs = model(inputs)


    model.train()
    print('\n')

def main(args):
    net = ERAutoEncoder()
    target_size = 256
    print('target patch size: {}'.format(target_size))
    test_path = args.data_root_dir + '/test'
    test_dataset = AECellDataset(test_path, is_train=False, target_size=target_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8,
                                              drop_last=False)

    print('loading model')
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)


    net = tensor_to_gpu(net, use_cuda)
    net.eval()
    eval_dev_set(args, net, test_loader)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=r"D:\MSc\cancer\Data\ER"),
    parser.add_argument('--model_path', type=str,
                        default=r"D:\MSc\cancer\CellsProject\classification_network\models\efficient-b6-patch\7_0.95.pt"),
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)
