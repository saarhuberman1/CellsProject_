import random
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from classification_network.ae_dataset import AECellDataset
from classification_network.auto_encoder.auto_encoder import ERAutoEncoder
import os
import argparse
import logging

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


def save_checkpoint(args, epoch, acc, model):
    model_name = '{}_{:.2f}.pt'.format(epoch, acc)
    # freeze model
    for object in [model]:
        object.eval()

    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.experiment_dir, model_name))

    # unfreeze model
    for object in [model]:
        object.train()

    logging.info('Saved checkpoint: {}'.format(model_name))


def start_log(args):
    logfile = os.path.join(args.experiment_dir, 'log.txt')
    os.makedirs(args.experiment_dir,exist_ok=True)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        handlers=[stream_handler,
                                  logging.FileHandler(filename=logfile)])
    logging.info('*** START ARGS ***')
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))
    logging.info('*** END ARGS ***')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def classification_accuracy(output, targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = output.max(1)
        pred = torch.squeeze(pred)
        correct = pred.eq(targets).sum(0, keepdim=True).item()

        return correct / batch_size


def confusion_matrix(y_true, y_pred, num_classes):
    """
    :param y_true:
    :param y_pred:
    :return:
    """

    CM = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for i in range(len(y_true)):
        x = y_pred[i]
        y = y_true[i]
        if y >= num_classes or y < 0:
            continue
        CM[y][x] += 1

    return CM


def get_data_loaders(args):
    train_path = args.data_root_dir + '/train'
    test_path = args.data_root_dir + '/test'
    # train_dataset = AECellDataset(train_path, is_train=True, target_size=args.target_size)
    # test_dataset = AECellDataset(test_path, is_train=False, target_size=args.target_size)
    train_dataset = AECellDataset(args.train_list_path, is_train=True, target_size=args.target_size)
    test_dataset = AECellDataset(args.test_list_path, is_train=False, target_size=args.target_size)
    return train_dataset, test_dataset


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    m = epoch // 5
    lr = args.lr * (0.1 ** m)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(args, model, data_loader, optimizer, scheduler, criterion, epoch, global_step):
    """"""
    # load mono model for passport id preservation
    min_test_loss = 1000
    test_loss = min_test_loss
    model.train()

    iter_count = global_step

    # init statistics meters
    losses = AverageMeter()
    accuracy = AverageMeter()

    logging.info('Start training...')

    for epoch in range(epoch, args.epochs):
        model.train()
        for batch_idx, batch in enumerate(data_loader['train']):


            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(Variable(batch[1]), use_cuda)

            # net_input = prepare_input(inputs, is_stereo=args.input_mode)
            optimizer.zero_grad()
            outputs = model(inputs)


            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch[0].size(0))


            if (batch_idx % (args.log_interval) == 0 and batch_idx > 0) or (epoch == 0 and batch_idx == 0):
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Total Loss {loss.avg:.4f}'.format(
                    epoch, batch_idx, len(data_loader['train']),
                    loss=losses, acc=accuracy))

                # reset statistics meter
                losses = AverageMeter()
                accuracy = AverageMeter()

            iter_count += 1

        # epoch statistics
        if epoch > 0 and epoch % args.eval_interval == 0:
            test_loss = eval_dev_set(args, model, data_loader['dev'], criterion, epoch, min_test_loss)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
        scheduler.step()

    return model, iter_count


# def eval_dev_set(args, model, data_loader: CellDataset, criterion, epoch):
def eval_dev_set(args, model, data_loader, criterion, epoch, min_test_loss):
    """"""
    model.eval()
    test_losses = AverageMeter()

    logging.info('\n')
    logging.info('##### Evaluating Epoch {} ####'.format(epoch))

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(Variable(batch[1]), use_cuda)
            # inputs = inputs[...,500:1000, 500:1000]
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.update(loss.item(), batch[0].size(0))


    model.train()
    logging.info('Test loss: {:.4f}'.format(test_losses.avg))
    logging.info('\n')
    if test_losses.avg <= min_test_loss:
        logging.info('improved test loss from {:.4f} to {:.4f}, saving model'.format(min_test_loss, test_losses.avg))
        save_checkpoint(args, epoch, test_losses.avg, model)
    return test_losses.avg
#
#
#
# def init_weights(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform(m.weight)
#         nn.init.constant(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         size = m.weight.size()
#         fan_out = size[0]  # number of rows
#         fan_in = size[1]  # number of columns
#         variance = np.sqrt(2.0 / (fan_in + fan_out))
#         m.weight.data.normal_(0.0, variance)


def set_optimizer(args, net):
    modules = [
        {'params': net.parameters(), 'weight_decay': args.weight_decay},
    ]

    if args.optimizer == 'SGD':
        # optimizer = optim.SGD(net.parameters(),
        optimizer = optim.SGD(modules,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    return optimizer, scheduler


# CDR
def get_args_file():
    if os.environ.get('ARGS_FILE') is not None:
        return os.environ.get('ARGS_FILE')
    if os.path.exists('/home/cdsw/train/params.txt'):
        return '/home/cdsw/train/params.txt'
    return None


def log_train_args(args):
    if args.spoof_detector:
        logging.info('TRAINING SPOOF DETECTOR!')
    logging.info('Sphereface Model Size: {}'.format(args.resnet_blocks))
    logging.info('Auxiliary tasks: {}'.format(args.auxiliary))
    logging.info('alpha: {}'.format(args.alpha))
    logging.info('beta: {}'.format(args.beta))
    logging.info('gamma: {}'.format(args.gamma))
    logging.info('lamda: {}'.format(args.lamda))
    logging.info('coordinates mode: {}'.format(args.coord_map_mode))
    logging.info('input mode: {}'.format('Stereo' if args.input_mode else 'Mono'))
    logging.info('mask loss: {}'.format(args.mask_loss))
    logging.info('blur: {}'.format(args.blur))
    logging.info('loss_dropout_prob: {}'.format(args.loss_dropout_prob))
    logging.info('net_dropout_prob: {}'.format(args.net_dropout_prob))
    logging.info('classifier_weight_decay: {}'.format(args.classifier_weight_decay))
    logging.info('coord_normalization_method: {}'.format(args.coord_normalization_method))
    logging.info('pyramid_levels: {}'.format(args.pyramid_levels))
    logging.info('left_plus_depth: {}'.format(args.left_plus_depth))
    logging.info('passport_id: {}'.format(args.passport_id))
    logging.info('reconstruction_depth: {}'.format(args.reconstruction_depth))


def main(args):
    """"""

    seed_everything(0)
    start_log(args)
    net = ERAutoEncoder(in_channels=4)


    train_loader, test_loader = get_data_loaders(args)

    dataset = {}
    dataloader = {}

    data_loader = {
        'train': torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=4, drop_last=True),
        'dev': torch.utils.data.DataLoader(test_loader, batch_size=1,
                                                       shuffle=False, num_workers=4, drop_last=False)
    }


    # net = EfficientNetFCNN(name='efficientnet-b0')
    logging.info('Model has {} trainable parameters'.format(net.get_num_trainable_params()))

    net = tensor_to_gpu(net, use_cuda)


    optimizer, scheduler = set_optimizer(args, net)
    criterion = torch.nn.MSELoss()
    epoch, global_step = 0, 0

    net, iter_count = train(args, net, data_loader, optimizer, scheduler, criterion, epoch, global_step)

    net.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir', type=str, default=r"D:\MSc\cancer\Data\ER"),
    parser.add_argument('--train_list_path', type=str, default=r"D:\omer_git\new\keras_2d_as_master\stereo\ae_train_list.txt"),
    parser.add_argument('--test_list_path', type=str,
                        default=r"D:\omer_git\new\keras_2d_as_master\stereo\ae_test_list.txt"),
    parser.add_argument('--model_dir', type=str,
                        default=r'D:\MSc\cancer\CellsProject\cell_counter_network\find_cells_kernel_5_of_ones_weight1'),
    parser.add_argument('--lr', type=float, default=1e-2),
    parser.add_argument('--lr_decay_steps', type=int, default=20),
    parser.add_argument('--lr_decay_rate', type=int, default=0.1),
    parser.add_argument('--weight_decay', type=float, default=1e-6),
    parser.add_argument('--momentum', type=float, default=0.9),
    parser.add_argument('--optimizer', type=str, default='SGD'),
    parser.add_argument('--epochs', type=int, default=80),
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--full_im', action='store_true')
    parser.add_argument('--gray', action='store_true')
    parser.add_argument('--experiment_dir', type=str,
                        default=r'D:\omer_git\new\keras_2d_as_master\stereo\ae')
    parser.add_argument('--target_size', type=int, default=512)
    args = parser.parse_args()

    main(args)
