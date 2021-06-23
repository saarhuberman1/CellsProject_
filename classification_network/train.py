from sklearn.utils.extmath import softmax
from sklearn import metrics
import random
import torch
import time
import numpy as np
import torch.optim as optim
from torchvision import utils
from torch.autograd import Variable
from classification_network.dataset import CellDataset
from classification_network.dataset_multi_inputs import CellDataset as MultiCellDataset
from classification_network.efficient_fcnn import EfficientNetFCNN
import glob
import copy
import os
import argparse
import logging
import torch.functional as F
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import torch_optimizer
from classification_network.models.sphere_res import StereoSphereRes
from classification_network.models import resnet_v2

# general preferences
use_cuda = torch.cuda.is_available()


def ae_eval_dev_set(args, model, data_loader, criterion, epoch):
    """"""
    model.eval()
    test_losses = AverageMeter()

    logging.info('\n')
    logging.info('##### Evaluating Epoch {} ####'.format(epoch))

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(inputs.clone(), use_cuda)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.update(loss.item(), batch[0].size(0))

    model.train()
    logging.info('Test loss: {:.4f}'.format(test_losses.avg))
    logging.info('\n')
    save_checkpoint(args, epoch, test_losses.avg, 0, model)


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


def save_checkpoint(args, epoch, auc_per_im, auc_per_id, model):
    model_name = '{}_{:.2f}_{:.2f}.pt'.format(epoch, auc_per_im, auc_per_id)
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
    os.makedirs(args.experiment_dir, exist_ok=True)
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


def get_data_loaders(target_size, args):
    if args.multi_inputs_depth > 0:
        train_dataset = MultiCellDataset(args.train_list_path, args.images_root_dir, is_train=True, target_size=target_size,
                                    full_im=args.full_im, gray=args.gray, args=args, multi_input=True)
        test_dataset = MultiCellDataset(args.test_list_path, args.images_root_dir, is_train=False, target_size=target_size,
                                   full_im=args.full_im, gray=args.gray, args=args, multi_input=True)
    else:
        train_dataset = CellDataset(args.train_list_path, args.images_root_dir, is_train=True, target_size=target_size,
                                    full_im=args.full_im, gray=args.gray, args=args)
        test_dataset = CellDataset(args.test_list_path, args.images_root_dir, is_train=False, target_size=target_size,
                                   full_im=args.full_im, gray=args.gray, args=args)
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
            if args.multi_inputs_depth > 0:
                image_names = batch[-1]
            if not args.train_ae:
                targets = tensor_to_gpu(Variable(batch[1]), use_cuda).long()
            else:
                targets = tensor_to_gpu(inputs.clone(), use_cuda)

            # net_input = prepare_input(inputs, is_stereo=args.input_mode)
            optimizer.zero_grad()
            if args.multi_inputs_depth > 0:
                outputs = model(inputs, image_names)
            else:
                outputs = model(inputs)

            loss = criterion(torch.squeeze(outputs), targets)
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            if not args.train_ae:
                acc = classification_accuracy(outputs, targets)
            else:
                acc = 1

            losses.update(loss.item(), batch[0].size(0))
            accuracy.update(acc)

            if (batch_idx % (args.log_interval) == 0 and batch_idx > 0) or (epoch == 0 and batch_idx == 0):
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                             'Total Loss {loss.avg:.4f},'
                             'Acc {acc.avg:.3f}'.format(
                    epoch, batch_idx, len(data_loader['train']),
                    loss=losses, acc=accuracy))

                # reset statistics meter
                losses = AverageMeter()
                accuracy = AverageMeter()

            iter_count += 1


        # epoch statistics
        if epoch > 0 and epoch % args.eval_interval == 0:
            if not args.train_ae:
                eval_dev_set(args, model, data_loader['dev'], criterion, epoch)
            else:
                ae_eval_dev_set(args, model, data_loader['dev'], criterion, epoch)

        scheduler.step()

    return model, iter_count


def eval_dev_set(args, model, data_loader: CellDataset, criterion, epoch):
    """"""
    model.eval()

    num_im = 0
    num_im0 = 0
    num_im1 = 0
    num_im_09 = 0
    num_im_01 = 0

    im_correct_avg = 0

    sm = torch.nn.Softmax(dim=1)

    logging.info('\n')
    logging.info('##### Evaluating Epoch {} ####'.format(epoch))

    score_list = []
    gt_list = []

    correct_avg05 = 0
    im_correct1_avg = 0
    im_correct0_avg = 0
    correct09 = 0
    correct01 = 0
    score_per_id = {}
    gt_per_id = {}
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            inputs = tensor_to_gpu(Variable(batch[0]), use_cuda)
            targets = tensor_to_gpu(Variable(batch[1]), use_cuda).long()
            ids = batch[2].cpu().numpy()
            labels = batch[1].cpu().numpy()

            if args.multi_inputs_depth > 0:
                image_names = batch[-1]
                outputs = model(inputs, image_names)
            else:
                outputs = model(inputs)

            b, c, h, w = outputs.size()
            for i in range(b):
                score = outputs.cpu().numpy()[i, :, 0, 0]
                if ids[i] in score_per_id.keys():
                    score_per_id[ids[i]] += [score]
                    if ids[i] != -1 and labels[i] != gt_per_id[ids[i]]:
                        raise Exception('mismatch in labeles within subject id {}'.format(ids[i]))
                else:
                    score_per_id[ids[i]] = [score]
                    gt_per_id[ids[i]] = labels[i]

            outputs = sm(outputs)
            _, pred = outputs.max(1)

            # image-level pred
            im_avg_score = outputs[:, 1].float().mean((1, 2))

            score_list += [float(i) for i in im_avg_score]
            gt_list += [float(t) for t in targets]

            pred05 = im_avg_score > 0.5
            pred09 = im_avg_score > 0.75
            pred01 = im_avg_score > 0.25

            correct_avg05 += pred05.long().eq(targets).sum().item()

            num_im += b

            im_correct1_avg += (targets * pred05.long().eq(targets).long()).sum().item()
            num_im1 += targets.sum().item()

            im_correct0_avg += ((1 - targets) * pred05.long().eq(targets).long()).sum().item()
            num_im0 += targets.nelement() - targets.sum().item()

            num_im_09 += pred09.sum()
            correct09 += (pred09 * targets).sum().item()

            num_im_01 += (~pred01).sum()
            correct01 += ((~pred01) * (1 - targets)).sum().item()

    model.train()
    num_im_09 = int(num_im_09)
    num_im_01 = int(num_im_01)

    logging.info('total accuracy: {:.3f}'.format(correct_avg05 / num_im))
    logging.info('label 0 accuracy: {:.3f} ({} images)'.format(im_correct0_avg / num_im0, num_im0))
    logging.info('label 1 accuracy: {:.3f} ({} images)'.format(im_correct1_avg / num_im1, num_im1))
    logging.info('high score (top 25%) accuracy: {:.3f} ({} images)'.format(correct09 / num_im_09 if num_im_09 > 0
                                                                            else -1, num_im_09))
    logging.info('low score (bottom 25%) accuracy: {:.3f} ({} images)'.format(correct01 / num_im_01 if num_im_01 > 0
                                                                              else -1, num_im_01))

    # calculate AUC:
    y = np.array(gt_list)
    pred = np.array(score_list)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc_per_image = metrics.auc(fpr, tpr)
    logging.info('AUC per image: {}'.format(auc_per_image))

    total_score_per_id = [softmax(np.expand_dims(np.average(score_list, axis=0), axis=0))[0, 1] for score_list in
                          score_per_id.values()]
    label_per_id = [l for l in gt_per_id.values()]
    y = np.array(label_per_id)
    pred = np.array(total_score_per_id)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc_per_id = metrics.auc(fpr, tpr)
    logging.info('AUC per ID: {}'.format(auc_per_id))
    logging.info('\n')

    save_checkpoint(args, epoch, auc_per_image, auc_per_id, model)


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
    elif args.optimizer == 'RAdam':
        optimizer = torch_optimizer.RAdam(net.parameters(), weight_decay=args.weight_decay)
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


def load_checkpoint(path, model, optimizer=None, classifier=None, train=True):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=train)
    if classifier is not None:
        classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=train)
    return


def main(args):
    """"""
    #

    seed_everything(0)
    start_log(args)

    if 'sphere' in args.model_type:
        target_size = int(args.model_type.split('_')[1])
        sphere_zise = int(args.model_type.split('_')[2])
        net = StereoSphereRes(target_size, input_channels=3, sphereface_size=sphere_zise, train_ae=args.train_ae,
                              multi_inputs_depth=args.multi_inputs_depth)
    elif 'resnet' in args.model_type:
        target_size = int(args.model_type.split('_')[1])
        net = resnet_v2.PreActResNet50()
    else:
        net = EfficientNetFCNN(name=args.model_type)
        target_size = net.get_im_size()
        logging.info('target patch size: {}'.format(target_size))

    if args.resume_from != '':
        load_checkpoint(args.resume_from, net, train=True)
        for param in net.sphereface_blocks[:args.freeze_depth].parameters():
            param.requires_grad = False

    train_loader, test_loader = get_data_loaders(target_size, args)

    data_loader = {
        'train': torch.utils.data.DataLoader(train_loader, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers, drop_last=True),
        'dev': torch.utils.data.DataLoader(test_loader, batch_size=8 if args.full_im else 1,
                                           shuffle=False, num_workers=args.num_workers, drop_last=False)
    }

    # net = EfficientNetFCNN(name='efficientnet-b0')
    logging.info('Model has {} trainable parameters'.format(net.get_num_trainable_params()))

    net = tensor_to_gpu(net, use_cuda)

    optimizer, scheduler = set_optimizer(args, net)

    if not args.train_ae:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    epoch, global_step = 0, 0

    net, iter_count = train(args, net, data_loader, optimizer, scheduler, criterion, epoch, global_step)

    net.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_list_path', type=str, default=r"D:\MSc\cancer\Data\train_lists\toy.txt"),
    # parser.add_argument('--test_list_path', type=str, default=r"D:\MSc\cancer\Data\train_lists\toy.txt"),
    parser.add_argument('--train_list_path', type=str, default=r"D:\MSc\cancer\Data\data_from_web\lists\HE_labeled_balanced_total\fold1_test.txt"),
    parser.add_argument('--test_list_path', type=str, default=r"D:\MSc\cancer\Data\data_from_web\lists\HE_labeled_balanced_total\fold1_test.txt"),
    parser.add_argument('--images_root_dir', type=str, default=r"D:\MSc\cancer\Data\data_from_web\bliss"),
    parser.add_argument('--test_dir_name', type=str, default=None),
    parser.add_argument('--model_dir', type=str,
                        default=r'/home/cdsw/models'),
    parser.add_argument('--lr', type=float, default=1e-2),
    parser.add_argument('--lr_decay_steps', type=int, default=60),
    parser.add_argument('--lr_decay_rate', type=int, default=0.1),
    parser.add_argument('--weight_decay', type=float, default=1e-6),
    parser.add_argument('--momentum', type=float, default=0.9),
    parser.add_argument('--optimizer', type=str, default='SGD'), # also supports 'Adam','RAdam'
    parser.add_argument('--epochs', type=int, default=80),
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--full_im', action='store_true')
    parser.add_argument('--gray', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1),
    parser.add_argument('--experiment_dir', type=str,
                        # default=r'/home/cdsw/models/new_models/efficient-b5-patch-AE-SGD')
                        default=r'D:\MSc\cancer\CellsProject\classification_network\models\test')
    parser.add_argument('--model_type', type=str,
                        # default='efficientnet-b5') # efficientnet model supports b0 to b7 networks.
                        # default='resnet_512') # _512 defines the input resolution (e.g 512x512)
                        default='sphere_512_12') # _512_12 defines the input resolution (e.g 512x512) and resnet depth (e.g 12)
    parser.add_argument('--temp_aug_prob', type=float, default=0.01), # probability for temperature change augmentation
    parser.add_argument('--hue_aug_prob', type=float, default=0.05), # probability for hue change augmentation
    parser.add_argument('--train_ae', action='store_true'),
    parser.add_argument('--multi_inputs_depth', type=int, default=0),
    parser.add_argument('--resume_from', type=str, default=''),
    parser.add_argument('--freeze_depth', type=int, default=5),


    args = parser.parse_args()
    args.gray = False
    args.full_im = True
    args.train_ae = False
    args.multi_inputs_depth = 2
    # args.resume_from = r"C:\Users\amirlivn\Downloads\5_0.02_0.00.pt"

    main(args)

