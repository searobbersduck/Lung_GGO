import os

import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
sys.path.append(os.path.join(root, 'external_lib/3D-ResNets-PyTorch/models'))

# from resnet import generate_model
# import ggo_model.GGOModel as M1
# import ggo_model_middle as M2
from ggo_model import GGOModel as M1
from ggo_model_middle import GGOModel as M2

from datasets.datasets import GGO_DS

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import time
import torch.nn.functional as F
from tqdm import tqdm
import math
import warnings

warnings.filterwarnings('ignore')

from sklearn import metrics

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train ggo model')
    parser.add_argument('--arch', default='final')
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--ckpt', default=None)
    return parser.parse_args()


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


def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.kernel_size[2]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            if m.bias is not None:
                m.bias.data.zero_()


def train(train_dataloader, model, criterion, optimizer, epoch, display, phase='train'):
    if phase == 'train':
        model.train()
    else:
        model.eval()
    tot_pred = np.array([], dtype=int)
    tot_label = np.array([], dtype=int)
    tot_prob = np.array([], dtype=np.float32)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, labels) in tqdm(enumerate(train_dataloader)):
        data_time.update(time.time()-end)
        output = model(images.cuda())
        loss = criterion(output, labels.cuda())
        _, pred = torch.max(output, 1)
        
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        pred = pred.cpu().data.numpy()
        labels = labels.numpy()
        tot_pred = np.append(tot_pred, pred)
        tot_label = np.append(tot_label, labels)
        tot_prob = np.append(tot_prob, F.softmax(output).cpu().detach().numpy()[:,1])
        losses.update(loss.data.cpu().numpy(), len(images))
        accuracy.update(np.equal(pred, labels).sum()/len(labels), len(labels))
        if (num_iter+1) % display == 0:
            correct = np.equal(tot_pred, tot_label).sum()/len(tot_pred)
            print_info = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}\tAccuray {accuracy.avg:.4f}'.format(
                epoch, num_iter, len(train_dataloader),batch_time=batch_time, data_time=data_time,
                loss=losses, accuracy=accuracy
            )
            print(print_info)
            logger.append(print_info)
    return accuracy.avg, logger, tot_prob, tot_pred, tot_label

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    
    opts = parse_args()
    opts.epochs = 200
    opts.lr = 1e-3
    opts.cos = True
    # opts.train_batch_size = s
    print(opts)

    n_epochs = opts.epochs
    display = 10
    phase = 'train'

    data_root = None
    if opts.block_size == 32:
        data_root = '/data/medical/hospital/cz/ggo/cz/block_46'
    elif opts.block_size == 64:
        data_root = '/data/medical/hospital/cz/ggo/cz/block_66'
    elif opts.block_size == 128:
        data_root = '/data/medical/hospital/cz/ggo/cz/block_132'
    elif opts.block_size == 256:
        data_root = '/data/medical/hospital/cz/ggo/cz/block_270'

    train_config_file = '/data/medical/hospital/cz/ggo/cz/config/train.txt'
    val_config_file = '/data/medical/hospital/cz/ggo/cz/config/val.txt'
    model_dir = '/data/medical/hospital/cz/ggo/cz/model'
    model_dir = os.path.join(model_dir, opts.arch, '{}'.format(opts.block_size))
    os.makedirs(model_dir, exist_ok=True)
    image_shape = [opts.block_size, opts.block_size, opts.block_size]
    
    if phase == 'train':
        train_ds = GGO_DS(data_root, train_config_file, image_shape, phase='train')
        train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True, num_workers=2, pin_memory=False)

        val_ds = GGO_DS(data_root, val_config_file, image_shape, phase='val')
        val_dataloader = DataLoader(val_ds, batch_size=2, shuffle=False, drop_last=True, num_workers=2, pin_memory=False)
    else:
        test_config_file = '/data/medical/hospital/cz/ggo/cz/config/test.txt'
        test_ds = GGO_DS(data_root, test_config_file, image_shape, phase='test')
        test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=False, drop_last=True, num_workers=2, pin_memory=False)        

    # model = generate_model(18, n_classes=2, n_input_channels=1)
    # initial_cls_weights(model)
    # pretrained = '/home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/LungGGO/checkpoint_0010.pth.tar'
    # pretrained = '/home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/checkpoint_0180.pth.tar'
    pretrained = '/home/zhangwd/code/work/MedCommon_Self_Supervised_Learning/trainer/LungGGO/checkpoint_0280.pth.tar'
    if opts.arch == 'final':
        model = M1(2, pretrained)
    elif opts.arch == 'middle':
        model = M2(2, pretrained)

    # weights = '/fileser/zhangwd/data/hospital/cz/ggo/cz/model/ggo_0200_best_0.724_0.725.pth'
    # weights = '/fileser/zhangwd/data/hospital/cz/ggo/cz/model/ggo_0057_best_0.755_0.720.pth'
    # weights = '/fileser/zhangwd/data/hospital/cz/ggo/cz/model/ggo_0052_best_0.802_0.676.pth'
    if opts.ckpt is not None:
        model.load_state_dict(torch.load(opts.ckpt, map_location='cpu'))

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


    if phase == 'train':
        best_acc = 0.1
        best_auc = 0.1
        for epoch in range(n_epochs):
            adjust_learning_rate(optimizer, epoch, opts)
            print('====> train:\t')
            acc, logger, tot_prob, tot_pred, tot_label = train(train_dataloader, torch.nn.DataParallel(model.cuda()), criterion, optimizer, epoch, display, phase='train')
            print('train acc:\t{:.3f}'.format(acc))
            # print(tot_pred)
            # print(tot_label)
            print('====> validate:\t')
            acc, logger, tot_prob, tot_pred, tot_label = train(val_dataloader, torch.nn.DataParallel(model.cuda()), criterion, optimizer, epoch, display, phase='val')
            # print(tot_pred)
            # print(tot_label)
            fpr, tpr, thresholds = metrics.roc_curve(tot_label, tot_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            print('val acc:\t{:.3f}\tauc:\t{:.3f}'.format(acc, auc))
            # if acc > best_acc:
            if auc > best_auc:
                if np.all(tot_pred == 1) or np.all(tot_pred == 0):
                    continue
                print('\ncurrent best accuracy is: {}\n'.format(acc))
                best_acc = acc
                best_auc = auc
                saved_model_name = os.path.join(model_dir, 'ggo_{:04d}_best_{:.3f}_{:.3f}.pth'.format(epoch, acc, auc))
                torch.save(model.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))
    else:
        acc, logger, tot_prob, tot_pred, tot_label = train(test_dataloader, torch.nn.DataParallel(model.cuda()), criterion, optimizer, 0, display, phase='test')
        fpr, tpr, thresholds = metrics.roc_curve(tot_label, tot_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(auc)


if __name__ == '__main__':
    main()

