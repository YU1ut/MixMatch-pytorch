# Supervised baseline for data anomaly classification

import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset,DataLoader
import torch.backends.cudnn as cudnn
from torchvision import models as models
from torchvision import transforms as transforms
from PIL import Image
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
import copy
import shutil
import os
import argparse
import random


# -------------------- Definition of global variables used in the training  process --------------------

parser = argparse.ArgumentParser(description=' Supervised Data Anomaly Detection')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batch size')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--n_labeled', default=28272, type=int, help='Number of labeled data for training')
parser.add_argument('--train_iteration', default=441, type=int, help='Number of iteration per epoch')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='anomaly-timehistory-supervised', help='Directory to save the result')
parser.add_argument('--gamma', default=2, type=float, help='parameter of focal loss')
parser.add_argument('--number_class', default=7, type=int, help='number of class')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Path of dataset for training and test
path_01 = './dataset/time_history_01_120_100/'
path_02 = './dataset/time_history_02_120_100/'
path_01_label = './201201.txt'
path_02_label = './201202.txt'
path_02_label_part = './201202fold.txt'

best_acc = 0


# -------------------- Definition of main, training, validation functions --------------------

# definition of the whole training process
def main():
    torch.cuda.empty_cache()
    global best_acc
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # dataset preprocessing
    print("==> Data preprocessing before training")
    gray_mean = 0.1307
    gray_std = 0.3081
    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=gray_mean, std=gray_std)])
    transform_val = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=gray_mean, std=gray_std)])

    dataset_train, dataset_val, dataset_test = get_anomaly(transform_train=transform_train, transform_val=transform_val)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    # model preparation
    def create_model(ema=False):
        model = WideResNet()
        # model = MyNetwork()
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    train_criterion = nn.CrossEntropyLoss()
    # train_criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    step_schedule = optim.lr_scheduler.StepLR(step_size=32, gamma=0.9, optimizer=optimizer)
    start_epoch = 0

    # Training process record
    title = 'data_anomaly_classification_transfer-learning'
    logger = Logger(os.path.join(args.out,'log.txt'), title=title)
    logger.set_names(['Train Loss', 'Train Acc', 'Valid Loss', 'Valid Acc', 'Test Loss', 'Test Acc'])
    writer = SummaryWriter(args.out)

    # start training
    test_accs = []
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, args.lr))
        train_loss = train(dataloader_train, model, optimizer, step_schedule, train_criterion, epoch, use_cuda)
        _, train_acc = validate(dataloader_train, model,train_criterion, use_cuda, mode='Train State')
        val_loss, val_acc = validate(dataloader_val, model,train_criterion, use_cuda, mode='Valid State')
        test_loss, test_acc = validate(dataloader_test, model, train_criterion, use_cuda, mode='Test State')

        # recording the training index for each epoch
        writer.add_scalar('losses/train_loss', train_loss, epoch)
        writer.add_scalar('losses/valid_loss', val_loss, epoch)
        writer.add_scalar('losses/test_loss', test_loss, epoch)
        writer.add_scalar('accuracy/train_acc', train_acc, epoch)
        writer.add_scalar('accuracy/val_acc', val_acc, epoch)
        writer.add_scalar('accuracy/test_acc', test_acc, epoch)
        logger.append([train_loss, train_acc, val_loss, val_acc, test_loss, test_acc])

        # save the best model parameter
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)
    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


# definition of the training function of each epoch
def train(dataloader_train, model, optimizer, step_schedule, train_criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    bar = Bar('Training', max=args.n_labeled/args.batch_size)

    model.train()
    for batch_idx, data in enumerate(dataloader_train):
        imgs, labels = data
        data_time.update(time.time() - end)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda(non_blocking=True)
        outputs = model(imgs)
        optimizer.zero_grad()
        loss = train_criterion(outputs,labels)
        loss.backward()
        losses.update(loss.item(),imgs.size(0))
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} '.format(
                                                                    batch=batch_idx + 1,
                                                                    size=args.train_iteration,
                                                                    data=data_time.avg,
                                                                    bt=batch_time.avg,
                                                                    loss=losses.avg,
                                                                )
        bar.next()
    bar.finish()
    return losses.avg


# definition of the validation function of each epoch
def validate(dataloader_val, model, criterion, use_cuda, mode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(dataloader_val))
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader_val):
            imgs, labels = data
            data_time.update(time.time() - end)
            if use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda(non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1.item(), imgs.size(0))
            top5.update(prec5.item(), imgs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s  | Loss: {loss:.4f} | ' \
                     'top1: {top1: .4f} | top5: {top5: .4f}'.format(
                                                                    batch=batch_idx + 1,
                                                                    size=len(dataloader_val),
                                                                    data=data_time.avg,
                                                                    bt=batch_time.avg,
                                                                    loss=losses.avg,
                                                                    top1=top1.avg,
                                                                    top5=top5.avg,
                                                                )
            bar.next()
        bar.finish()
    return losses.avg, top1.avg


# --------------- Definition of self-constructed dataset class  ---------------
class Mydataset(Dataset):

    def __init__(self,data,transform,path):
        super().__init__()
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = os.path.join(self.path, self.data[item][0])
        img = Image.open(image_path).convert('L')
        img = self.transform(img)
        label = self.data[item][1]
        return img,label


# Definition of obtaining Training/Validation/Test data index (path + label)
def get_anomaly(transform_train=None, transform_val=None):
    num_class = args.number_class
    f_dataset_train = open(path_01_label, 'r')
    dataset = f_dataset_train.readlines()
    dataset_train = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_class):
                dataset_train.append((words[0], int(words[1])))
            else:
                pass

    f_dataset_val = open(path_02_label_part, 'r')
    dataset = f_dataset_val.readlines()
    dataset_val = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_class):
                dataset_val.append((words[0], int(words[1])))
            else:
                pass

    f_dataset_test = open(path_02_label, 'r')
    dataset = f_dataset_test.readlines()
    dataset_test = []
    for line in dataset:
        line = line.rstrip(' ')
        words = line.split(' ')
        if len(line) > 1:
            if (int(words[1]) >= 0) and (int(words[1]) < num_class):
                dataset_test.append((words[0], int(words[1])))
            else:
                pass

    train_labeled_dataset = Mydataset(dataset_train, transform=transform_train, path=path_01)
    val_dataset = Mydataset(dataset_val, transform=transform_val, path=path_02)
    test_dataset = Mydataset(dataset_test, transform=transform_val, path=path_02)

    return train_labeled_dataset, val_dataset, test_dataset


# -------------------- Definition of Network Model used for training --------------------
class MyNetwork(nn.Module):

    def __init__(self,spp_level=1,number_class=args.number_class):
        super().__init__()
        self.spp_level = spp_level
        self.num_grid = 1
        feature_extractor = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1,3,7)
        self.net = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu
                            , feature_extractor.maxpool, feature_extractor.layer1,
                             feature_extractor.layer2, feature_extractor.layer3, feature_extractor.layer4)
        self.spp_layer = SPPLayer(spp_level)
        self.l1 = nn.Linear(self.num_grid * 512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(256,number_class)

    def forward(self,x):
        x = self.conv1(x)
        x = self.net(x)
        x = self.spp_layer(x)
        x = self.l1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        return x


# Spatial Pyramid Pooling Module for random input image size
class SPPLayer(nn.Module):

    def __init__(self,num_levels,pool_type='maxpool'):
        super(SPPLayer,self).__init__()
        self.num_levels = num_levels
        self.pooltype = pool_type

    def forward(self,x):
        num,channel,height,width = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(height/level),math.ceil(width/level))
            stride = (math.ceil(height/level),math.ceil(width/level))
            padding = (math.floor((kernel_size[0]*level-height+1)/2),math.floor((kernel_size[1]*level-width+1)/2))

            if self.pooltype == 'maxpool':
                tensor = F.max_pool2d(x,kernel_size=kernel_size,stride=stride,padding=padding).view(num,-1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding).view(num, -1)

            if i == 0:
                x_flatten = tensor.view(num,-1)
            else:
                x_flatten = torch.cat((x_flatten,tensor.view(num,-1)),1)

        return x_flatten


# The definition of WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_class=args.number_class, depth=28, widen_factor=2, dropRate=0.0,spplevel=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.spp = SPPLayer(spplevel)
        self.fc = nn.Linear(1792, num_class)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.spp(out)
        out = self.fc(out)
        return out


# -------------------- Definition of loss functions for training --------------------
class FocalLoss(nn.Module):

    def __init__(self, gamma=args.gamma, num_classes=args.number_class, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# checkpoint (model and training parameter save)
def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# -------------------- Main Function Run --------------------
if __name__ == '__main__':
    main()








