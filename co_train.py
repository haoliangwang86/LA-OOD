import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import backbone_models.vgg as vgg
import backbone_models.resnet as resnet
import backbone_models.densenet as densenet
from sklearn.preprocessing import StandardScaler
from global_settings import *
from utility import load_ood_detector

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--model', default='vgg16', type=str,
                    help='model name')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='training dataset name')
parser.add_argument('--lambda_value', type=float, default=0.001,
                    help='ood loss regularization')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.model == "vgg16":
        batch_size = 128
        if args.dataset == "cifar100":
            model = vgg.vgg16_cifar100()
        else:
            model = vgg.vgg16()

    elif args.model == "resnet34":
        batch_size = 128
        if args.dataset == "cifar100":
            model = resnet.ResNet34_cifar100()
        else:
            model = resnet.ResNet34()

    elif args.model == "densenet100":
        batch_size = 64
        if args.dataset == "cifar100":
            model = densenet.DenseNet100_cifar100()
        else:
            model = densenet.DenseNet100()

    model.cuda()

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    if args.dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.dataset == "cifar100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = co_train_loss

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # prec1 = validate(val_loader, model, criterion)
    # print('Before co-training, Best prec@1 {:.3f}'.format(prec1))

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch in [0, 1, 2, 4]:
            torch.save(
                model.state_dict(),
                f"pre_trained_backbones/{args.model}-{args.dataset}-lambda-{args.lambda_value}-epoch-{epoch+1}.h5")

        print('Best prec@1 {:.3f}'.format(best_prec1))



def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    mean_list, std_list = get_feature_mean_and_std(model, train_loader)
    ood_detectors = load_all_ood_detectors(args.model, args.dataset)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output_list = model.intermediate_forward(input_var)
        loss = criterion(output_list, target_var, ood_detectors, mean_list, std_list)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_list[-1].float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    mean_list, std_list = get_feature_mean_and_std(model, val_loader)
    ood_detectors = load_all_ood_detectors(args.model, args.dataset)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output_list = model.intermediate_forward(input_var)
            loss = criterion(output_list, target_var, ood_detectors, mean_list, std_list)

            output = output_list[-1].float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_feature_mean_and_std(model, train_loader):
    model.eval()
    features = None

    with torch.no_grad():
        for i, (input, _) in enumerate(train_loader):
            input_var = input.cuda()
            if args.half:
                input_var = input_var.half()

            # compute output
            outputs = model.intermediate_forward(input_var)

            # get channel mean
            for i in range(len(outputs)):
                outputs[i] = outputs[i].cpu().numpy()
                if len(outputs[i].shape) == 4:
                    outputs[i] = np.mean(outputs[i], axis=(2, 3))  # batchsize x C

            if features is None:
                features = outputs  # layers x batchsize x C
            else:
                for i in range(len(features)):
                    features[i] = np.vstack((features[i], outputs[i]))  # stack each batch

    mean_list = []
    std_list = []
    for feature in features:
        ss = StandardScaler()
        ss.fit(feature)
        mean_list.append(ss.mean_)
        std_list.append(ss.scale_)

    return mean_list, std_list


def load_all_ood_detectors(model_name, ind_name):
    ood_detectors = []
    layers = []
    if args.model == "vgg16":
        layers = VGG16_LAYERS
    elif args.model == "resnet34":
        layers = RESNET34_LAYERS
    elif args.model == "densenet100":
        layers = DENSENET100_LAYERS

    for layer in layers:
        model = load_ood_detector(model_name, ind_name, layer)
        ood_detectors.append(model)
    print("OOD detectors are loaded.")
    return ood_detectors


def rbf_kernel(feature, support_vector, gamma):
    norm = torch.norm(feature-support_vector, dim=1)
    norm = norm * norm
    result = -gamma * norm
    result = torch.exp(result)
    return result


def prepare_detector_inputs(output, mean, std):
    if len(output.shape) == 4:
        output = torch.mean(output, dim=(2, 3))

    # normalize feature
    for i in range(output.shape[1]):
        output[:, i] = (output[:, i] - mean[i]) / std[i]

    X = output
    Y = torch.ones(len(X))
    return X, Y


def co_train_loss(output_list, target, ood_detectors, mean_list, std_list):
    classification_criterion = nn.CrossEntropyLoss().cuda()
    classification_loss = classification_criterion(output_list[-1], target)

    detector_loss = torch.tensor(0.).cuda()

    if args.model == "vgg16":
        detector_idx = VGG16_LAYERS
    elif args.model == "resnet34":
        detector_idx = random.sample(RESNET34_LAYERS, 10)
    elif args.model == "densenet100":
        detector_idx = random.sample(RESNET34_LAYERS, 10)

    for i in detector_idx:
        detector = ood_detectors[i]
        gamma = detector.gamma
        support_vectors = detector.support_vectors_
        dual_coef = detector.dual_coef_[0]

        gamma = torch.tensor(gamma).cuda()
        support_vectors = torch.tensor(support_vectors).cuda()
        dual_coef = torch.tensor(dual_coef).cuda()

        features, _ = prepare_detector_inputs(output_list[i], mean_list[i], std_list[i])

        current_loss = torch.tensor(0.).cuda()
        sum = torch.tensor(0.).cuda()
        for j in range(len(support_vectors)):
            sum += torch.sum(dual_coef[j] * rbf_kernel(features, support_vectors[j], gamma))

        current_loss += sum
        detector_loss += current_loss

    detector_final_loss = detector_loss / (2 * len(detector_idx))

    print(f"Classification loss: {classification_loss:.4f}")
    print(f"OOD detector loss: {-args.lambda_value * detector_final_loss:.4f}")
    loss = classification_loss - args.lambda_value * detector_final_loss
    return loss


if __name__ == '__main__':
    main()
