from __future__ import print_function
import argparse
import math
import random
import torch
import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from timeit import default_timer as timer
from copy import deepcopy

import models
import loader
import optim
import numpy as np
from utils import *

from measures import get_flat_measure
from experiment_config import EvaluationMetrics, DatasetSubsetType
from logger import WandbLogger


parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
ap('-b', help='Train batch size', type=int, default=128)
ap('-eval-b', help='Val, Test batch size', type=int, default=5000)
ap('-B', help='Max epochs', type=int, default=100)
ap('--lr', help='Learning rate', type=float, default=0.1)
ap('--l2', help='L2', type=float, default=0.0)
ap('-L', help='Langevin iterations', type=int, default=0)
ap('--gamma', help='gamma', type=float, default=1e-4)
ap('--scoping', help='scoping', type=float, default=1e-3)
ap('--noise', help='SGLD noise', type=float, default=1e-4)
ap('-g', help='GPU idx.', type=int, default=0)
ap('-s', help='seed', type=int, default=42)
ap('-epoch-step', help='epoch step to save results', type=int, default=100)
ap('-batch-step', help='batch step to save results', type=int, default=100)
ap('-exp-tag', help='tag of the experiment', type=str, default=None)
ap('-wandb-mode', help='mode of the wandb logger', type=str, default='online')
ap('--lr-step', help='step to apply learning rate decay', type=int, default=60)
ap('--lr-decay', help='Decay factor applied to the learning rate',
    type=float, default=0.2)
ap('--apply-scoping', action='store_true',
    help='weather or not the gamma scooping is applied')
ap('--nesterov', action='store_true',
    help='weather or not nesterov is applied')
opt = vars(parser.parse_args())

th.set_num_threads(2)
opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    opt['g'] = -1
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
    cudnn.benchmark = True
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = WandbLogger(opt['exp_tag'], hps=opt, mode=opt['wandb_mode'])

if 'mnist' in opt['m']:
    opt['dataset'] = 'mnist'
elif 'allcnn' in opt['m']:
    opt['dataset'] = 'cifar10'
else:
    assert False, "Unknown opt['m']: " + opt['m']

train_loader, train_eval_loader, val_loader = getattr(loader, opt['dataset'])(opt)
model = getattr(models, opt['m'])(opt)

# Initial model to compare with the trained model and understand its
# generalization
init_model = deepcopy(model)

criterion = nn.CrossEntropyLoss()

if opt['cuda']:
    print('Cuda is available!!')
    model = model.cuda()
    init_model = init_model.cuda()
    criterion = criterion.cuda()

optimizer = optim.EntropySGD(model.parameters(),
        config = dict(lr=opt['lr'],
                      momentum=0.9,
                      nesterov=opt['nesterov'],
                      weight_decay=opt['l2'],
                      L=opt['L'],
                      eps=opt['noise'],
                      g0=opt['gamma'],
                      g1=opt['scoping']))

# Controls if gamma scoping will be applied
optimizer.state['gamma_scoping'] = opt['apply_scoping']

# Controls the LR scheduling
scheduler = StepLR(optimizer, step_size=opt['lr_step'], gamma=opt['lr_decay'])

print(opt)


# Reference: https://github.com/nitarshan/robust-generalization-measures
def evaluate_complexity_measures(model: nn.Module,
                                 init_model: nn.Module,
                                 device,
                                 seed: int,
                                 dataset_subset_type: DatasetSubsetType,
                                 train_eval_loader: DataLoader,
                                 val_loader: DataLoader,
                                 compute_all_measures: bool = False) -> EvaluationMetrics:
    model.eval()
    init_model.eval()
    data_loader = [train_eval_loader, val_loader][dataset_subset_type]

    print('Before the cross entropy evaluation ...\n')
    loss, acc, num_correct = evaluate_cross_entropy(model,
                                                    device,
                                                    train_eval_loader,
                                                    val_loader,
                                                    dataset_subset_type,
                                                    )
    print('Finished cross entropy evaluation! \n')

    complexities = {}
    if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
        print('Calculating measures ...\n')
        complexities = get_flat_measure(model,
                                        init_model,
                                        data_loader,
                                        acc,
                                        seed)
        print('Measures successfully calculated!!\n')

    return EvaluationMetrics(acc,
                             loss,
                             num_correct,
                             data_loader.n,
                             complexities)


def evaluate_cross_entropy(model,
                           device: int,
                           train_eval_loader: DataLoader,
                           val_loader: DataLoader,
                           dataset_subset_type: DatasetSubsetType):
    model.eval()
    cross_entropy_loss = 0
    num_correct = 0

    data_loader = [train_eval_loader, val_loader][dataset_subset_type]
    num_to_evaluate_on = data_loader.n

    for batch in range(data_loader.total_loops):
        data, target = next(data_loader)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(data)
        cross_entropy = F.cross_entropy(logits, target, reduction='sum')
        cross_entropy_loss += cross_entropy.item()  # sum up batch loss

        pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
        batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
        num_correct += batch_correct.sum()

    cross_entropy_loss /= num_to_evaluate_on
    acc = num_correct.item() / num_to_evaluate_on

    return cross_entropy_loss, acc, num_correct


def train(epoch):
    model.train()

    fs, top1 = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = int(math.ceil(train_loader.n/bsz))

    for bi in range(maxb):
        def helper():
            def feval():
                x, y = next(train_loader)
                if opt['cuda']:
                    x, y = x.cuda(), y.cuda()

                x, y = Variable(x), Variable(y.squeeze())
                bsz = x.size(0)

                optimizer.zero_grad()
                yh = model(x)
                f = criterion.forward(yh, y)
                f.backward()

                prec1, = accuracy(yh.data, y.data, topk=(1,))
                # Percentage errors
                err = 100. - prec1.item()
                return (f.data.item(), err)
            return feval

        f, err = optimizer.step(helper(), model, criterion)

        fs.update(f, bsz)
        top1.update(err, bsz)

        if bi % opt['batch_step'] == 0 and bi != 0:
            print('[%2d][%4d/%4d] Mean Loss: %2.4f Mean Errors: %2.2f%%'%(epoch, bi, maxb, fs.avg, top1.avg))

    scheduler.step()
    logger.log_lr(epoch, DatasetSubsetType.TRAIN, optimizer.param_groups[0]['lr'])
    print('\nLearning rate at this epoch is: %0.9f' % optimizer.param_groups[0]['lr'])

    if epoch % opt['epoch_step'] == 0 or epoch == opt['B'] - 1:
        print(f'Evaluating complexity measures at epoch {epoch}')
        train_eval = evaluate_complexity_measures(model,
                                                  init_model,
                                                  device,
                                                  opt['s'],
                                                  DatasetSubsetType.TRAIN,
                                                  train_eval_loader,
                                                  val_loader,
                                                  compute_all_measures=True)

        val_eval = evaluate_complexity_measures(model,
                                                init_model,
                                                device,
                                                opt['s'],
                                                DatasetSubsetType.TEST,
                                                train_eval_loader,
                                                val_loader,
                                                compute_all_measures=True)
        logger.log_generalization_gap(epoch,
                                      train_eval.acc,
                                      val_eval.acc,
                                      train_eval.avg_loss,
                                      val_eval.avg_loss,
                                      train_eval.all_complexities)

    print('Train: [%2d] %2.4f %2.2f%% [%.2fs]'% (epoch, fs.avg, top1.avg, timer() - ts))
    print()
    logger.log_all_epochs(epoch, DatasetSubsetType.TRAIN, fs.avg, top1.avg)
    logger.log_gamma(epoch, DatasetSubsetType.TRAIN, optimizer.gamma)


def set_dropout(cache = None, p=0):
    if cache is None:
        cache = []
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                cache.append(l.p)
                l.p = p
        return cache
    else:
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                assert len(cache) > 0, 'cache is empty'
                l.p = cache.pop(0)


def dry_feed():
    cache = set_dropout()
    maxb = int(math.ceil(train_loader.n/opt['b']))
    for bi in range(maxb):
        x, y = next(train_loader)
        if opt['cuda']:
            x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            x, y = Variable(x), Variable(y.squeeze())
            yh = model(x)
    set_dropout(cache)


def val(e, data_loader):
    dry_feed()
    model.eval()

    maxb = int(math.ceil(data_loader.n/opt['b']))

    fs, top1 = AverageMeter(), AverageMeter()
    for bi in range(maxb):
        x,y = next(data_loader)
        bsz = x.size(0)

        if opt['cuda']:
            x,y = x.cuda(), y.cuda()

        with torch.no_grad():
            x,y =   Variable(x), Variable(y.squeeze())
            yh = model(x)

            f = criterion.forward(yh, y).data.item()
            prec1, = accuracy(yh.data, y.data, topk=(1,))
            err = 100-prec1.item()

        fs.update(f, bsz)
        top1.update(err, bsz)

    print('Test: [%2d] %2.4f %2.4f%%\n'%(e, fs.avg, top1.avg))
    print()
    logger.log_all_epochs(epoch, DatasetSubsetType.TEST, fs.avg, top1.avg)


for epoch in range(opt['B']):
    train(epoch)
    val(epoch, val_loader)
    torch.cuda.empty_cache()
