import argparse
import math
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

import models
import loader
import optim
from utils import AverageMeter, accuracy
from measures import get_flat_measure
from experiment_config import EvaluationMetrics, DatasetSubsetType
from logger import WandbLogger
from utils import check_models

# Keeping the code as similar as the original one
# due to reproduction purpose.
# Added the evaluation of the complexity measures

parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
ap('-b', help='Train batch size', type=int, default=100)
ap('-eval-b', help='Val, Test batch size', type=int, default=5000)
ap('-B', help='Max epochs', type=int, default=100)
ap('-lr', help='Learning rate', type=float, default=0.1)
ap('-l2', help='L2', type=float, default=0.0)
ap('-L', help='Langevin iterations', type=int, default=0)
ap('-gamma', help='gamma', type=float, default=1e-4)
ap('-scoping', help='scoping', type=float, default=1e-3)
ap('-noise', help='SGLD noise', type=float, default=1e-4)
ap('-g', help='GPU idx.', type=int, default=0)
ap('-s', help='seed', type=int, default=42)
ap('-epoch-step', help='epoch step to save results', type=int, default=20)
ap('-batch-step', help='batch step to save results', type=int, default=100)
ap('-exp-tag', help='tag of the experiment', type=str, default=None)
ap('-wandb-mode', help='mode of the wandb logger', type=str, default='online')
ap('-lr-step', help='step to apply learning rate decay', type=int, default=60)
ap('-lr-decay', help='Decay factor applied to the learning rate',
    type=float, default=0.2)
ap('-apply-scoping', action='store_true',
    help='whether or not the gamma scoping is applied')
ap('-nesterov', action='store_true',
    help='whether or not nesterov is applied')
ap('-momentum', help='whether or not apply momentum on the optimizer',
    type=float, default=0)
ap('-calculate', help='whether or not calculate complexity measures',
    action='store_true')
ap('-min-loss', help='minimum loss to be reached on training',
    type=float, default=0.1)
opt = vars(parser.parse_args())

th.set_num_threads(2)
opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
    # Not ideal for reproducibility, but keeping the format from the
    # original code
    cudnn.benchmark = True
    # Reference:
    # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4
    th.cuda.empty_cache()

# Set seeds for reproducibility
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])

device = th.device('cuda' if opt['cuda'] else 'cpu')
opt['device'] = device

# Set the dataset
if 'mnist' in opt['m']:
    opt['dataset'] = 'mnist'
elif 'allcnn' in opt['m']:
    opt['dataset'] = 'cifar10'
else:
    assert False, "Unknown opt['m']: " + opt['m']

train_loader, train_eval_loader, val_loader = \
    getattr(loader, opt['dataset'])(opt)

# Set the network based on the dataset
model = getattr(models, opt['m'])(opt)

# Initial model
# The initial model will be compared with the trained model
# Used by the complexity measures
init_model = deepcopy(model)

criterion = nn.CrossEntropyLoss()

if opt['cuda']:
    model = model.cuda()
    init_model = init_model.cuda()
    criterion = criterion.cuda()

# Set the optimizer
optimizer = optim.EntropySGD(model.parameters(),
                             config=dict(lr=opt['lr'],
                                         momentum=opt['momentum'],
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

# Set logger and time events with Wandb
logger = WandbLogger(opt['exp_tag'], hps=opt, mode=opt['wandb_mode'])
start = th.cuda.Event(enable_timing=True)
end = th.cuda.Event(enable_timing=True)


# Reference: https://github.com/nitarshan/robust-generalization-measures
def evaluate_complexity_measures(model,
                                 init_model,
                                 device,
                                 epoch,
                                 seed,
                                 dataset_subset_type,
                                 train_eval_loader,
                                 val_loader,
                                 compute_all_measures):
    model.eval()
    init_model.eval()
    data_loader = [train_eval_loader, val_loader][dataset_subset_type]

    # Evaluates on the whole training set and with the current model
    loss, acc = evaluate_cross_entropy(model,
                                       epoch,
                                       device,
                                       train_eval_loader,
                                       val_loader,
                                       dataset_subset_type)

    complexities = {}
    if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
        print('Calculating measures...\n')
        complexities = get_flat_measure(model,
                                        init_model,
                                        data_loader,
                                        acc,
                                        seed)
        print('Measures successfully calculated!!\n')

    return EvaluationMetrics(acc,
                             loss,
                             data_loader.n,
                             complexities)


def evaluate_cross_entropy(model,
                           epoch,
                           device,
                           train_eval_loader,
                           val_loader,
                           dataset_subset_type):
    model.eval()

    data_loader = [train_eval_loader, val_loader][dataset_subset_type]
    bsz = data_loader.b
    # Make sure that the loop starts on the beginning
    data_loader.sidx = 0
    data_loader.train = False
    loss, acc = AverageMeter(), AverageMeter()

    total_loops = int(data_loader.n / bsz)

    for _ in range(total_loops):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        with th.no_grad():
            data = Variable(data)
            target = Variable(target.squeeze())
            yh = model(data)
            f = criterion.forward(yh, target).data.item()
            # List of size k=1 with the percentage of correct
            # top_1 value
            prec1 = accuracy(yh.data, target.data, topk=(1,))

        loss.update(f)
        acc.update(prec1[0].item() / 100, bsz)

    cross_entropy_loss = loss.avg

    # Average accuracy over batches
    avg_acc = acc.avg

    logger.log_batch_correctness(epoch,
                                 'eval-ce/' + dataset_subset_type.name.lower(),
                                 acc.count)

    return cross_entropy_loss, avg_acc


def train(epoch, found_stop_epoch):
    model.train()

    fs, top1 = AverageMeter(), AverageMeter()

    # The floor function guarantees that we do not
    # repeat samples over a given epoch
    # If the number of samples is not divisible
    # by the batch size, we leave some samples
    # out of the loop
    maxb = int(math.floor(train_loader.n / train_loader.b))
    bsz = train_loader.b

    start.record()
    for bi in range(maxb):
        # Closure to be passed to the optimizer
        def helper():
            def feval():
                x, y = next(train_loader)
                x, y = x.to(device), y.to(device)
                x = Variable(x)
                y = Variable(y.squeeze())

                optimizer.zero_grad()
                yh = model(x)
                # Computes the crossentropy loss
                # nn.NLLLoss(reduction='mean')(nn.LogSoftmax()(yh), y)
                f = criterion.forward(yh, y)
                f.backward()

                # Computes the number of correct samples over
                # the batch
                prec1 = accuracy(yh.data, y.data, topk=(1,))
                return (f.data.item(), prec1[0].item())
            return feval

        f, acc = optimizer.step(helper(), model, criterion)

        # Average loss over the bacthes
        fs.update(f)
        # Average of the percentage of correct values over batches
        # (10%, 20%) -> 15%
        top1.update(acc, bsz)

        if bi % opt['batch_step'] == 0 and bi != 0:
            print(f'[{epoch}][{bi} / {maxb}],'
                  f'Mean Loss: {np.round(fs.avg, 6)},'
                  f' Perc. top1 accuracy: {np.round(top1.avg, 4)}')

    scheduler.step()
    end.record()
    # Explanation about the ordering:
    # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/9
    th.cuda.synchronize()

    # Log the training time by epoch
    logger.log_time(epoch, start.elapsed_time(end))

    # Log the learning rate
    logger.log_lr(epoch,
                  DatasetSubsetType.TRAIN,
                  optimizer.param_groups[0]['lr'])

    logger.log_batch_correctness(epoch, 'train', top1.count)

    # Check if the complexity measure will be calculated
    # considering two criteria: achievement of the
    # epoch step or achievement of the minimum loss
    evaluate_first_op = epoch % opt['epoch_step'] == 0 \
        or epoch == opt['B'] - 1
    evaluate_second_op = not found_stop_epoch and \
        np.round(fs.avg, 2) < opt['min_loss']

    if evaluate_second_op:
        msg = f'The learning rate {np.round(fs.avg, 2)} was reached.'
        f' on epoch {epoch}.'
        print(msg)
        found_stop_epoch = True

    logger.log_stop_criteria(epoch, np.int(found_stop_epoch))

    # Evaluate complexity if necessary
    if (evaluate_first_op or evaluate_second_op) and opt['calculate']:
        msg = f'Evaluating complexity measures at epoch {epoch}.'
        print(msg)
        model_before = deepcopy(model)
        train_eval = evaluate_complexity_measures(model,
                                                  init_model,
                                                  device,
                                                  epoch,
                                                  opt['s'],
                                                  DatasetSubsetType.TRAIN,
                                                  train_eval_loader,
                                                  val_loader,
                                                  compute_all_measures=True)

        val_eval = evaluate_complexity_measures(model,
                                                init_model,
                                                device,
                                                epoch,
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
        model_after = model
        result = check_models(model_before, model_after)
        if not result:
            msg = 'Something is wrong in the complexity measures!'
            print(msg)
            raise Exception('The models are being manipulated wrong!')
        else:
            msg = 'Model is manipulated right inside the complexity measures!'
            print(msg)

    print(f'Train: [{epoch}]: Loss: {np.round(fs.avg, 6)},'
          f'Perc. top1 accuracy: {np.round(top1.avg, 4)}\n')

    logger.log_all_epochs(epoch,
                          DatasetSubsetType.TRAIN,
                          fs.avg,
                          top1.avg)

    logger.log_optim_params(epoch,
                            DatasetSubsetType.TRAIN,
                            optimizer.gamma,
                            optimizer.langevin_lr,
                            optimizer.alpha,
                            optimizer.momentum,
                            int(optimizer.nesterov))

    return found_stop_epoch


def set_dropout(cache=None, p=0):
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
    """
    Dry feed for warming up the gpu
    making sure it is not on stand by mode
    """
    cache = set_dropout()
    maxb = int(math.floor(train_loader.n / train_loader.b))
    for bi in range(maxb):
        x, y = next(train_loader)
        if opt['cuda']:
            x, y = x.cuda(), y.cuda()
        with th.no_grad():
            x, y = Variable(x), Variable(y.squeeze())
            yh = model(x)
    set_dropout(cache)


def val(e, data_loader):
    dry_feed()
    model.eval()

    # Make sure to cover all samples from the beginning
    data_loader.sidx = 0
    data_loader.train = False

    bsz = data_loader.b
    maxb = int(math.floor(data_loader.n / bsz))

    fs, top1 = AverageMeter(), AverageMeter()
    for _ in range(maxb):
        x, y = next(data_loader)
        x, y = x.to(device), y.to(device)

        with th.no_grad():
            x = Variable(x)
            y = Variable(y.squeeze())
            yh = model(x)

            f = criterion.forward(yh, y).data.item()
            prec1 = accuracy(yh.data, y.data, topk=(1,))

        fs.update(f)
        top1.update(prec1[0].item(), bsz)

    print(f'Test: [{e}], Loss: {np.round(fs.avg, 6)}, '
          f' Perc. top1 accuracy: {np.round(top1.avg, 4)}\n')

    logger.log_all_epochs(epoch,
                          DatasetSubsetType.TEST,
                          fs.avg,
                          top1.avg)

    logger.log_batch_correctness(epoch, 'val', top1.count)


# Controls in which epoch a minimum required learning rate was found.
# Currently, the value is just logged.
found_stop_epoch = False
for epoch in range(opt['B']):
    stopping = train(epoch, found_stop_epoch)
    found_stop_epoch = stopping
    val(epoch, val_loader)
