import argparse
import math
import random
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import models
import loader
import optim
from utils import AverageMeter, accuracy
from measures import calculate_flatness_measures
from experiment_config import EvaluationMetrics, DatasetSubsetType
from logger import WandbLogger
from utils import check_models

# Keeping the code as similar as the original one due to reproduction purpose.
# Added the evaluation of the complexity measures.

parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-b', help='mini-batch for training and validation', type=int, default=100)
ap('-eval-b', help='mini-batch for complexity measures',
    type=int, default=5000)
ap('-B', help='number of epochs', type=int, default=100)
ap('-lr', help='learning rate of outer loop', type=float, default=0.1)
ap('-l2', help='L2', type=float, default=0.0)
ap('-L', help='langevin iterations', type=int, default=0)
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
ap('-lr-decay', help='decay factor applied to the learning rate',
    type=float, default=0.2)
ap('-apply-scoping', action='store_true',
    help='whether or not the gamma scoping is applied')
ap('-nesterov', action='store_true',
    help='whether or not nesterov is applied')
ap('-momentum', help='whether or not apply momentum on the optimizer',
    type=float, default=0)
ap('-calculate', help='whether or not calculate complexity measures',
    action='store_true')
ap('-deterministic', help='whether or not use deterministic mode in torch',
    action='store_true')

opt = vars(parser.parse_args())

# Number set by the reference code
th.set_num_threads(2)

# Device management
opt['cuda'] = th.cuda.is_available()
if opt['cuda']:
    # Selected device
    th.cuda.device(opt['g'])
    # Assign to device variable
    device = th.device(f"cuda:{opt['g']}")
    # Reproducibility management
    # (Benchmark False is deterministic)
    cudnn.benchmark = not opt['deterministic']
    th.backends.cudnn.deterministic = opt['deterministic']
else:
    device = th.device('cpu')

# Reproducibility management
random.seed(opt['s'])
np.random.seed(opt['s'])
# Seed the RNG for all devices (both CPU and CUDA)
th.manual_seed(opt['s'])
th.use_deterministic_algorithms(opt['deterministic'])

# Set the dataset and the model: cifar10 and allcnn
train_loader, train_eval_loader, val_loader = getattr(loader,
                                                      'cifar10')(opt)
model = getattr(models, 'allcnn')(opt)

# Initial model to be compared with the trained model
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

# Set logger with Wandb
logger = WandbLogger(opt['exp_tag'], hps=opt, mode=opt['wandb_mode'])

# Set cuda events for time analysis
start = th.cuda.Event(enable_timing=True)
end = th.cuda.Event(enable_timing=True)


# Reference: https://github.com/nitarshan/robust-generalization-measures
@th.no_grad()
def evaluate_complexity_measures(model,
                                 init_model,
                                 device,
                                 epoch,
                                 factor,
                                 seed,
                                 train_eval_loader,
                                 compute_all_measures):
    model.eval()
    init_model.eval()

    # Evaluates on the whole training set and with the current model
    train_eval_acc = average_accuracy(model,
                                      epoch,
                                      factor,
                                      device,
                                      train_eval_loader)

    complexities = {}
    if compute_all_measures:
        complexities, sigma, mag_sigma = \
            calculate_flatness_measures(model,
                                        init_model,
                                        train_eval_loader,
                                        train_eval_acc,
                                        seed)
        print('Measures successfully calculated!!\n')
        logger.log_pacbayes_details((epoch + 1) * factor, sigma, 'sigma')
        logger.log_pacbayes_details((epoch + 1) * factor, mag_sigma, 'magsima')

    return EvaluationMetrics(train_eval_acc,
                             train_eval_loader.n,
                             complexities)


@th.no_grad()
def average_accuracy(model,
                     epoch,
                     factor,
                     device,
                     data_loader):
    model.eval()
    bsz = data_loader.b

    # Double check for the initial index
    data_loader.sidx = 0

    avg_accuracy = AverageMeter()

    total_loops = int(math.floor(data_loader.n / bsz))

    for _ in range(total_loops):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        target = target.squeeze()
        yh = model(data)
        prec1 = accuracy(yh.data, target.data, topk=(1,))
        avg_accuracy.update(prec1[0].item() / 100, bsz)

    logger.log_batch_correctness((epoch + 1) * factor,
                                 'train_accuracy',
                                 avg_accuracy.count)

    return avg_accuracy.avg


def train(epoch, factor):
    model.train()

    fs, top1 = AverageMeter(), AverageMeter()

    # The floor function guarantees that we do not repeat data over a
    # given epoch. If the size of the dataset is not divisible by the
    # batch size, we leave some samples out of the loop
    maxb = int(math.floor(train_loader.n / train_loader.b))
    bsz = train_loader.b

    start.record()
    for bi in range(maxb):

        with th.set_grad_enabled(True):
            # Closure used by Entropy SGD
            def helper():
                def feval():
                    x, y = next(train_loader)
                    x, y = x.to(device), y.to(device)
                    y = y.squeeze()

                    optimizer.zero_grad()
                    yh = model(x)

                    # Computes the cross-entropy loss
                    # nn.NLLLoss(reduction='mean')(nn.LogSoftmax()(yh), y)
                    f = criterion.forward(yh, y)
                    f.backward()

                    # Computes the total correct predictions on the mini-batch
                    prec1 = accuracy(yh.data, y.data, topk=(1,))

                    return (f.data.item(), prec1[0].item())
                return feval

            f, acc = optimizer.step(helper(), model, criterion)

        # Average loss over the batches
        fs.update(f, bsz)

        # Average of the percentage of correct values over batches
        # (10%, 20%) -> 15%
        top1.update(acc, bsz)

        if bi % opt['batch_step'] == 0:
            print(f'[{epoch}][{bi} / {maxb}] | Avg. Loss: {round(fs.avg, 6)}; '
                  f' Perc. top1 accuracy: {round(top1.avg, 4)}')

    scheduler.step()
    end.record()
    # Explanation about the ordering:
    # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964/9
    th.cuda.synchronize()

    # Logs the training time by epoch
    logger.log_time((epoch + 1) * factor, start.elapsed_time(end))

    # Logs the learning rate
    logger.log_lr((epoch + 1) * factor,
                  optimizer.param_groups[0]['lr'])

    # Logs the size of the dataset used on the given epoch
    logger.log_batch_correctness((epoch + 1) * factor, 'train', top1.count)

    # Check if the complexity measure will be calculated
    # Calculates on every step and on the last epoch
    evaluate = epoch % opt['epoch_step'] == 0 or epoch == opt['B'] - 1

    # Evaluates complexity if applicable
    if evaluate and opt['calculate']:
        msg = f'Evaluating complexity measures at epoch {epoch}.'
        print(msg)
        measure_model = deepcopy(model)
        measure_init_model = deepcopy(init_model)
        train_eval = evaluate_complexity_measures(measure_model,
                                                  measure_init_model,
                                                  device,
                                                  epoch,
                                                  factor,
                                                  opt['s'],
                                                  train_eval_loader,
                                                  compute_all_measures=True)

        logger.log_complexity_measures((epoch + 1) * factor,
                                       train_eval.all_complexities)

    print(f'\nTrain [{epoch}] | Avg. Loss: {round(fs.avg, 6)}; '
          f'Perc. top1 accuracy: {round(top1.avg, 4)}')

    logger.log_all_epochs((epoch + 1) * factor,
                          DatasetSubsetType.TRAIN,
                          fs.avg,
                          top1.avg)

    logger.log_optim_params((epoch + 1) * factor,
                            optimizer.gamma,
                            optimizer.langevin_lr,
                            optimizer.alpha,
                            optimizer.momentum,
                            int(optimizer.nesterov))


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


@th.no_grad()
def dry_feed():
    """
    Dry feed for warming up the gpu
    making sure it is not on stand by mode
    """
    # Dry feed is performed without dropout (p=0).
    cache = set_dropout()
    maxb = int(math.floor(train_loader.n / train_loader.b))
    for _ in range(maxb):
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)
        y = y.squeeze()
        yh = model(x)

    # Once dry feed is finished, dropout probabilities are reset.
    set_dropout(cache)


@th.no_grad()
def val(e, factor, data_loader):

    dry_feed()
    model.eval()

    bsz = data_loader.b
    maxb = int(math.floor(data_loader.n / bsz))

    # Double check for the initial index
    data_loader.sidx = 0

    fs, top1 = AverageMeter(), AverageMeter()
    for _ in range(maxb):
        x, y = next(data_loader)
        x, y = x.to(device), y.to(device)

        y = y.squeeze()
        yh = model(x)
        f = criterion.forward(yh, y).data.item()
        prec1 = accuracy(yh.data, y.data, topk=(1,))

        fs.update(f, bsz)
        top1.update(prec1[0].item(), bsz)

    print(f'Test [{e}] | Avg. Loss: {round(fs.avg, 6)}; '
          f' Perc. top1 accuracy: {round(top1.avg, 4)}\n')

    logger.log_all_epochs((epoch + 1) * factor,
                          DatasetSubsetType.TEST,
                          fs.avg,
                          top1.avg)

    logger.log_batch_correctness((epoch + 1) * factor, 'val', top1.count)


for epoch in range(opt['B']):

    # Define factor to log results (L x epochs)
    factor = opt['L'] if opt['L'] > 0 else 1

    train(epoch, factor)
    val(epoch, factor, val_loader)
