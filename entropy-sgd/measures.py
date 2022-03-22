# https://github.com/nitarshan/robust-generalization-measures

from contextlib import contextmanager
from copy import deepcopy
import math

import torch
import numpy as np
from torch.autograd import Variable

from experiment_config import ComplexityType as CT
from utils import AverageMeter, accuracy


# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
# This function reparametrizes the networks with batch normalization in a way
# that it calculates the same function as the original network but without
# batch normalization. Instead of removing batch norm completely, we set the
# bias and mean to zero, and scaling and variance to one.
# Warning: This function only works for convolutional and fully connected
# networks. It also assumes that module.children() returns the children of
# a module in the forward pass order. Recurssive construction is allowed.
@torch.no_grad()
def _reparam(model):
    def in_place_reparam(model, prev_layer=None):
        for child in model.children():
            prev_layer = in_place_reparam(child, prev_layer)
            if child._get_name() == 'Conv2d':
                prev_layer = child
            elif child._get_name() == 'BatchNorm2d':
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.\
                    copy_(child.bias
                          + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                # permute to match the dimensions and apply the scale
                prev_layer.weight.\
                    copy_((prev_layer.weight.permute(perm)
                           * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
        return prev_layer
    model = deepcopy(model)
    in_place_reparam(model)
    return model


@contextmanager
def _perturbed_model(
    model,
    sigma,
    rng,
    magnitude_eps=None
):
    device = next(model.parameters()).device
    if magnitude_eps is not None:
        noise = [torch.normal(0, sigma**2 * torch.abs(p)**2
                 + magnitude_eps**2, generator=rng)
                 for p in model.parameters()]
    else:
        noise = [torch.normal(0, sigma**2, p.shape, generator=rng).to(device)
                 for p in model.parameters()]
    model = deepcopy(model)
    try:
        [p.add_(n) for p, n in zip(model.parameters(), noise)]
        yield model
    finally:
        [p.sub_(n) for p, n in zip(model.parameters(), noise)]
        del model


# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
    model,
    dataloader,
    acc,
    seed,
    magnitude_eps=None,
    search_depth=15,
    montecarlo_samples=10,
    accuracy_displacement=0.1,
    displacement_tolerance=1e-2,
):
    lower, upper = 0, 2
    sigma = 1

    BIG_NUMBER = 10348628753
    device = next(model.parameters()).device
    rng = torch.Generator(device=device) if magnitude_eps is not None \
        else torch.Generator()
    rng.manual_seed(BIG_NUMBER + seed)

    print(f'Total depths: {search_depth}')
    print(f'Total MC samples: {montecarlo_samples}')

    for depth in range(search_depth):
        print(f'Current depth: {depth}')
        sigma = (lower + upper) / 2
        accuracy_samples = []
        for sample in range(montecarlo_samples):
            print(f'Monte Carlo sample: {sample}')
            with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
                loss_estimate = 0
                dataloader.sidx = 0
                dataloader.train = False
                top1 = AverageMeter()
                bsz = dataloader.b

                total_loops = int(dataloader.n / bsz)
                for _ in range(total_loops):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)

                    with torch.no_grad():
                        x = Variable(x)
                        y = Variable(y.squeeze())
                        yh = p_model(x)
                        prec1 = accuracy(yh.data, y.data, topk=(1,))
                    top1.update(prec1[0].item() / 100, bsz)
                loss_estimate = top1.avg
                print(f'the total number of samples is {top1.count}')
                accuracy_samples.append(loss_estimate)

        displacement = abs(np.mean(accuracy_samples) - acc)
        print(f'Current dist: {abs(displacement - accuracy_displacement)}')
        if abs(displacement - accuracy_displacement) < displacement_tolerance:
            break
        elif displacement > accuracy_displacement:
            # Too much perturbation
            upper = sigma
        else:
            # Not perturbed enough to reach target displacement
            lower = sigma
    return sigma


@torch.no_grad()
def get_flat_measure(
    model,
    init_model,
    dataloader,
    acc,
    seed
):

    measures = {}

    model = _reparam(model)
    init_model = _reparam(init_model)

    # Total number of samples
    m = dataloader.n

    def get_weights_only(model):
        blacklist = {'bias', 'bn'}
        # In allcnn all layers have name .weight or .bias
        return [p for name, p in model.named_parameters()
                if all(x not in name for x in blacklist)]

    weights = get_weights_only(model)
    dist_init_weights = [p - q for p, q in
                         zip(weights, get_weights_only(init_model))]

    def get_vec_params(weights):
        return torch.cat([p.view(-1) for p in weights], dim=0)

    print('Get vec params')
    w_vec = get_vec_params(weights)
    dist_w_vec = get_vec_params(dist_init_weights)
    num_params = len(w_vec)

    print("Calculating Flatness-based measures \n")
    sigma = _pacbayes_sigma(model, dataloader, acc, seed)

    def _pacbayes_bound(reference_vec):
        first = (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2)
        return first + math.log(m / sigma) + 10
    measures[CT.PACBAYES_INIT] = _pacbayes_bound(dist_w_vec)  # 48
    measures[CT.PACBAYES_ORIG] = _pacbayes_bound(w_vec)  # 49
    measures[CT.PACBAYES_FLATNESS] = torch.tensor(1 / sigma ** 2)  # 53

    print("Magnitude-aware Perturbation Bounds")
    mag_eps = 1e-3
    mag_sigma = _pacbayes_sigma(model, dataloader, acc, seed, mag_eps)
    omega = num_params

    def _pacbayes_mag_bound(reference_vec):
        numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) \
            * (reference_vec.norm(p=2)**2) / omega
        denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
        first = (1 / 4) * (numerator / denominator).log().sum()
        return first + math.log(m / mag_sigma) + 10
    measures[CT.PACBAYES_MAG_INIT] = _pacbayes_mag_bound(dist_w_vec)  # 56
    measures[CT.PACBAYES_MAG_ORIG] = _pacbayes_mag_bound(w_vec)  # 57
    measures[CT.PACBAYES_MAG_FLATNESS] = torch.tensor(1 / mag_sigma ** 2)  # 61
