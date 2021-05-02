from contextlib import contextmanager
from copy import deepcopy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from experiment_config import ComplexityType as CT
from models import ExperimentBaseModel


# Adapted from https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
  def in_place_reparam(model, prev_layer=None):
    for child in model.children():
      prev_layer = in_place_reparam(child, prev_layer)
      if child._get_name() == 'Conv2d':
        prev_layer = child
      elif child._get_name() == 'BatchNorm2d':
        scale = child.weight / ((child.running_var + child.eps).sqrt())
        prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
        perm = list(reversed(range(prev_layer.weight.dim())))
        prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
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
  model: ExperimentBaseModel,
  sigma: float,
  rng,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model


# Adapted from https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
  model: ExperimentBaseModel,
  dataloader: DataLoader,
  accuracy: float,
  seed: int,
  magnitude_eps: Optional[float] = None,
  search_depth: int = 15,
  montecarlo_samples: int = 10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
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
        for batch in range(dataloader.total_loops):
          data, target = next(dataloader)
          data = data.to(device)
          target = target.to(device)
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
          loss_estimate += batch_correct.sum()
        loss_estimate /= dataloader.n
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
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
  model: ExperimentBaseModel,
  init_model: ExperimentBaseModel,
  dataloader: DataLoader,
  acc: float,
  seed: int,
) -> Dict[CT, float]:
  measures = {}

  model = _reparam(model)
  init_model = _reparam(init_model)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  m = dataloader.n

  def get_weights_only(model: ExperimentBaseModel) -> List[Tensor]:
     blacklist = {'bias', 'bn'}
     return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]

  weights = get_weights_only(model)
  dist_init_weights = [p-q for p,q in zip(weights, get_weights_only(init_model))]
  d = len(weights)

  def get_vec_params(weights: List[Tensor]) -> Tensor:
     return torch.cat([p.view(-1) for p in weights], dim=0)

  w_vec = get_vec_params(weights)
  print('Get vec params')
  dist_w_vec = get_vec_params(dist_init_weights)
  num_params = len(w_vec)

  print("Calculating Flatness-based measures \n")
  sigma = _pacbayes_sigma(model, dataloader, acc, seed)
  def _pacbayes_bound(reference_vec: Tensor) -> Tensor:
    return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10
  measures[CT.PACBAYES_INIT] = _pacbayes_bound(dist_w_vec) # 48
  measures[CT.PACBAYES_ORIG] = _pacbayes_bound(w_vec) # 49
  measures[CT.PACBAYES_FLATNESS] = torch.tensor(1 / sigma ** 2) # 53

  print("Magnitude-aware Perturbation Bounds")
  mag_eps = 1e-3
  mag_sigma = _pacbayes_sigma(model, dataloader, acc, seed, mag_eps)
  omega = num_params
  def _pacbayes_mag_bound(reference_vec: Tensor) -> Tensor:
    numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2)**2) / omega
    denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
    return 1/4 * (numerator / denominator).log().sum() + math.log(m / mag_sigma) + 10
  measures[CT.PACBAYES_MAG_INIT] = _pacbayes_mag_bound(dist_w_vec) # 56
  measures[CT.PACBAYES_MAG_ORIG] = _pacbayes_mag_bound(w_vec) # 57
  measures[CT.PACBAYES_MAG_FLATNESS] = torch.tensor(1 / mag_sigma ** 2) # 61

  # Adjust for dataset size
  def adjust_measure(measure: CT, value: float) -> float:
    if measure.name.startswith('LOG_'):
      return 0.5 * (value - np.log(m))
    else:
      return np.sqrt(value / m)
  return {k: adjust_measure(k, v.item()) for k, v in measures.items()}
