import time
from typing import Dict, Optional

from torch import Tensor
import wandb

from experiment_config import (
  ComplexityType,
  Config,
  DatasetSubsetType,
  HParams,
  State,
  EvaluationMetrics,
  Verbosity,
)


class BaseLogger(object):
  def log_metrics(self, step: int, metrics: Dict[str, float]):
    raise NotImplementedError()

  def log_generalization_gap(self, epoch: int, train_acc: float, val_acc: float, train_loss: float, val_loss: float, all_complexities: Dict[ComplexityType, float]) -> None:
    self.log_metrics(
        epoch,
      {
        'generalization/error': train_acc - val_acc,
        'generalization/loss': train_loss - val_loss,
        **{'complexity/{}'.format(k.name): v for k,v in all_complexities.items()}
      })

  def log_all_epochs(self, epoch: int, datasubset: DatasetSubsetType, avg_loss: float, acc: float) -> None:
    self.log_metrics(
      epoch,
      {
        'cross_entropy/{}'.format(datasubset.name.lower()): avg_loss,
        'accuracy/{}'.format(datasubset.name.lower()): acc,
      })

  def log_gamma(self, epoch: int, datasubset: DatasetSubsetType, gamma: float):
    self.log_metrics(
      epoch,
      {
        'gamma/{}'.format(datasubset.name.lower()): gamma,
      })

  def log_lr(self, epoch: int, datasubset: DatasetSubsetType, lr: float):
    self.log_metrics(
      epoch,
      {
        'learning_rate/{}'.format(datasubset.name.lower()): lr,
      })


class WandbLogger(BaseLogger):
  def __init__(self, tag: Optional[str] = None, hps: Optional[dict] = None, group: Optional[str] = None, mode: str='online'):
    wandb.init(project='Entropy SGD Reproduction', config=hps, tags=[tag], group=group, mode=mode)

  def log_metrics(self, step: int, metrics: dict):
    wandb.log(metrics, step=step)