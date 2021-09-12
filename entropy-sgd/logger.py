import wandb

from typing import Dict, Optional
from torch import Tensor

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
  """
  Metrics logging
  """
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

  def log_all_epochs(self,
                     epoch: int,
                     datasubset: DatasetSubsetType,
                     avg_loss: float,
                     acc_sum: float,
                     acc_count: float) -> None:
    self.log_metrics(
      epoch,
      {
        'cross_entropy/{}'.format(datasubset.name.lower()): avg_loss,
        'percentage_errors/{}'.format(datasubset.name.lower()): 100*(acc_count - acc_sum)/acc_count,
        'percentage_accuracy/{}'.format(datasubset.name.lower()): 100*(acc_sum/acc_count)
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

  def log_time(self, epoch: int, time: float):
    self.log_metrics(
      epoch,
      {
        'time': time,
      })

  def log_stop_criteria(self, epoch: int, criteria: int):
    self.log_metrics(
      epoch,
      {
        'stop_criteria': criteria,
      })

  def log_batch_correctness(self, epoch: int, tag: str, correct: int, total: int):
    self.log_metrics(
      epoch,
      {
        'batch_correctness/{}/correct'.format(tag): correct,
        'batch_correctness/{}/batch'.format(tag): total
      })


class WandbLogger(BaseLogger):
  def __init__(self,
               tag: Optional[str] = None,
               hps: Optional[dict] = None,
               group: Optional[str] = None,
               mode: str='online'):
    wandb.init(project='Entropy SGD Reproduction',
               config=hps,
               tags=[tag],
               group=group,
               mode=mode)

  def log_metrics(self, step: int, metrics: dict):
    wandb.log(metrics, step=step)