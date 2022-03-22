from typing import Dict, Optional

import wandb

from experiment_config import (
    ComplexityType,
    DatasetSubsetType
)


class BaseLogger:
    """
    Metrics logging
    """
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        raise NotImplementedError()

    def log_generalization_gap(self,
                               epoch: int,
                               train_acc: float,
                               val_acc: float,
                               train_loss: float,
                               val_loss: float,
                               all_complexities:
                                   Dict[ComplexityType, float]) -> None:
        """
        Logs differences between training and validation sets, and
        the result for the complexity measures over the training set.
        """
        self.log_metrics(
            epoch,
            {
                'generalization/error': train_acc - val_acc,
                'generalization/loss': val_loss - train_loss,
                **{'complexity/{}'.format(k.name):
                    v for k, v in all_complexities.items()}
            })

    def log_all_epochs(self,
                       epoch: int,
                       datasubset: DatasetSubsetType,
                       avg_loss: float,
                       pctg_acc: float) -> None:
        """
        Logs the average loss, the percentage of errors, and the percentage
        accuracy at the end of the epoch.
        """
        self.log_metrics(
            epoch,
            {
                'cross_entropy/{}'.format(datasubset.name.lower()): avg_loss,
                'percentage_errors/{}'.format(datasubset.name.lower()):
                    100. - pctg_acc,
                'percentage_accuracy/{}'.format(datasubset.name.lower()):
                    pctg_acc
            })

    def log_gamma(self,
                  epoch: int,
                  datasubset: DatasetSubsetType,
                  gamma: float):
        """
        Logs gamma by epoch.
        """
        self.log_metrics(
            epoch,
            {
                'gamma/{}'.format(datasubset.name.lower()): gamma,
            })

    def log_lr(self, epoch: int, datasubset: DatasetSubsetType, lr: float):
        """
        Logs learning rate by the end of the epoch.
        """
        self.log_metrics(
            epoch,
            {
                'learning_rate/{}'.format(datasubset.name.lower()): lr,
            })

    def log_time(self, epoch: int, time: float):
        """
        Logs time to run an epoch.
        """
        self.log_metrics(
            epoch,
            {
                'time': time,
            })

    def log_stop_criteria(self, epoch: int, criteria: int):
        """
        Logs if the stop criteria was reached at the end of the epoch.
        """
        self.log_metrics(
            epoch,
            {
                'stop_criteria': criteria,
            })

    def log_batch_correctness(self,
                              epoch: int,
                              tag: str,
                              total: int):
        """
        Logs the total data considered on the epoch for sanity check.
        """
        self.log_metrics(
            epoch,
            {
                'batch_correctness/{}/batch'.format(tag): total
            })


class WandbLogger(BaseLogger):
    def __init__(self,
                 tag: Optional[str] = None,
                 hps: Optional[dict] = None,
                 group: Optional[str] = None,
                 mode: str = 'online'):
        wandb.init(project='Entropy SGD Reproduction',
                   config=hps,
                   tags=[tag],
                   group=group,
                   mode=mode)

    def log_metrics(self, step: int, metrics: dict):
        wandb.log(metrics, step=step)
