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

    def log_complexity_measures(self,
                                epoch: int,
                                all_complexities:
                                Dict[ComplexityType, float]) -> None:
        """
        Logs the complexity measures over the training set.
        """
        self.log_metrics(
            epoch,
            {
                **{'complexity/{}'.format(k.name):
                    v for k, v in all_complexities.items()}
            })

    def log_pacbayes_details(self,
                             epoch: int,
                             details: dict,
                             tag: str) -> None:
        """
        Logs parameters of the _pacbayes_sigma measure: depth and displacement.
        depth: number of iterations required to reach the target displacement.
        displacement: first value lower than the target displacement.
        """

        self.log_metrics(
            epoch,
            {
                f'complexity/{tag}_depth': details['final_depth'],
                f'complexity/{tag}_displacement': details['found_displacement']
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
                'general/cross_entropy_{}'.
                format(datasubset.name.lower()): avg_loss,
                'general/percentage_errors_{}'.
                format(datasubset.name.lower()): 100. - pctg_acc,
                'general/percentage_accuracy_{}'.
                format(datasubset.name.lower()): pctg_acc
            })

    def log_optim_params(self,
                         epoch: int,
                         gamma: float,
                         langevin_lr: float,
                         mean_weight: float,
                         momentum: float,
                         nesterov: bool):
        """
        Logs Entropy SGD inner parameters
        """
        self.log_metrics(
            epoch,
            {
                'optim/gamma': gamma,
                'optim/langevin_learning_rate': langevin_lr,
                'optim/alpha_mean_weight': mean_weight,
                'optim/momentum': momentum,
                'optim/nesterov': nesterov
            })

    def log_lr(self, epoch: int, lr: float):
        """
        Logs learning rate by the end of the epoch.
        """
        self.log_metrics(
            epoch,
            {
                'optim/learning_rate': lr,
            })

    def log_time(self, epoch: int, time: float):
        """
        Logs time to run an epoch.
        """
        self.log_metrics(
            epoch,
            {
                'time/training_time': time,
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
                'batch_correctness/{}'.format(tag): total
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
