# Entropy-sgd reproduction

- [Entropy-sgd reproduction](#entropy-sgd-reproduction)
  - [Objectives](#objectives)
  - [Environment setup](#environment-setup)
    - [Build the container docker](#build-the-container-docker)
    - [Run the container docker](#run-the-container-docker)
  - [Pipeline](#pipeline)
    - [Data preprocessing](#data-preprocessing)
    - [Training the model](#training-the-model)
    - [Results](#results)
    - [Additional results offline](#additional-results-offline)
  - [Notes](#notes)
    - [Notes on cuda update](#notes-on-cuda-update)
    - [Notes on docker restart](#notes-on-docker-restart)

## Objectives

The reproduction of the paper [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/abs/1611.01838) had the following goals:

1. Evaluate the performance of Entropy-SGD on a benchmark dataset (e.g., classification with CIFAR-10 dataset). A CNN-based model [ALL-CNN-C](entropy-sgd/models.py) was trained with Entropy-SGD for $L = 0$ (vanilla SGD) and $L > 0$. The model was defined by [Springenberg et al., 2014](https://arxiv.org/pdf/1412.6806.pdf).
2. Evaluate Entropy-SGD result with flatness-based complexity measures adapted from [this repository](https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py) and defined by [Jiang et al, 2019](https://arxiv.org/pdf/1912.02178.pdf): 
   * pacbayes-flatness
   * pacbayes-mag-flatness.

    Table below list the Kendall’s Rank-Correlation Coefficient on CIFAR-10 dataset obtained by Jiang et al, 2019 (Table 5).

    | Measure | Index on the paper |Kendall's rank correlation coefficient |
    | - |  - | - |
    | Pacbayes flatness| 53| 0.303 |
    | Pacbayes mag flatness| 61| 0.365|

## Environment setup

### Build the container docker

```
docker build -t reproduction:0.1.0 .
```

### Run the container docker

```
docker run -it --rm -v $PWD/:/entropy-reproduction/ --gpus=all --name="cifar_reproduction" -p 8888:8888 --ipc="host" reproduction:0.1.0
```

## Pipeline

### Data preprocessing

First download the CIFAR-10 dataset from [this source](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and save inside the folder `entropy-sgd`. Afterwards, run the `process_cifar.py` script to execute the pre-processing steps.

```
cd entropy-sgd
python process_cifar.py -d cifar-10-python
```

```
usage: process_cifar.py [-h] -d DATA

Process CIFAR-10

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Directory containing cifar-10-batches-py
```

Results are going to be saved at `proc`.

### Training the model

```
cd entropy-sgd
python train.py \
        -B 10 \
        -exp-tag test_entropy \
        -wandb-mode online \
        -b 100 \
        -eval-b 100 \
        -lr 1 \
        -weight-decay 0 \
        -L 20 \
        -g 0 \
        -s 42 \
        -epoch-step 1 \
        -batch-step 100 \
        -lr-step 4 \
        -lr-decay 0.2 \
        -gamma 0.03 \
        -scoping 0.001 \
        -noise 0.0001 \
        -nesterov \
        -momentum 0.9 \
        -apply-scoping \
        -deterministic \
        -dropout 0.5
```

```
usage: train.py [-h] [-b B] [-eval-b EVAL_B] [-B B] [-lr LR]
                [-weight-decay WEIGHT_DECAY] [-damp DAMP] [-L L]
                [-gamma GAMMA] [-scoping SCOPING] [-noise NOISE] [-g G] [-s S]
                [-epoch-step EPOCH_STEP] [-batch-step BATCH_STEP]
                [-exp-tag EXP_TAG] [-wandb-mode WANDB_MODE] [-lr-step LR_STEP]
                [-lr-decay LR_DECAY] [-apply-scoping] [-nesterov]
                [-momentum MOMENTUM] [-calculate] [-deterministic]
                [-dropout DROPOUT]

PyTorch Entropy-SGD

optional arguments:
  -h, --help            show this help message and exit
  -b B                  mini-batch for training and validation
  -eval-b EVAL_B        mini-batch for complexity measures
  -B B                  number of epochs
  -lr LR                learning rate of outer loop
  -weight-decay WEIGHT_DECAY
                        weight decay
  -damp DAMP            momentum factor, 1 - damp, please check the optimizer.
  -L L                  langevin iterations
  -gamma GAMMA          gamma
  -scoping SCOPING      scoping
  -noise NOISE          SGLD noise
  -g G                  GPU idx.
  -s S                  seed
  -epoch-step EPOCH_STEP
                        epoch step to save results
  -batch-step BATCH_STEP
                        batch step to save results
  -exp-tag EXP_TAG      tag of the experiment
  -wandb-mode WANDB_MODE
                        mode of the wandb logger
  -lr-step LR_STEP      step to apply learning rate decay
  -lr-decay LR_DECAY    decay factor applied to the learning rate
  -apply-scoping        whether or not the gamma scoping is applied
  -nesterov             whether or not nesterov is applied
  -momentum MOMENTUM    whether or not apply momentum on the optimizer
  -calculate            whether or not calculate complexity measures
  -deterministic        whether or not use deterministic mode in torch
  -dropout DROPOUT      probability of the first dropout layer at allcnn model
```

For debugging purposes, a script with a test experiment may be executed:

```bash
cd entropy-sgd
experiments/test.sh
```

Experiments and their associated scripts are listed on the [training notes](entropy-sgd/train_notes.md).

### Results

Reports and experiments are documented with [Weights & Biases](https://docs.wandb.ai/). Please, check [this link](https://wandb.ai/mmcmelissa/Entropy%20SGD%20Reproduction?workspace=user-mmcmelissa).

### Additional results offline

Data downloaded from Weights and Biases was saved at a specific folder (`WANDB_RESULTS`).

Plots were created with notebooks `notebooks/1_wandb_plots_cifar.ipynb`, and `notebooks/2_wandb_plots_cifar_final.ipynb`.

`jupyter notebook --ip 0.0.0.0 --no-browser --port $PORT --allow-root`

## Notes

### Notes on cuda update

- [Reference](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local)
- [GPG error fix](https://github.com/NVIDIA/nvidia-docker/issues/1632)

### Notes on docker restart

- `sudo systemctl daemon-reload`
- `sudo systemctl restart docker`
