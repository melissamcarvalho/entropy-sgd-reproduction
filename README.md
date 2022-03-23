# Entropy-sgd reproduction

- [Entropy-sgd reproduction](#entropy-sgd-reproduction)
  - [Objectives](#objectives)
  - [Environment setup](#environment-setup)
    - [Build the container docker](#build-the-container-docker)
    - [Run the container docker](#run-the-container-docker)
    - [Steps to be executed inside the container docker](#steps-to-be-executed-inside-the-container-docker)
      - [Data preprocessing](#data-preprocessing)
      - [Training the model](#training-the-model)
      - [Ploting the results](#ploting-the-results)
    - [Quick reminders](#quick-reminders)

## Objectives

The reproduction of the paper [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/abs/1611.01838) had the following goals:

1. Reproduce experiments for training the CIFAR-10 dataset. A CNN-based model [allcnn](entropy-sgd/models.py) was trained with Entropy-SGD for $L = 0$ (vanilla SGD) and $L > 0$.
2. Evaluate six complexity measures adapted from [this repository](https://github.com/nitarshan/robust-generalization-measures/blob/master/data/generation/measures.py): 
   * pacbayes-init
   * pacbayes-orig
   * pacbayes-flatness
   * pacbayes-mag-init
   * pacbayes-mag-orig
   * pacbayes-mag-flatness.

## Environment setup

### Build the container docker

```
docker build -t reproduction:0.1.0 .
```

### Run the container docker

```
docker run -it --rm -v $PWD/:/entropy-reproduction/ --gpus=all --name="cifar_reproduction" -p 8888:8888 --ipc="host" reproduction:0.1.0
```

### Steps to be executed inside the container docker

#### Data preprocessing

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

#### Training the model

```
cd entropy-sgd
python train.py \
        -B 1 \
        -exp-tag initial_test_one_epoch_allcnn \
        -wandb-mode disabled \
        -m allcnn \
        -b 128 \
        -eval-b 1000 \
        --lr 0.1 \
        --l2 0 \
        -L 0 \
        --gamma 0 \
        --scoping 0 \
        --noise 0 \
        -g 0 \
        -s 51 \
        -epoch-step 100 \
        -batch-step 100
```

```
usage: train.py [-h] [-m M] [-b B] [-eval-b EVAL_B] [-B B] [--lr LR] [--l2 L2]
                [-L L] [--gamma GAMMA] [--scoping SCOPING] [--noise NOISE]
                [-g G] [-s S] [-epoch-step EPOCH_STEP]
                [-batch-step BATCH_STEP] [-exp-tag EXP_TAG]
                [-wandb-mode WANDB_MODE]

PyTorch Entropy-SGD

optional arguments:
  -h, --help            show this help message and exit
  -m M                  mnistfc | mnistconv | allcnn
  -b B                  Train batch size
  -eval-b EVAL_B        Val, Test batch size
  -B B                  Max epochs
  --lr LR               Learning rate
  --l2 L2               L2
  -L L                  Langevin iterations
  --gamma GAMMA         gamma
  --scoping SCOPING     scoping
  --noise NOISE         SGLD noise
  -g G                  GPU idx.
  -s S                  seed
  -epoch-step EPOCH_STEP
                        epoch step to save results
  -batch-step BATCH_STEP
                        batch step to save results
  -exp-tag EXP_TAG      tag of the experiment
  -wandb-mode WANDB_MODE
                        mode of the wandb logger
```

For debugging purposes, a script with a test experiment may be executed:

```bash
cd entropy-sgd
./test_metrics_experiment.sh
```

#### Ploting the results

Data downloaded from Weights and Biases were saved at a specific folder (`WANDB_RESULTS`).

Plots were created with notebooks `notebooks/1_wandb_plots_cifar.ipynb`, and `notebooks/2_wandb_plots_cifar_final.ipynb`.

`jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`

### Quick reminders

`sudo systemctl daemon-reload`

`sudo systemctl restart docker`