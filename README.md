# Entropy-sgd reproduction

- [Entropy-sgd reproduction](#entropy-sgd-reproduction)
  - [Objectives](#objectives)
  - [Environment setup](#environment-setup)
    - [Build the container docker](#build-the-container-docker)
    - [Run the container docker](#run-the-container-docker)
  - [Pipeline](#pipeline)
    - [Data preprocessing](#data-preprocessing)
    - [Training the model](#training-the-model)
    - [Plotting the results](#plotting-the-results)
  - [Notes](#notes)
    - [Notes on cuda update](#notes-on-cuda-update)
    - [Notes on docker restart](#notes-on-docker-restart)

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
        -l2 0 \
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
        -deterministic
```

```
usage: train.py [-h] [-b B] [-eval-b EVAL_B] [-B B] [-lr LR] [-l2 L2] [-L L]
                [-gamma GAMMA] [-scoping SCOPING] [-noise NOISE] [-g G] [-s S]
                [-epoch-step EPOCH_STEP] [-batch-step BATCH_STEP]
                [-exp-tag EXP_TAG] [-wandb-mode WANDB_MODE] [-lr-step LR_STEP]
                [-lr-decay LR_DECAY] [-apply-scoping] [-nesterov]
                [-momentum MOMENTUM] [-calculate] [-deterministic]

PyTorch Entropy-SGD

optional arguments:
  -h, --help            show this help message and exit
  -b B                  mini-batch for training and validation
  -eval-b EVAL_B        mini-batch for complexity measures
  -B B                  number of epochs
  -lr LR                learning rate of outer loop
  -l2 L2                L2
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
```

For debugging purposes, a script with a test experiment may be executed:

```bash
cd entropy-sgd
./test.sh
```

Experiments and their associated scripts are listed on the [training notes](entropy-sgd/train_notes.md).

### Plotting the results

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
