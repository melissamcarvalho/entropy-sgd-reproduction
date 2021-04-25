# Entropy-sgd-reproduction

0. Gerar imagem Docker

```
docker build -t reproduction:0.1.0 .
```

1. Executar container

```
docker run -it --rm -v /home/jbflorindo/entropy-sgd-reproduction:/entropy-reproduction --gpus=all --name="cifar_reproduction" -p 8888:8888 --ipc="host" reproduction:0.1.0
```

2. python process_cifar.py -d ../cifar-10-python/

3. Exemplo de execução básica

```
bash
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

4. Requisitos faltantes no container:

pip install --upgrade torch torchvision

5. Reinicialização do container

`sudo systemctl daemon-reload`
`sudo systemctl restart docker`