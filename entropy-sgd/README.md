## Entropy-SGD: Biasing Gradient Descent Into Wide Valleys

This is the implementation for [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/abs/1611.01838) which will be presented at [ICLR '17](http://iclr.cc). It contains implementation in PyTorch in the [python](python) folder.

-----------------------------

### Instructions for PyTorch

The code for this is inside the [python](python) folder. You will need the Python packages `torch` and `torchvision` installed from [pytorch.org](pytorch.org).

1. The MNIST example downloads and processes the dataset the first time it is run. The files will be stored in the `proc` folder (same as CIFAR-10 in the Lua version)

2. Run ``python train.py -h`` to check out the command line arguments. The default is to run SGD with Nesterov's momentum on LeNet. You can run Entropy-SGD with
   ```
   python train.py -m mnistconv -L 20 --gamma 1e-4 --scoping 1e-3 --noise 1e-4
   ```
Everything else is identical to the Lua version.

-----------------------------

### Computing the Hessian

The code in [hessian.py](python/hessian.py) computes the Hessian for a small convolutional neural network using SGD and Autograd. Please note that this takes a lot of time, a day or so, and you need to be careful of the memory usage. The experiments in the paper were run on EC2 with 256 GB RAM. Note that this code uses the MNIST dataset downloaded when you run the PyTorch step above.
