# Experiments overview

- [Experiments overview](#experiments-overview)
  - [CIFAR-10 dataset](#cifar-10-dataset)
  - [Convolutional neural network model](#convolutional-neural-network-model)
  - [Experiments](#experiments)

## CIFAR-10 dataset

The CIFAR-10 dataset is divided in two subsets:

* Train: 50000 three channel images of size 32 x 32.
* Validation: 10000 three channel images of size 32 x 32.

The training and validation sets are built with the `sampler_t` class. During training, batches of a predefined size are randomly selected. On the other hand, during validation all the validation set is selected with the following indexes:

* [0, 1, ... , batch\_size - 1]
* [batch\_size, batch\_size + 1, ..., 2*batch\_size - 1]
* ...

## Convolutional neural network model

The model `allcnn` (1,667,166 trainable parameters) is a convolutional neural network. The network is built by blocks of the following layers: 2D-CNNs, batch normalization, and relu layers. Below we list how the dimensions of the network evolve:

**TODO:** Add a visual diagram here.

|Layer (type) |              Output Shape    |    Param #|
| --- | --- | --- |
|   Dropout-1     |       [-1, 3, 32, 32]        |       0|
| Conv2d-2    |       [-1, 96, 32, 32]     |      2,688|
|  BatchNorm2d-3      |     [-1, 96, 32, 32]    |         192|
|     ReLU-4      |     [-1, 96, 32, 32]     |          0|
|  Conv2d-5       |    [-1, 96, 32, 32]      |    83,040|
|  BatchNorm2d-6   |        [-1, 96, 32, 32]    |         192|
|     ReLU-7        |   [-1, 96, 32, 32]       |        0|
|   Conv2d-8   |        [-1, 96, 16, 16]       |   83,040|
| BatchNorm2d-9    |       [-1, 96, 16, 16]         |    192|
|    ReLU-10     |      [-1, 96, 16, 16]       |        0|
|   Dropout-11   |        [-1, 96, 16, 16]   |            0|
|   Conv2d-12       |   [-1, 192, 16, 16] |        166,080|
|BatchNorm2d-13   |       [-1, 192, 16, 16]     |        384|
|    ReLU-14   |       [-1, 192, 16, 16]  |             0|
|   Conv2d-15     |     [-1, 192, 16, 16] |        331,968|
|  BatchNorm2d-16       |   [-1, 192, 16, 16]    |         384|
|    ReLU-17       |   [-1, 192, 16, 16]  |             0|
|  Conv2d-18      |      [-1, 192, 8, 8]    |     331,968|
|    BatchNorm2d-19    |        [-1, 192, 8, 8]     |        384|
|     ReLU-20   |         [-1, 192, 8, 8]          |     0|
|   Dropout-21     |       [-1, 192, 8, 8]        |       0|
|    Conv2d-22     |       [-1, 192, 8, 8]         |331,968|
|    BatchNorm2d-23   |         [-1, 192, 8, 8]    |         384|
|          ReLU-24       |     [-1, 192, 8, 8]    |           0|
|      Conv2d-25     |       [-1, 192, 8, 8]      |   331,968|
| BatchNorm2d-26        |    [-1, 192, 8, 8]  |           384|
|        ReLU-27     |       [-1, 192, 8, 8]         |      0|
|   Conv2d-28           |  [-1, 10, 8, 8]     |      1,930|
|   BatchNorm2d-29       |      [-1, 10, 8, 8]      |        20|
|     ReLU-30     |        [-1, 10, 8, 8]     |          0|
| AvgPool2d-31     |        [-1, 10, 1, 1]          |     0|
|    View-32      |             [-1, 10]            |   0|

The loss function used to train the network is the [cross entropy loss function](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## Experiments

`test.sh`: Experiment to test L=0 and L>0 behaviour, and the capability of two distinct runs with the same patameters generate similar results (reproducibility analysis).

`langevin.sh`: Variation of langevin and gamma for the analysis of the test errors. Afterwards, the goal is to make the analysis of the complexity measure on critical points.

- L x epochs is fixed in 400, and L has the following values: [0, 10, 20, 40, 50, 100, 200].
- Gamma varies in the range (3e-5, 3) in multiples of 10.
- Learning rate is fixed.
- Gamma is fixed.

| L| E | Gamma |
| - | - | - | 
| 0 | 400| 3e-5|
|10 | 40 | 3e-5 |
|20 | 20 | 3e-5|
|40 | 10 | 3e-5 |
|50 | 8 | 3e-5|
|100 | 4| 3e-5|
|200 | 2| 3e-5|
| 0 | 400| 3e-4|
|10 | 40 | 3e-4 |
|20 | 20 | 3e-4|
|40 | 10 | 3e-4 |
|50 | 8 | 3e-4|
|100 |4| 3e-4|
|200 |2| 3e-4|
| 0 | 400| 3e-3|
|10 | 40 | 3e-3 |
|20 | 20 | 3e-3|
|40 | 10 | 3e-3 |
|50 | 8 | 3e-3|
|100 |4| 3e-3|
|200 |2| 3e-3|
| 0 | 400| 3e-2|
|10 | 40 | 3e-2 |
|20 | 20 | 3e-2|
|40 | 10 | 3e-2|
|50 | 8 | 3e-2|
|100 |4| 3e-2|
|200 |2| 3e-2|
| 0 | 400| 3e-1|
|10 | 40 | 3e-1 |
|20 | 20 | 3e-1|
|40 | 10 | 3e-1 |
|50 | 8 | 3e-1|
|100 |4| 3e-1|
|200 |2| 3e-1|
| 0 | 400| 3|
|10 | 40 | 3 |
|20 | 20 | 3|
|40 | 10 | 3 |
|50 | 8 | 3|
|100 |4| 3|
|200 |2| 3|

`gamma.sh`: Application of gamma scoping on most promising results. Learning rate annealing will be also considered.

`reproduction.sh`: Experiments to reproduce the paper. "We train for 200 epochs with SGD and Nesterov’s momentum during which the initial learning rate of 0.1 decreases by a factor of 5 after every 60 epochs."

"We train Entropy-SGD with L = 20 for 10 epochs with the original dropout of 0.5. The initial learning rate of the outer loop is set to 1 and drops by a factor of 5 every 4 epochs, while the learning rate of the SGLD updates is fixed to 0.1 with thermal noise 10−4. As the scoping scheme, we set the initial value of the scope to gamma=0.03 which increases by a factor of 1.001 after each parameter update."

**SGD**: 
- 200 epochs;
- Optimizer step is performed with nesterov and momentum=0.9;
- Learning rate starts with 0.1 and drops 0.2 every 60 epochs;
- Network is fixed as allcnn. Initial dropout is 0.2;
- Complexity measure is calculated every 20 epochs or if the minimum learning rate is reached;

**Entropy SGD**:
- 10 epochs with 20 inner loops
- Optimizer step is performed with nesterov and momentum=0.9;
- Learning rate starts with 1 and drops 0.2 every 4 epochs;
- Langevin learning rate is fixed in 0.1;
- Noise is set to 0.0001;
- Gamma starts as 0.03 and increases by a factor of 1.001 at every parameter step;
- Alpha parameter that weights the expected \mu value is set to 0.75 (beta1=0.25);
- Initial dropout is 0.5;
- Complexity measure is calculated in all epochs;
