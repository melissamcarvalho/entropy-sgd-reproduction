# Trainining process

## CIFAR-10 dataset

The CIFAR-10 dataset is divided in two subsets:

* Train: 50000 three channel images of size 32 x 32.
* Validation: 10000 three channel images of size 32 x 32.

The training and validation sets are built with the `sampler_t` class. During training, batches of a predefined size are randomly selected. On the other hand, during validation all the validation set is selected on the same order by the following process:

* [0, 1, ... , batch\_size - 1]
* [batch\_size, batch\_size + 1, ..., 2*batch\_size - 1]
* ...

## Convolutional neural network model

The model `allcnn` is a convolutional neural network. The network is built by blocks of the following layers: 2D-CNNs, batch normalization, and relu layers. Below we list how the dimensions of the network evolve:

**NOTE:** Add a visual diagram here.

```
    (N, 03, 32, 32) -> (N, 96, 32, 32) -> (N, 96, 32, 32) ->
->  (N, 96, 16, 16) -> (N, 96, 16, 16) -> (N, 192, 16, 16) ->
->  (N, 192, 16, 16) -> (N, 192, 08, 08) -> (N, 192, 08, 08) ->
->  (N, 192, 08, 08) -> (N, 192, 08, 08) -> (N, 10, 08, 08) -> 
->  (N, 10, 01, 01)
```

The loss function used to train the network is the [cross entropy loss function](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

### Experiments documentation

`final-experiments.sh`: Experiment to reproduce the paper. "We train for 200 epochs with SGD and Nesterov’s momentum during which the initial learning rate of 0.1 decreases by a factor of 5 after every 60 epochs."

"We train Entropy-SGD with L = 20 for 10 epochs with the original dropout of 0.5. The initial learning rate of the outer loop is set to 1 and drops by a factor of 5 every 4 epochs, while the learning rate of the SGLD updates is fixed to 0.1 with thermal noise 10−4. As the scoping scheme, we set the initial value of the scope to gamma=0.03 which increases by a factor of 1.001 after each parameter update."

The only information being ignored is the initial dropout of 0.5. We are using the fixed `allcnn` network with initial dropout of 0.2.

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
- Network is fixed as allcnn. Initial dropout is 0.2;
- Complexity measure is calculated in all epochs;

`exp_reproduce_nomeasure.sh`: Same experiments from `final-experiments.sh` but with no calculation of the complexity measures. The goal is to identify that indeed the complexity measure does not affect the training process. It is a sanity check.

`gamma_variation.sh`: Variation of gamma for the analysis of the flatness measures.

`langevin-experiments.sh`: Variation of langevin for the analysis of the flatness measures. Given the reproduciton parameters provided on `final-experiments.sh`, we add variations for L to keep the same number of epochs (200), and also to keep the same step on the learning rate update (80 L x epochs). Number 80 was defined on the experiment with L=20, E=10 and step=4. 

| L| E| step|
| - | - | - |
|2 | 100|40 |
|4|50| 20|
|5 |40 | 16 |
| 8| 25| 10|
|10 |20 | 8|
|40 | 5 | 2| 

`exp_reproduce_seed_variation`: Repeat the `final-experiments.sh` with different seeds for comparison. 