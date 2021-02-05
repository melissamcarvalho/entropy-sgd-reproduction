# In Search of Robust Measures of Generalization
[![arXiv](https://img.shields.io/badge/arXiv-2010.11924-b31b1b)](https://arxiv.org/abs/2010.11924)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Python](https://img.shields.io/badge/language-Python%203.7%2B-green?logo=python&logoColor=green)](https://www.python.org)


**Gintare Karolina Dziugaite**, **Alexandre Drouin**, Brady Neal, Nitarshan Rajkumar, Ethan Caballero, Linbo Wang, Ioannis Mitliagkas, Daniel M. Roy

>One of the principal scientific challenges in deep learning is explaining generalization, i.e., why the particular way the community now trains networks to achieve small training error also leads to small error on held-out data from the same population. It is widely appreciated that some worst-case theories -- such as those based on the VC dimension of the class of predictors induced by modern neural network architectures -- are unable to explain empirical performance. A large volume of work aims to close this gap, primarily by developing bounds on generalization error, optimization error, and excess risk. When evaluated empirically, however, most of these bounds are numerically vacuous. Focusing on generalization bounds, this work addresses the question of how to evaluate such bounds empirically. Jiang et al. (2020) recently described a large-scale empirical study aimed at uncovering potential causal relationships between bounds/measures and generalization. Building on their study, we highlight where their proposed methods can obscure failures and successes of generalization measures in explaining generalization. We argue that generalization measures should instead be evaluated within the framework of distributional robustness.


![Cover figure](https://github.com/nitarshan/robust-generalization-measures/raw/master/paper_graphic.png)


## Directory Structure

```
├── experiments
    ├── coupled_networks
        └── ...
    └── single_network
        └── ...
├── data
    └── generation
      ├── ...
      └── train.py
    └── nin.cifar10_svhn.csv
```

You can also look at the exact state of the code as submitted during the peer-review process [here](https://github.com/nitarshan/banana-smoothie-recipe-1776).

## Data

The data used in this study are available in a csv file with all experimental records (model configurations, generalization measures, generalization error) (`data/nin.cifar10_svhn.csv`)

## Contact us

karolina.dziugaite@elementai.com, alexandre.drouin@elementai.com
