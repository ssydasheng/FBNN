# Functional Variational Bayesian Neural Networks
This code is jointly contributed by [Shengyang Sun](https://github.com/ssydasheng), [Guodong Zhang](https://github.com/gd-zhang) and [Jiaxin Shi](https://github.com/thjashin/).
## Introduction
Code for "Functional variational Bayesian neural networks" (https://arxiv.org/abs/1903.05779)

## Dependencies
This project runs with Python 3.6. Before running the code, you have to install
* [Tensorflow](https:www.tensorflow.org)
* [GPflow-Slim](https://github.com/ssydasheng/GPflow-Slim)

## Experiments
Below we shows some examples to run the experiments.
### x3 regression
```
python exp/toy.py -d x3 -in 0.01
```
### sinusoidal extrapolation
```
python exp/toy.py -d sin -na 40 -nh 5 -nu 500 -e 50000 -il -2
```
### Inference on Implicit Piecewise Priors
```
python exp/piecewise.py -d p_const
```
### Contextual Bandits
```
python exp/bandits.py --data_type statlog
```
## Citation
To cite this work, please use
```
@article{sun2019functional,
  title={Functional Variational Bayesian Neural Networks},
  author={Sun, Shengyang and Zhang, Guodong and Shi, Jiaxin and Grosse, Roger},
  journal={arXiv preprint arXiv:1903.05779},
  year={2019}
}
```
