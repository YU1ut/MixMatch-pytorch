# MixMatch
This is an unofficial PyTorch implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). 
The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch).

Experiments on CIFAR-10 and STL-10 are available.

This repository carefully implemented important details of the official implementation to reproduce the results.


## Requirements
- Python 3.6+
- PyTorch 1.0
- **torchvision 0.2.2 (older versions are not compatible with this code)** 
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train
Train the model by 250 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 250 --out cifar10@250
```

Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 4000 --out cifar10@4000
```

Train STL-10:

```
python train.py --resolution <32|48|96> --out stl10 --data_root data/stl10 --dataset STL10 --n-labeled 5000
```


### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10@250
```

## Results (Accuracy)

### CIFAR10

| #Labels | 250 | 500 | 1000 | 2000| 4000 |
|:---|:---:|:---:|:---:|:---:|:---:|
|Paper | 88.92 ± 0.87 | 90.35 ± 0.94 | 92.25 ± 0.32| 92.97 ± 0.15 |93.76 ± 0.06|
|This code | 88.71 | 88.96 | 90.52 | 92.23 | 93.52 |

(Results of this code were evaluated on 1 run. Results of 5 runs with different seeds will be updated later. )

### STL10

Using the entire 5000 point dataset:

| Resolution | 32 | 48 | 96 |
|:---|:---:|:---:|:---:|
|Paper | - | - | 94.41 |
|This code | 82.69 | 86.41 | 91.33 |

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```