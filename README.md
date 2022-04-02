# MixMatch

This is a PyTorch implementation of MixMatch, which allows training with customized dataset.

The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch) and the forked Pytorch implementation is [here](https://github.com/YU1ut/MixMatch-pytorch)

## Revision Note

Two revised training functions are updated compared to [repository](https://github.com/YU1ut/MixMatch-pytorch)

1. train_anomaly_SSL.py

   Revised the dataset part to allow customized dataset for training.
   Revised the original MixMatch loss function by considering the potential class imbalance issue in the labeled data.
  
3. train_anomaly_TL.py
   
   This is a simple baseline training process by supervised learning only using labeled data with the same number as that of SSL training.
   This allows performance evaluation.

## Usage

### Environment

Check code environment "requirements.txt"

### Train

1. Customized dataset preparation
   Put the data under "dataset/".
   Put the training/validatioin/test txt under the current location.
   Update the path information in the train_anomaly_SSL.py 
   
   


Train the model by 250 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 250 --out cifar10@250
```

Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 4000 --out cifar10@4000
```

### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10@250
```

## Results (Accuracy)
| #Labels | 250 | 500 | 1000 | 2000| 4000 |
|:---|:---:|:---:|:---:|:---:|:---:|
|Paper | 88.92 ± 0.87 | 90.35 ± 0.94 | 92.25 ± 0.32| 92.97 ± 0.15 |93.76 ± 0.06|
|This code | 88.71 | 88.96 | 90.52 | 92.23 | 93.52 |

(Results of this code were evaluated on 1 run. Results of 5 runs with different seeds will be updated later. )

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
