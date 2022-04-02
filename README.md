# MixMatch-pytorch-customized-dataset

This is a PyTorch implementation of MixMatch, which allows training with customized dataset.

The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch) and the forked Pytorch implementation is [here](https://github.com/YU1ut/MixMatch-pytorch).

## Revision Note

Two revised training functions are updated compared to the original forked [repository](https://github.com/YU1ut/MixMatch-pytorch).

In addition, I adjusted the code structure of the original Pytorch implementation and made several notes for better understanding.

1. train_SSL.py

   Revised the dataset part to allow customized dataset for training.
   
   Revised the original MixMatch loss function by considering the potential class imbalance issue in the labeled data.
  
2. train_TL.py
   
   This is a simple baseline training process by supervised learning only using labeled data with the same number as that of SSL training.
   
   This allows performance evaluation with SSL training.

## Usage

### Environment

Check code environment "requirements.txt".

### Train

1. Customized dataset preparation.

   Put the data under "dataset/".
   
   Put the training/validatioin/test txt under the current location.
   
   Update the path information both in the train_SSL.py and train_TL.py.
   
2. Parameter settting by users. For example, update the number of labeled data for training.

3. Train the model in SSL mode:

   python train_SSL.py

4. Train the model in TL mode:

   python train_TL.py


## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```
