# Fashion-MNIST: MLP vs CNN vs Vision Transformer

Image classification on Fashion-MNIST with three neural network families implemented in PyTorch (an MLP, a CNN and a Vision Transformer written from scratch), plus PCA implemented from scratch in NumPy for dimensionality reduction.

Team project for the Introduction to Machine Learning course at EPFL.

## Headline result

The CNN reached **89.3% accuracy (macro F1 0.893)** on the held-out validation set. PCA cut the MLP's parameters by 74% (235,146 → 60,042) and its training time by 23%, for a drop of under 1 accuracy point.

| Model | Validation accuracy | Macro F1 |
|-------|--------------------:|---------:|
| CNN (3 conv layers, channels 16/32/64) | **89.3%** | **0.893** |
| MLP | 84.2% | 0.843 |
| MLP + PCA (100 components) | 83.5% | 0.834 |
| Vision Transformer | 83.5% | 0.80 |
| LeNet-style CNN (first attempt) | 58% | – |

All numbers come from the [project report](report.pdf). The validation set is a random third of the training data.

## What we did

- Implemented an MLP, a CNN and a **Vision Transformer from scratch** (patch embedding, multi-head self-attention, positional embeddings, class token) in PyTorch, with a shared training loop.
- Implemented **PCA from scratch** (eigendecomposition of the covariance matrix) and measured the accuracy/efficiency trade-off when training the MLP on 100 components.
- Compared two CNN architectures, then tuned optimizer (SGD vs Adam), learning rate and channel widths. Adam with a learning rate of 1e-3 worked best.
- Tuned the Transformer's learning rate, batch size and number of epochs, and documented the over- and underfitting behaviour in the report.

## Tech stack

Python, PyTorch, NumPy, torchinfo.

## How to run

```bash
pip install -r requirements.txt
python main.py --data dataset --nn_type cnn --lr 1e-3 --max_iters 10
```

`--nn_type` can be `mlp`, `cnn` or `transformer`. Add `--use_pca --pca_d 100` to train the MLP on PCA-reduced features. `--max_iters` is the number of epochs.

The code expects the data as NumPy arrays in `dataset/`: `train_data.npy`, `train_label.npy` and `test_data.npy`. The dataset is not included in this repo. Fashion-MNIST is available from [Zalando Research](https://github.com/zalandoresearch/fashion-mnist).

By default, a third of the training set is held out for validation. `--test` trains on the full training set and predicts on the (unlabelled) test set instead.

## Repository structure

```
├── main.py                    # Data preparation, training and evaluation
├── src/
│   ├── data.py                # Dataset loading
│   ├── utils.py               # Normalization and metrics (accuracy, macro F1)
│   └── methods/
│       ├── deep_network.py    # MLP, CNN, Vision Transformer and the Trainer
│       └── pca.py             # PCA from scratch
└── report.pdf                 # Method, experiments and results
```
