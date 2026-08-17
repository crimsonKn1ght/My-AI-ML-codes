# My-AI-ML-codes

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=flat)](#running-them)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=flat)](#running-them)
[![Format](https://img.shields.io/badge/format-scripts%20%2B%20notebooks-f37626.svg?style=flat)](#architectures)

Reference implementations of architectures and classical methods, written from the paper
rather than imported from a library. Each one is standalone: no shared package, no
cross-imports, so any single file can be read or run on its own.

Most architectures come as a `.py` module holding the model definition, and several also
have a notebook that builds and exercises it.

## Architectures

| Directory | Implementation | Files |
|---|---|---|
| [`ResNet/`](ResNet/) | ResNet-34 from residual blocks | `resnet.py`, `resnet_implementation.ipynb` |
| [`DenseNet/`](DenseNet/) | DenseNet-121 from dense blocks and transition layers | `densenet.ipynb` |
| [`U-Net/`](U-Net/) | Encoder-decoder with skip connections for segmentation | `unet.py`, `unet.ipynb` |
| [`HRNet-Implementation/`](HRNet-Implementation/) | High-resolution network with parallel multi-scale branches | `hrnet.py` |
| [`EfficientNet implementation/`](EfficientNet%20implementation/) | EfficientNet-B0 with inverted residual blocks | `EfficientNet B0.py` |
| [`vision-transformer-implementation/`](vision-transformer-implementation/) | ViT, with the attention mechanism broken out separately | `vision_transformer.py`, `notebooks/vision-transformer.ipynb`, `notebooks/multi-head attn.ipynb` |
| [`BERT_implementation/`](BERT_implementation/) | BERT encoder | `bert.py` |
| [`tiny-GPT/`](tiny-GPT/) | Character-level GPT trained on Shakespeare | `nanoGPT.py`, `text_dataset.txt` |

## Classical methods and analysis

| Directory | Implementation |
|---|---|
| [`Classification/`](Classification/) | Linear discriminant analysis worked through on a small two-class set (`LDA.py`) |
| [`Dimensionality Reduction/`](Dimensionality%20Reduction/) | Principal component analysis (`PCA.py`) |
| [`Exploratory Data Analysis/`](Exploratory%20Data%20Analysis/) | A walk through the standard EDA plots on a tabular dataset (`eda.ipynb`) |

## At the repository root

[`Backpropagation from scratch for NN.ipynb`](Backpropagation%20from%20scratch%20for%20NN.ipynb)
derives and implements a network's backward pass by hand, without autograd. It sits at the
root rather than in a directory because it is a single self-contained notebook with no
accompanying module.

## Running them

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm
```

Most of the architecture modules are definitions only: import the model class and
instantiate it, rather than running the file for output. The notebooks are the ones that
run a model end to end.

`tiny-GPT/nanoGPT.py` reads its corpus by relative path, so run it from inside its
directory:

```bash
cd tiny-GPT
python nanoGPT.py
```
