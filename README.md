## Overview
This repository provides an implementation of the Vision Transformer (ViT) from scratch. ViT is a deep learning model designed for image recognition tasks, leveraging self-attention mechanisms to process images efficiently. This project is a personal initiative aimed at gaining a deeper understanding of transformer models in the context of computer vision. It is built with the intention to explore and experiment with the architecture of ViT, and offers a modular and extensible implementation for further research and learning.

## Table of Contents
- [Architecture](#architecture)
  - [Components](#components)
- [Requirements](#requirements)
  - [Installation](#installation)
- [Explainability](#explainability)
  - [Attention Maps](#attention-maps)
- [Training](#training)
  - [Configuration](#configuration)
- [Dataset](#dataset)
  - [MNIST](#mnist)
  - [Tomato Leaf Diseases](#tomato-leaf-diseases)
- [Acknowledgments](#acknowledgments)

## Architecture

The Vision Transformer (ViT) architecture follows the transformer model used for natural language processing, adapted for images. It divides each image into fixed-size non-overlapping patches, linearly embeds each patch, and adds positional encodings. These embeddings are then processed by the transformer encoder, which consists of multiple layers of self-attention and feed-forward networks.

![vit-architecture](documentation/imgs/vit-architecture.png)

### Components

1. **Patch Embedding**: Images are split into patches, and each patch is flattened into a vector, which is linearly projected into a higher-dimensional space. 
2. **Positional Encoding**: Since transformers do not inherently capture spatial information, positional encodings are added to the patch embeddings to retain the spatial structure of the image.
3. **Transformer Encoder**: The patch embeddings (along with positional encodings) are passed through several layers of multi-head self-attention and feed-forward networks. These layers help the model capture relationships between different parts of the image.
4. **Classification Head**: After processing through the transformer encoder, the final representation is passed to a classification head (a simple fully connected layer) to predict the class of the image.

## Requirements
This repository requires the following software to run:
- **Python Version**: 3.12.9
- **PyTorch Version**: 2.6

Additional dependencies are specified in [`environment/requirements.txt`](environment/requirements.txt), which includes other necessary libraries.

### Installation
It is recommended to use Conda for dependency management. To create the required environment, execute the following command:

```sh
conda env create -f environment/environment.yml
```

> If Conda is not installed, it can be obtained from:
> - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
> - [Anaconda](https://www.anaconda.com/)

To activate the environment, use:

```sh
conda activate CV-Transformer
```

## Explainability

This section discusses interpretability features of the model to help users understand how the Vision Transformer makes decisions.

### Attention Maps
The model includes functionality to visualize attention maps, offering insights into how the transformer processes and attends to different regions of an input image. By plotting these attention maps, users can gain a better understanding of which areas of the image the model focuses on when making predictions. This can be especially useful for debugging the model or improving its accuracy by identifying biases or areas of the image that are being ignored.

## Training
To train the model, use the following command:

```sh
python3 train.py [-h] --config CONFIG [--experiment-name EXPERIMENT_NAME]
```

Where:
- **`--config CONFIG`**: Specifies the path to the YAML configuration file containing the training parameters.
- **`--experiment-name EXPERIMENT_NAME`**: Defines the name of the experiment, which is used for logging and saving results.

### Configuration
The training process is governed by a YAML configuration file, located in the [`configs/`](configs/) directory. An example configuration is shown below:

```yaml
device: "cuda:0"

data:
  name: 'MNIST'
  class_path: 'dataset.mnist_dataset.MNISTDataset'
  base_path: "dataset/mnist"
  img_height: 28
  img_width: 28
  n_channels: 1
  num_classes: 10

model:
  cnn_backbone: False
  cnn_out_channels: 32
  patch_size: 2
  latent_size: 120
  num_heads: 12
  num_encoders: 3
  dropout: 0.1

training:
  epochs: 20
  lr: 0.001
  batch_size: 16

logger:
  base_path: './logs'
  step: 100
  max_grid_dim: 16
```

This structure allows for flexible adjustments to model architecture, dataset selection, and training hyperparameters. You can modify the configuration to suit different datasets, model architectures, or training strategies.

- **`device`**: Specifies the computing device (e.g., "cuda:0" for GPU, "cpu" for CPU execution).
- **`data`**: Defines dataset parameters such as name, file path, image dimensions, and the number of classes.
- **`model`**: Specifies the model architecture, including patch size, latent dimension, number of heads, encoder layers, and dropout rate.
- **`training`**: Contains hyperparameters such as number of epochs, learning rate, and batch size.
- **`logger`**: Configures logging settings, including log storage path and step intervals.

## Dataset

This section details the datasets supported by the project, which can be used for training and evaluation.

### MNIST
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. It is widely used for benchmarking classification algorithms.

- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Classes**: 10
- **Dataset Link**: [MNIST](http://yann.lecun.com/exdb/mnist/)

### Tomato Leaf Diseases
The Tomato Leaf Diseases dataset contains images of tomato leaves affected by various plant diseases. It is used for training models in agricultural disease detection.

- **Categories**: Multiple plant diseases
- **Image Type**: RGB
- **Dataset Link**: [Tomato Leaf Diseases](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf/data)

## Acknowledgments

The code in this repository is based on the work of the original Vision Transformer paper. If you use this implementation in your research, please cite the following paper:

```bibtex
@inproceedings{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```