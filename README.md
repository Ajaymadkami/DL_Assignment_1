# DL_Assignment_1

## Overview

This project focuses on training fully connected neural networks from scratch to classify images from both the Fashion-MNIST and MNIST datasets. The model is built using Python and experiments with different optimizers.
#### Datasets

Fashion-MNIST

The Fashion-MNIST dataset is a drop-in replacement for MNIST but contains images of clothing items instead of handwritten digits. It consists of:

- **60,000 training images and 10,000 test images**.

- **Each image is 28x28 pixels in grayscale.**

- **10 classes, representing different clothing items (e.g., T-shirt, sneakers, bag, etc.).**

MNIST

The MNIST dataset is a classic benchmark for image classification. It consists of:

- **60,000 training images and 10,000 test images.**

- **Each image is 28x28 pixels in grayscale.**

- **10 classes (digits 0-9).**
Training results are logged using Weights & Biases (WandB).
## WandB Report
Check out the detailed training logs and analysis in our [WandB Report](https://wandb.ai/ajay-madkami-iitm-indian-institute-of-technology-mad/Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTY5NzQ0MQ?accessToken=v0t50qbr5oilxh7nj17hciwicokuw71k8cyofh9nt8xont61uh0dulg4otg89q5t).
## GitHub Link
[GitHub Report](https://github.com/Ajaymadkami/DL_Assignment_1/tree/main)


Here's a `README.md` file for my project:

```markdown
# Neural Network Training with Fashion MNIST

This project implements a customizable neural network to train on the Fashion MNIST dataset using different optimizers and activation functions. The training process is logged using [Weights & Biases](https://wandb.ai/).

## Features

- Implements a fully connected neural network from scratch using NumPy.
- Supports various activation functions (`relu`, `tanh`, `sigmoid`).
- Includes multiple optimizers (`sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`, `nadam`).
- Uses Weights & Biases (wandb) for experiment tracking and hyperparameter tuning.
- Provides configurable network architecture (number of layers, nodes per layer, batch size, learning rate, etc.).

## Installation

Make sure you have Python installed, then install the required dependencies:

```bash
pip install numpy keras wandb
```

## Usage

1. Run the script to start training:

```bash
python train.py
```

The script initializes a Weights & Biases sweep to optimize hyperparameters and trains the model accordingly.

## Configuration

The training parameters are defined in `sweep_config`:

```python
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_hidden_layers': {'values': [3, 4, 5]},
        'num_hidden_nodes': {'values': [32, 64, 128]},
        'batch_size': {'values': [16, 32, 64]},
        'learning_rate': {'values': [0.0001, 0.001]},
        'activation': {'values': ['relu', 'tanh','sigmoid']},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'init_method': {'values': ['xavier', 'random']}
    }
}
```

You can modify these values to experiment with different configurations.

## Best Model Configuration

Through hyperparameter tuning, the best-performing model was found with the following settings:

- **Activation Function**: `tanh`
- **Batch Size**: `16`
- **Epochs**: `10`
- **Weight Initialization**: `Xavier`
- **Learning Rate**: `0.001`
- **Number of Hidden Layers**: `3`
- **Nodes per Hidden Layer**: `64`
- **Optimizer**: `Nadam`

## Confusion Matrix
Once we found the best hyperparameter configuration, we evaluated the model on the test dataset to determine its final accuracy. To get a clearer understanding of the model's performance, we plotted the confusion matrix, which shows how well the model predicts each class.
To visualize the confusion matrix, run Question 7 in DL_Assignment1.ipynb after finalizing the best model configuration. This will generate the plot with the true vs. predicted labels for the test data.


## Results

The script prints the loss and accuracy for both training and validation sets at each epoch:

```
Epoch 10/10, Train Accuracy: 0.90, Validation Accuracy: 0.88 Test Accuracy: 0.87.
```
