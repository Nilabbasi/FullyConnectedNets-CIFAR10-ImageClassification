# Deep Learning Project with *Fully Connected Networks* on CIFAR-10 Dataset 

## Overview
This project demonstrates a deep learning workflow using the CIFAR-10 dataset for image classification. 
It includes custom implementations of gradient checks, preprocessing steps, and provides visualization and model training functions. 
The architecture primarily focuses on **fully connected networks** with dropout for regularization.

## Table of Contents
- [Project Title](#project-title)
- [Overview](#overview)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)

## File Descriptions

### **libs/\_\_init\_\_.py**
This file imports key modules from the project, including `data.py`, `grad.py`, and the `Solver` from `solver.py`.

### **libs/data.py**
Provides essential dataset management and preprocessing functions for the CIFAR-10 dataset.
- **reset_seed()**: Resets the random seed for reproducibility.
- **tensor_to_image()**: Converts PyTorch tensors into NumPy arrays for visualization.
- **_extract_tensors()**: Extracts and converts dataset images and labels into tensors.
- **cifar10()**: Downloads and processes the CIFAR-10 dataset for training and testing.
- **preprocess_cifar10()**: Preprocesses the dataset with normalization, reshaping, and optional bias trick. The function can also display sample images from the dataset.

### **libs/grad.py**
Provides utilities for gradient checking and numerical gradient computation.
- **grad_check_sparse()**: Computes sparse numeric gradient checks for a given function.
- **compute_numeric_gradient()**: Computes the numeric gradient of a function using the centered difference method.
- **rel_error()**: Computes the relative error between two tensors to validate gradient calculations.

### **libs/solver.py**
Contains the implementation of the `Solver` class, managing the training process for classification models using stochastic gradient descent (SGD) and other update rules. 
It performs tasks such as managing datasets, tracking losses, checking accuracy, and saving model checkpoints.

#### Key Features:
- **Training Management:** Encapsulates the logic for training classification models using SGD and other update rules.
- **Gradient Descent:** Updates model parameters using the provided update rule (default is SGD) and keeps track of the loss history during training.
- **Accuracy Checks:** Periodically evaluates the model's accuracy on training and validation datasets to monitor overfitting and adjust hyperparameters.
- **Checkpointing:** Saves model checkpoints at the end of each epoch to allow for recovery and analysis.
- **Learning Rate Decay:** Supports learning rate decay to adjust the learning rate over time for better optimization.

#### Class Structure:
  - `Solver`: The core class handling the training process.
  - `__init__()`: Initializes the solver with the model, dataset, and training hyperparameters such as learning rate, batch size, number of epochs, and more.
  - `_step()`: Performs a single gradient update by calculating the loss and adjusting the model parameters.
  - `train()`: Runs the optimization procedure and trains the model, returning the parameters that achieved the best validation accuracy.
  - `check_accuracy()`: Evaluates model accuracy on a dataset.
  - `_save_checkpoint()`: Saves model checkpoints during training.

### **fully_connected_networks.py**
Implements a fully connected neural network using PyTorch. This module focuses on linear transformations, ReLU activations, and softmax classification.
The structure follows a clean modular design that separates forward and backward passes for individual layers, making it easy to extend or modify for different architectures. 

The important components included:
- **Linear Layer**: Implements both forward and backward passes for a fully connected layer.
- **ReLU Layer**: Implements the non-linear activation function and its backward pass.
- **Linear_ReLU Convenience Layer**: A combination of a linear layer followed by a ReLU layer.
- **Softmax Loss**: Calculates the softmax loss and its gradient, useful for multi-class classification.
- **Two-Layer Network Class**: Handles the network's overall structure, initialization, saving/loading, and computation of the loss and gradients.
- **FullyConnectedNet Class**: The `FullyConnectedNet` class implements a fully connected neural network that supports an arbitrary number of hidden layers with ReLU activations and an optional dropout for regularization. The network's output is passed through a softmax layer to compute the final classification loss. It also includes methods to handle both forward and backward passes for efficient training.

  **Key Components:**
  - **Initialization:**  
    The network is initialized based on specified parameters such as input dimensions, hidden layers, number of classes, dropout probability, and L2 regularization strength. The weights are randomly initialized using a normal distribution scaled by `weight_scale`, and the biases are initialized to zero. The network architecture can be modified by providing a list of hidden layer dimensions.
  
  - **Forward Pass:**  
    The forward pass computes the class scores for the input data, propagating through linear and ReLU layers. If dropout is enabled, it is applied during the forward pass to regularize the network. The scores are computed at the final layer using a softmax loss function.

  - **Loss Computation and Backward Pass:**  
    The loss function combines the softmax loss and L2 regularization to minimize both classification error and overfitting. Gradients for each layer are computed during the backward pass to update the weights and biases using optimization algorithms such as Stochastic Gradient Descent (SGD), RMSprop, or Adam.

  - **Dropout Handling:**  
    The `Dropout` class manages the dropout layer's forward and backward passes. Dropout is applied during training to randomly set some neuron outputs to zero and scale the remaining outputs to prevent co-adaptation of neurons.

**Optimization Methods:**
- `sgd`: Standard stochastic gradient descent.
- `sgd_momentum`: Gradient descent with momentum for faster convergence.
- `rmsprop`: Uses adaptive learning rates based on a moving average of squared gradients.
- `adam`: Combines momentum and RMSprop techniques to adaptively adjust learning rates and improve convergence.


### **fully_connected_networks.ipynb**

A Jupyter notebook that showcases the implementation and training of fully connected neural networks on the CIFAR-10 dataset, including experiments with dropout and various optimization techniques.

**Google Colab Setup**
Since this notebook runs on Google Colab, I am mounting Google Drive to access my data and necessary scripts. This allows seamless integration with `.py` files, which can be dynamically reloaded using the autoreload extension.

## Installation
To run this project, you need Python 3.x and the following libraries:
- PyTorch
- NumPy
- Matplotlib

You can install the required libraries using pip:
```bash
pip install torch torchvision numpy matplotlib
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/   i HAVE TO WRITE MY REPO NAME HERE / FINAL NAME :).git
   cd MY-repo-name
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook fully_connected_networks.ipynb
   ```

## Experiments
The project includes several experiments to evaluate the performance of different architectures and optimization techniques. Key experiments include:
- **Overfitting with Various Network Depths**: Training simple three-layer and five-layer networks on a small dataset.
- **Comparing Optimization Techniques**: Implementing and evaluating SGD, SGD with momentum, RMSProp, and Adam optimizers.
- **Dropout Regularization**: Implementing dropout to reduce overfitting.
