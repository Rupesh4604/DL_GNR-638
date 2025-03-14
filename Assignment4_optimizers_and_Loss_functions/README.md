# CNN Optimizer Comparison with W&B

This repository contains a PyTorch implementation to compare the performance of different optimizers on a Convolutional Neural Network (CNN) using a UC Merced Dataset . The experiments are tracked using **Weights & Biases (W&B)** for real-time monitoring, logging, and visualization of training metrics such as loss and training time. The code runs in Google Colab with GPU support for accelerated training.

## Overview

The project evaluates the following optimizers with various hyperparameters:

- **SGD**: Momentum values of 0.5, 0.9, and 0.95.
- **Adam**: Beta configurations of `(0.9, 0.999)`, `(0.8, 0.99)`, and `(0.95, 0.9)`.
- **RMSprop**: Default configuration with momentum 0.9.
- **Adagrad**: Default configuration.

Each optimizer configuration is treated as a separate experiment, run sequentially, with results logged to W&B and plotted locally using Matplotlib.

## Features

- **GPU Acceleration**: Utilizes Colab’s GPU (e.g., NVIDIA Tesla T4) for faster training.
- **W&B Integration**: Logs batch-level and epoch-level losses, training time, and model gradients/parameters.
- **Sequential Experiments**: Runs one experiment at a time to ensure stability and data persistence.
- **Visualization**: Generates loss curves for comparison across optimizers.

## Plots 

![](Assignment4_optimizers_and_Loss_functions/epoch_loss.png)
Loss per Epoch

![](Assignment4_optimizers_and_Loss_functions/batch_loss.png)
Batch Loss

![](Assignment4_optimizers_and_Loss_functions/training_time.png)
Total training Time
