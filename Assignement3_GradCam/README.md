# Deep Learning Image Classification with CNN and Grad-CAM

## Overview

This project implements Convolutional Neural Networks (CNNs) for image classification using PyTorch. The models are trained on an image dataset and include Grad-CAM visualization to highlight important regions in the images.

## Features

- Implements two CNN architectures optimized for CUDA T4.
- Trains and validates models using an image dataset in `ImageFolder` format.
- Uses Grad-CAM for visualization to interpret model predictions.

## Model Architectures

### Deep CNN

- 5 convolutional layers with increasing filter sizes (64, 128, 256, 512, 1024).
- ReLU activation after each convolution.
- Max pooling applied to reduce spatial dimensions.
- Adaptive average pooling for fixed-size feature extraction.
- Fully connected layers for classification with dropout regularization.

### Alternative CNN Model

- 4 convolutional layers with filter sizes (32, 64, 128, 256).
- Batch normalization after each convolution for stable training.
- Max pooling for downsampling.
- Global average pooling before classification.
- Fully connected layers with ReLU activation and dropout.

## Training

The models are trained using:

- **Loss Function:** CrossEntropy Loss
- **Optimizer:** Adam (learning rate = 0.001)
- **Epochs:** 10
- **Batch Size:** 32

Training steps:

1. Load dataset and split into training (80%) and validation (20%).
2. Perform forward and backward propagation.
3. Compute training and validation accuracy per epoch.

## Grad-CAM Visualization

Grad-CAM helps visualize the important regions in an image that influence the model's prediction. The process:

1. Extracts gradients from the last convolutional layer.
2. Computes a weighted heatmap based on the gradients.
3. Overlays the heatmap onto the original image for interpretability.

Usage:

```python
heatmap = get_gradcam(model, sample_img, predicted_class)
```

## Results

- The trained models achieve competitive validation accuracy.
- Grad-CAM visualization enhances interpretability by highlighting significant image regions.

## Usage

1. Prepare the dataset in `ImageFolder` format.
2. Train the model by running the script.
3. Use Grad-CAM for visualization.

## License

This project is open-source and available for use under the MIT License.
