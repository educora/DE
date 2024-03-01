# PyTorch Code for Image Detection using Convolutional Neural Networks (CNNs)

This document explains the provided Python script which utilizes PyTorch, a popular deep learning library, for image detection tasks. The script covers the entire process from data preparation, model selection, training, to prediction.

## Importing Libraries

The script begins by importing necessary libraries and modules:

- `torch` and related modules for deep learning functionalities.
- `torchvision` for datasets, model architectures, and image transformations.
- `matplotlib.pyplot` for plotting and visualizing data.
- `PIL.Image` for image file operations.
- `os`, `time`, and `copy` for file path operations, timing, and deep copying objects, respectively.

## ImageDetection Class

A class named `ImageDetection` is defined to encapsulate all the functionalities related to loading models, transforming images, and predicting classes of images.

### Attributes:

- `torch_models`: A list of available model names in PyTorch.
- `dataset` and `class_labels`: Directory paths for the dataset and class labels.
- `device`: Determines if a GPU (with CUDA) or CPU is used for computations.

### Methods:

- `__init__`: Initializes the class with the dataset and class labels directory.
- `load_model`: Loads a pre-trained model from PyTorch's model zoo.
- `load_model_from_disk`: Loads a model saved on the disk.
- `transform`: Defines the image transformations for preprocessing.
- `predict`: Predicts the class for a given image.
- `train_model`: Trains the model with specified parameters.

## Utility Functions

- `imshow`: A function to display an image tensor.
- `train_model` (outside the class): Trains a given model using specified criteria, optimizer, and scheduler.
- `visualize_model`: Displays images along with their predicted classes.
- `predict_image`: Predicts the class of a single image.

## Main Procedure

In the `if __name__ == '__main__':` block, the script demonstrates a complete workflow:

1. **Data Preparation**: It sets up data transformations for training and validation sets, loads the datasets, and prepares data loaders.

2. **Model Preparation**: A pre-trained ResNet18 model is loaded, and its final layer is modified to match the number of classes in the dataset. The model is then moved to the appropriate device (GPU or CPU).

3. **Training**: The model is trained using the defined criterion, optimizer, and learning rate scheduler. The training involves iterating over epochs and optimizing the model parameters based on the training and validation datasets.

4. **Saving and Loading Model**: The trained model is saved to disk and then loaded for later use.

5. **Prediction**: The script demonstrates how to predict the class of new images using both the trained model and a model loaded from disk.

6. **Visualization**: Although the visualization function calls are commented out, they can be used to display a grid of training images or visualize predictions.

This script showcases how to utilize PyTorch for building, training, and deploying an image detection model, including handling datasets, using pre-trained models, and implementing custom training loops.
