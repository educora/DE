# Decision Tree Implementation for Animal Classification

This document explains the Python script for predicting animal classifications using decision tree classifiers on the `zoo.csv` and `class.csv` datasets.

## Data Preparation
- **Loading Data:** The script loads the `zoo.csv` and `class.csv` datasets.
- **Merging Data:** It merges these datasets on the class type to include descriptive labels for the animal classes.

## Data Visualization
- Uses seaborn and matplotlib for visualizing the distribution of animal classes in the dataset.

## Feature Selection
- Selects features relevant to animal characteristics (e.g., hair, feathers, eggs) for model training.

## Model Training and Evaluation
- **Model Training:** Splits the data into training and test sets, and trains decision tree classifiers with different configurations.
  - **Model 1:** Uses all features with 20% of data for training.
  - **Model 2:** Same as Model 1 but with only 10% of data for training.
  - **Model 3:** Focuses on visible features (e.g., hair, feathers) with 20% of data for training.
  - **Model 4:** Limits the tree depth to reduce memory usage, using visible features and 20% of data for training.
- **Evaluation:** Evaluates the models using accuracy, confusion matrices, and classification reports. It also examines feature importance for insights into which characteristics are most influential in predictions.

## Results
- Presents a comparison of model performance metrics (accuracy, precision, recall, F1 score) to identify the most effective configuration.

## Conclusion
- Summarizes the findings and suggests the best model based on the evaluation metrics.

