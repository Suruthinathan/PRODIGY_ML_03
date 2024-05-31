# Image Classification Project: Cats vs. Dogs
This repository contains the code for an image classification project to differentiate between images of cats and dogs using a Support Vector Machine (SVM) classifier.The dataset used is from [Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

# Project Overview
The goal of this project is to build a machine learning model capable of classifying images of cats and dogs. This is achieved through the following steps:

- Data Collection: Gather a dataset of cat and dog images.
+ Data Preprocessing: Resize and normalize images, and split the dataset into training, validation, and test sets.
* Dimensionality Reduction: Apply Principal Component Analysis (PCA) to reduce the dimensionality of the image data, making it more manageable for the model.
- Model Training: Train a Support Vector Machine (SVM) classifier on the processed data.
* Model Evaluation: Evaluate the model's performance using accuracy, confusion matrix, and classification report.
+ Prediction: Use the trained model to make predictions on new, unseen images and display the results.

# Dataset
The dataset used in this project consists of images of cats and dogs. Here are some important details about the dataset.The training dataset consists of 25,000 images (12,500 images of cats and 12,500 images of dogs). This balanced dataset helps in training a model that does not favor one class over the other.

# Prerequisites
Ensure you have the following libraries installed:

- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Scikit-learn
- Seaborn
- Plotly
- Numpy
- Pandas
- Skimage
- PIL

You can install these libraries using pip:
```
pip install pandas scikit-learn matplotlib
```

# Results
After training the SVM model, the following results were achieved:

+ Accuracy
- Classification Report
* Confusion matrix

# Acknowledgements
The project uses the dataset from the [Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) competition on Kaggle.
