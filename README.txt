# 🧠 Handwritten Digit Recognition using CNN

This project demonstrates a **Convolutional Neural Network (CNN)** model built using **TensorFlow and Keras** to recognize handwritten digits (0–9) from the **MNIST dataset**.  
It also includes an **interactive drawing interface** that allows users to draw a digit and see real-time predictions made by the trained model.

---

## 📘 Project Overview

The goal of this project is to classify handwritten digits using deep learning.  
The model learns features from thousands of labeled images and can predict digits with high accuracy.  
In addition, it includes an interactive canvas that lets users test the model by drawing custom digits.

---

## ⚙️ Features

- Loads and preprocesses the MNIST dataset.  
- Normalizes image data for better model performance.  
- Builds a **CNN** architecture with convolution, pooling, and dense layers.  
- Visualizes sample digits, training progress, and confusion matrix.  
- Implements an **interactive digit drawing tool** using Matplotlib.  
- Predicts drawn digits in real-time.

---

## 🧩 Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- idx2numpy  
- KaggleHub (for dataset download)

---

## 📊 Model Architecture

```text
Input (28x28x1)
│
├── Conv2D (32 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (128, ReLU)
├── Dropout (0.3)
└── Dense (10, Softmax)
