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




---

## ⚙️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Tkinter / Streamlit (for UI)**
- **Scikit-learn**

---

## 🧩 How It Works

1. **Data Preprocessing**  
   The MNIST dataset is normalized and reshaped for CNN input.  
   Labels are one-hot encoded.

2. **Model Architecture**  
   - Convolutional Layers (feature extraction)  
   - MaxPooling Layers (dimensionality reduction)  
   - Dense Layers (classification)  
   - Softmax Output Layer (digit prediction)

3. **Training**  
   - Optimizer: `Adam`  
   - Loss Function: `categorical_crossentropy`  
   - Metrics: `accuracy`

4. **Evaluation**  
   The model’s accuracy and confusion matrix visualize its classification performance.

---

## 📈 Results

- **Training Accuracy:** ~99%  
- **Validation Accuracy:** ~98%  
- **Low Loss** and excellent generalization on unseen data.

---

## 📸 Example Output

### Confusion Matrix  
Visualizes how well the model classifies each digit.

### Interactive Canvas  
Allows users to draw digits and see predictions instantly.

---

## 🧑‍💻 Author

**Ewli Kodjo Gato Didier**  
🎓 Master’s Student in AI | Data Scientist | Deep Learning Enthusiast  
📍 Nairobi, Kenya 
📧 kodjoewli@gmail.com

---

## 📜 License

This project is released under the **MIT License**.  
Feel free to use, modify, and share it for learning or research purposes.

---

## 🌟 Acknowledgments

- **MNIST Dataset** — by *Yann LeCun et al.*  
- **TensorFlow / Keras** for model building.  
- **Matplotlib** for visualization and interactivity.

⭐ If you like this project, don’t forget to **star the repo** and share it!



