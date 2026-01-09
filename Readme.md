# Grapevine Leaf Image Classification using VGG16 & VGG19

This repository presents a deep learning–based image classification system for **grapevine leaf images** using **Convolutional Neural Networks (CNNs)**. The project applies **transfer learning** with **VGG16 and VGG19** architectures to achieve effective classification performance on a limited dataset.

---

## Project Description

Grapevine leaf classification is an important task in agricultural image analysis and plant health monitoring. This project demonstrates how pre-trained CNN models can be fine-tuned for classifying grapevine leaf images. The implementation is provided as a single Jupyter Notebook that covers data preprocessing, model training, and evaluation.

---

## Objectives

- Develop a CNN-based image classification model for grapevine leaves  
- Utilize transfer learning with VGG16 and VGG19  
- Compare the performance of both architectures  
- Visualize training and validation metrics  

---


---

## Dataset

The dataset consists of grapevine leaf images arranged in class-wise directories. 

**Dataset requirements:**
- Images should be in `.jpg` or `.png` format  
- Each class must have its own folder  
- Folder name should represent the class label  

> **Note:** The dataset is not included in this repository and must be added manually.

---

## Methodology

### 1. Data Preprocessing
- Image resizing to match model input size  
- Pixel normalization  
- Data augmentation to improve generalization  

### 2. Model Architecture
- VGG16 and VGG19 pre-trained on ImageNet  
- Base model layers frozen initially  
- Custom fully connected layers added  
- Softmax activation for multi-class output  

### 3. Training Strategy
- Transfer learning approach  
- Training and validation split  
- Adam optimizer and categorical cross-entropy loss  

### 4. Evaluation
- Accuracy and loss visualization  
- Comparison between VGG16 and VGG19 performance  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook  

---

## Results

- The VGG16 and VGG19 models successfully learned discriminative features from grapevine leaf images.  
- Training and validation accuracy showed consistent improvement across epochs.  
- Loss curves indicate stable convergence with no significant overfitting.  
- Comparative analysis highlights the performance difference between VGG16 and VGG19 in terms of accuracy and training efficiency.  
- The results demonstrate the effectiveness of transfer learning for agricultural image classification tasks.

---

## Future Scope

- Incorporate a confusion matrix and detailed classification report (precision, recall, F1-score).  
- Experiment with lightweight architectures such as MobileNet or EfficientNet for faster inference.  
- Deploy the trained model as a web or mobile application for real-time prediction.  
- Expand the dataset to include more grapevine varieties and disease classes.  
- Apply advanced fine-tuning and hyperparameter optimization techniques to further improve accuracy.

---
