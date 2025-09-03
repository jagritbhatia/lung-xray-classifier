# lung-xray-classifier
CNN-based Pneumonia Detection from Chest X-ray Images

# Chest X-ray Classification using CNN Variations

This project implements and experiments with Convolutional Neural Networks (CNNs) for classifying **chest X-ray images** into two categories:

* **NORMAL (healthy lungs)**
* **PNEUMONIA (infected lungs)**

The work follows concepts and examples from *Computer Vision Projects with PyTorch (Apress, 2022)* and expands them with custom CNN variations, augmentation, and batch normalization.

---
## 📷 Sample Images

Here are a few sample chest X-ray images from the dataset:

![Normal](images/Normal.png)
![Pneumonia](images/Pneumonia.png)

---

## 📊 Model Performance Screenshots

### 1. Baseline CNN
![Baseline Model Output](images/baseline_output.png)

### 2. CNN with Augmentation
![Augmentation Model Output](images/augmentation_output.png)

### 3. CNN with Batch Normalization
![BatchNorm Model Output](images/batchnorm_output.png)

## 📂 Dataset

Dataset source: [Kaggle – Labeled Chest X-ray Images](https://www.kaggle.com/tolgadincer/labeled-chest-xray-images)

Structure:

```
train/NORMAL      (1349 images)
train/PNEUMONIA   (3883 images)
test/NORMAL       (234 images)
test/PNEUMONIA    (390 images)
```

---

## ⚙️ Project Workflow

1. **Data Exploration**

   * Count samples in each category
   * View random images for sanity check
   * Inspect image dimensions

2. **Data Preprocessing & Augmentation**

   * Resize and center-crop to 224×224
   * Convert to tensors
   * Normalize using ImageNet mean & std
   * Training augmentation variations:

     * Color jitter
     * Random horizontal flip
     * Random rotation

3. **Model Variations**

   * **Baseline CNN**: Stacked conv–ReLU–pooling layers with log\_softmax output
   * **CNN with Augmentation**: Improved generalization using data augmentation
   * **CNN with Batch Normalization**: Added BatchNorm layers after convolutions for stable training and higher accuracy

4. **Training & Evaluation**

   * Loss function: Negative Log Likelihood Loss (NLLLoss)
   * Optimizer: SGD with momentum
   * Learning rate scheduling: StepLR
   * Metrics tracked:

     * Training loss & accuracy
     * Test loss & accuracy
   * Visualization: Matplotlib plots for loss and accuracy trends

---

## 🧠 Model Architectures

### Baseline CNN

* Multiple convolutional blocks with ReLU
* 1×1 and 3×3 convolutions
* MaxPooling layers
* Global Average Pooling before final layer
* Output: 2 classes (NORMAL, PNEUMONIA)

### CNN with Batch Normalization

* Same as baseline
* BatchNorm2d added after each conv–ReLU
* Improved stability and accuracy

---

## 📊 Results Summary

* **Baseline CNN**
  Accuracy fluctuated, unstable training, test accuracy around \~40%

* **CNN with Augmentation**
  Accuracy peaked at \~80% in early epochs, then dropped due to instability

* **CNN with Batch Normalization**
  Accuracy reached \~90% by epoch 10 and stabilized around \~85%
  Model size increased slightly (from \~12MB to \~17MB)
  Parameters: \~8,732

---

## 📦 Dependencies

The following libraries were used in the notebook:

```python
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm
```

---

## 🚀 How to Run

1. Clone this repository
2. Download and unzip the dataset into a folder `chest_xray/`
3. Open and run `ChestXray_Classification_CNN_Variations.ipynb` in Jupyter/Colab
4. Adjust paths in the notebook to match your dataset location
5. Train and evaluate different CNN variations

---



These help visualize learning progress across epochs.

---


