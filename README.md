# Hybrid-Model-Of-VGG-19-ResNet-50

# 🧠 Hybrid Deep Learning Model (VGG19 + ResNet50) for Oral Health Classification

## 📌 Overview
This repository contains the implementation of a **Hybrid Convolutional Neural Network** combining **VGG19** and **ResNet50** architectures for classifying oral health conditions into three categories:
- **Dental Caries**
- **Mouth Ulcer**
- **Dental Calculus**

By leveraging **transfer learning**, **feature concatenation**, and **fine-tuning**, the model achieves **high accuracy and generalization capability**.

---

## 🚀 Key Results
| Metric               | Score |
|----------------------|-------|
| **Training Accuracy**| **99.41%** |
| **Validation Accuracy** | **96.91%** |
| **Loss Trend**       | Converged smoothly with minimal overfitting |

---

## 🛠️ Features
- Hybrid feature extraction using **VGG19** + **ResNet50**
- **Transfer learning** from ImageNet weights
- **Dropout & Batch Normalization** for regularization
- **EarlyStopping**, **ReduceLROnPlateau**, and **ModelCheckpoint** callbacks
- Dataset preprocessing with **ImageDataGenerator** (augmentation)
- Model saved as **`best_model.h5`** using Keras

---

## 📂 Dataset
The dataset contains oral cavity images divided into:
- **Dental caries**
- **Ulcer**
- **Calculus**

> Images were resized to `224x224` pixels and normalized to `[0, 1]`.

---

## 📊 Training Graphs
**Accuracy Curve**
![Accuracy Curve](assets/accuracy_plot.png)

**Loss Curve**
![Loss Curve](assets/loss_plot.png)

---

## 📜 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
