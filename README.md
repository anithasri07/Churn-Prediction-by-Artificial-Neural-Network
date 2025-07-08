# Churn-Prediction-by-Artificial-Neural-Network

# 📊 Customer Churn Prediction using TensorFlow & Keras

## 📝 Description

This project builds a machine learning model to predict **customer churn** (i.e., whether a customer will leave a service). It uses a **deep learning binary classification model** built with **TensorFlow** and **Keras**, trained on a dataset with **27 columns** (26 input features + 1 target column).

This is especially useful in scenarios like telecom, banking, or SaaS products, where retaining customers is critical.

---

## 🧠 Model Overview

### 🔨 Architecture

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
