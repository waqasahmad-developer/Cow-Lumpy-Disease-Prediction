# 🐄 Cow Lumpy Disease Prediction using Deep Learning

I built this project to predict **Lumpy Skin Disease in cows** using deep learning and computer vision. It features an end-to-end pipeline from model training to a deployed web app using **Streamlit**.

---

## 📁 Dataset

The dataset contains approximately **1000 images**:
- Healthy cows
- Cows infected with Lumpy Skin Disease

I curated the **test dataset manually** to ensure diverse representation.

---

## 🧪 Workflow

### 📓 Notebook 1: CNN_From_Scratch.ipynb
I trained a Convolutional Neural Network (CNN) from scratch. However, the model didn't perform well due to the small dataset size and overfitting.

### 📓 Notebook 2: Transfer_learning.ipynb
I applied **MobileNet**, a pretrained model on ImageNet, using transfer learning. This approach significantly improved performance and achieved **96% accuracy** on the test set.

---

## 🚀 Deployment

I deployed the model using **Streamlit Cloud**, creating an interactive and user-friendly web interface.

### 🔗 [Live App](#)
*(https://cow-lumpy-disease-prediction.streamlit.app/)*

---

## ⚙️ Features

- Upload an image of a cow and get an instant prediction (Healthy Cow or Lumpy Cow)
- Clean, centered UI with GitHub and LinkedIn buttons for networking

---
