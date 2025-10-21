# 🌾 Rice Leaf Disease Detection Using CNN

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange?logo=streamlit) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-red?logo=tensorflow) 
![Keras](https://img.shields.io/badge/Keras-2.15-purple?logo=keras) 
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deploy-blue?logo=huggingface) 
![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github) 
![License](https://img.shields.io/badge/License-MIT-green)  

This project uses **Deep Learning (CNN)** to detect diseases in rice leaves from uploaded images.  
It helps farmers identify diseases early and take preventive measures to improve crop yield.

---

## 🧠 Project Overview

- **Goal:** Automatically detect rice leaf diseases using image recognition  
- **Frontend & Deployment:** Streamlit, Hugging Face Spaces  
- **Backend & ML:** Python, TensorFlow, Keras  
- **Dataset:** Rice leaf images (Healthy + Diseased)  
- **Output:** Predicted disease name with confidence percentage  

---

## 💻 Features

✅ Upload an image of a rice leaf  
✅ Model analyzes and predicts the disease  
✅ Displays disease name and confidence percentage  
✅ User-friendly interface for farmers  
✅ Extendable for AI-based suggestions or chatbot integration  

---

## 🧩 Diseases Detected

| Label | Disease Name |
|-------|----------------|
| 0     | Bacterial Leaf Blight |
| 1     | Leaf Blast |
| 2     | Brown Spot |
| 3     | Tungro |

---

## ⚙️ Tech Stack

| Layer         | Technology               |
|---------------|--------------------------|
| Frontend      | Streamlit                |
| Backend       | Python                   |
| ML Model      | TensorFlow, Keras        |
| Deployment    | Hugging Face Spaces      |
| Version Control | Git & GitHub            |

---
## 📊Dataset
- https://data.mendeley.com/datasets/fwcj7stb8r/1
---
## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/RuchithYK/Rice-Disease-Detection-Using-CNN.git
cd Rice-Disease-Detection-Using-CNN

# Create virtual environment
python -m venv venv

# Activate environment
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app1.py
```
Your app will open in the browser at http://localhost:8501
---
## 🌐 Live Demo
- https://huggingface.co/spaces/RuchithYK/Rice_leaf_Disease_Detection
---
## ⚠️ Note
- Use clear, well-lit rice leaf images for accurate predictions
- Model works best on patterns seen during training
- Easily extendable to include more diseases
---


