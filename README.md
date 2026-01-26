# 🖐️ Real-Time End-to-End Sign Language Recognition
data link "https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset"
 
An end-to-end **American Sign Language (ASL) recognition system** using a custom
Convolutional Neural Network (CNN), FastAPI for inference, and Streamlit for
frontend visualization.

The system supports **real-time prediction**, optimized to perform inference
**every 50 frames** for better performance and stability.

---

## 📌 Features

- Custom CNN architecture trained on ASL hand gesture images
- 26-class classification (digits, alphabets, and common phrases)
- Real-time inference with frame skipping
- REST API using FastAPI
- Streamlit-based frontend
- Modular and scalable project structure

---

## 🧠 Supported Classes

The model classifies the following **26 ASL classes**:

```python
CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','bye','c','d','e','good','good morning',
    'hello','little bit','no','pardon','please',
    'project','whats up','yes'
]

🏗️ Model Architecture

Input: 128 × 128 RGB image

Backbone: Custom CNN

Blocks:

Conv2D → BatchNorm → ReLU

MaxPooling

Depth: 4 convolutional blocks (32 → 64 → 128 → 256 filters)

Regularization: L2 + Dropout

Pooling: Global Average Pooling

Output: Dense layer with Softmax (26 classes)

Loss: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy

📁 Project Structure

```│
│
├── data/
│   ├── data_loader.py
│   ├── data_explore.py
│   └── data_handle.py
│
├── models/
│   └── ASL.keras           # trained model 
│
├── notebooks/
│   └── hand-gesture-based-sign-language-detection.ipynb
│
├── src/
│   ├── API/
│   │   └── API.py          # FastAPI inference service
│   │
│   ├── frontend/
│   │   └── streamlit_app.py
│   │
│   ├── inference/
│   │   └── inference.py    # real-time webcam inference
│   │
│   ├── model/
│   │   ├── architecture.py
│   │   ├── callbacks.py
│   │   ├── evaluate.py
│   │   └── train.py
│   │
│   └── utils/
│       ├── plot_cm.py
│       └── plot_history.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

🚀 Running the Project

1️⃣ Install Dependencies
``` pip install -r requirements.txt ```

2️⃣ Run FastAPI Backend
``` uvicorn API:app --reload ```

Swagger UI:
``` http://127.0.0.1:8000/docs ```
 
3️⃣ Run Streamlit Frontend
``` streamlit run streamlit_app.py ```

🧪 Model Training
Model training is handled in:

``` src/model/train.py ```

# Includes:

Data loading

Label encoding

Train/validation split

Data augmentation

Early stopping & learning rate scheduling


📊 Evaluation

Confusion Matrix

Classification Report

Accuracy & Loss plots

# Implemented in:

``` src/model/evaluate.py```

``` src/utils/plot_cm.py ```

``` src/utils/plot_history.py ```

🌐 Deployment Strategy

Backend: FastAPI 

Frontend: Streamlit 


