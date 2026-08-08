# Parkinsons_Disease_Detection
#### Developed an end-to-end Machine Learning application designed for early-stage detection of Parkinson's Disease. The system analyzes critical biomedical voice measurements (like fundamental frequency, jitter, and shimmer) to classify and predict whether an individual is affected by the disease with high precision.
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-RandomForest-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Tkinter-GUI-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
</p>

<h1 align="center">🧠 Parkinson's Disease Detection</h1>
<p align="center">
  <b>Machine Learning project to detect Parkinson's Disease from voice measurements</b><br>
  B.Tech CSE | VITM Berhampur | Ankita
</p>

---

## 📌 Project Overview

Parkinson's Disease is a progressive neurological disorder. Early detection significantly improves quality of life. This project uses **vocal biomarker features** (jitter, shimmer, harmonic ratios) collected via sustained phonation tests to predict whether a person has Parkinson's Disease.

> **Model:** Random Forest Classifier  
> **Accuracy:** ~95%  
> **Interface:** Tkinter Desktop GUI for real-time prediction

---

## 📁 Project Structure

```
Parkinsons_Disease_Detection/
│
├── parkinsons.csv          ← Dataset (195 voice recordings, 23 features)
├── train_model.py          ← Model training + feature importance graph
├── gui_app.py              ← Desktop GUI for prediction (Tkinter)
├── parkinson_model.pkl     ← Pre-trained Random Forest model
├── scaler.pkl              ← StandardScaler (generated after training)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🗃️ Dataset

| Property | Detail |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons) |
| Rows | 195 voice recordings |
| Features | 23 columns (22 features + 1 target) |
| Target | `status` (1 = Parkinson's, 0 = Healthy) |
| Class Distribution | ~75% Parkinson's · ~25% Healthy |

### 🎙️ Feature Categories

| Category | Features |
|---|---|
| **Frequency Measures** | MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz) |
| **Jitter (Pitch Variation)** | MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP |
| **Shimmer (Amplitude Variation)** | MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA |
| **Noise Ratio** | NHR, HNR |
| **Nonlinear Dynamics** | RPDE, DFA, spread1, spread2, D2, PPE |

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 trees |
| Preprocessing | StandardScaler |
| Train/Test Split | 80% / 20% |
| Accuracy | ~95% |

**Why Random Forest?**
- Handles high-dimensional voice feature data well
- Resistant to overfitting
- Provides feature importance rankings
- No need to tune many hyperparameters

---

## 🚀 How to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Parkinsons_Disease_Detection.git
cd Parkinsons_Disease_Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python train_model.py
```
This will:
- Train the Random Forest model
- Save `parkinson_model.pkl` and `scaler.pkl`
- Generate `feature_importance.png`

### Step 4: Launch the GUI App
```bash
python gui_app.py
```
> ⚠️ Note: `tkinter` is included with Python by default on Windows/Mac. On Linux, install with `sudo apt-get install python3-tk`

---

## 🖥️ GUI Preview

The desktop app lets you enter 22 voice measurement values and click **"Predict"** to get the result:

```
🔴 Likely to have Parkinson's Disease
🟢 Healthy
```

A **"Show Feature Importance Graph"** button is also available to visualize which voice features matter most.

---

## 📊 Feature Importance

After training, `feature_importance.png` is generated showing the top predictive features. Key contributors include:
- `PPE` (Pitch Period Entropy)
- `spread1` (Nonlinear measure of fundamental frequency)
- `MDVP:Fo(Hz)` (Average vocal fundamental frequency)
- `DFA` (Signal fractal scaling exponent)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Machine Learning | Scikit-Learn (Random Forest) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Model Persistence | Joblib |
| GUI | Tkinter |

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. It is **not a medical diagnostic tool**. Always consult a qualified medical professional for clinical diagnosis.

---

## 👩‍💻 Author

**Ankita**  
B.Tech CSE (Data Science) | Vignan Institute of Technology and Management, Berhampur, Odisha  
Batch 2022–2026 | CGPA: 8.6

---

## 📄 License

This project is open-source and available for educational use.

---

<p align="center">Made with ❤️ for early detection of Parkinson's Disease | VITM 2026</p>
