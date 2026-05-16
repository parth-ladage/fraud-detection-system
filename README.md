# 🛡️ Real-Time Fraud Detection System with Explainable AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3.0-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)

**Live Dashboard:** [👉 Click here to view the live Streamlit Dashboard](#) *(Replace this # with your actual Streamlit link once deployed!)*

---

## 📖 Project Overview
Financial fraud costs the global economy over $5 trillion annually. Legacy rule-based systems miss novel attack patterns, and simple ML models are often "black boxes" that compliance teams cannot trust or audit.

This capstone project delivers an end-to-end, production-ready fraud detection pipeline. It combines state-of-the-art machine learning (LightGBM), handles severe class imbalance using SMOTE, and introduces **Explainable AI (SHAP)** via a live Streamlit operations dashboard to ensure that every flagged transaction is fully transparent and interpretable by human investigators.

Based on testing, this model achieves a **0.799 PR-AUC** and represents a potential **$175.3M in annual fraud prevention savings**.

---

## ✨ Key Features
- **Highly Accurate Detection:** Optuna-tuned LightGBM classifier trained on the massive 590K+ IEEE-CIS Fraud Detection dataset.
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) values to break down exactly *why* a transaction was flagged, providing plain-English reasoning for compliance teams.
- **Risk Segmentation:** Transactions are automatically binned into actionable tiers: `Critical Risk`, `Suspicious`, and `Clear`.
- **Live Operations Dashboard:** A multi-page Streamlit application simulating a real-world investigator's command center.

---

## 🏗️ Architecture & Deployment Strategy

To deploy a model trained on a 680MB dataset to the cloud without hitting memory bottlenecks, this project uses a highly optimized data pipeline:

1. **Jupyter Notebook (`analysis.ipynb`):** Handles heavy EDA, SMOTE oversampling, Optuna tuning, and model exporting.
2. **Data Prep Script (`dashboard/data_prep.py`):** Takes a stratified sample of the massive dataset (ensuring all rare fraud cases are preserved), applies the exact preprocessing steps expected by the model, and exports it as a highly compressed `.parquet` file.
3. **The Parquet Engine (`dashboard/sample_data.parquet`):** Shrinks the necessary dashboard data down to just ~5MB, allowing lightning-fast load times and bypassing GitHub/Streamlit memory limits.
4. **Streamlit App (`dashboard/app.py`):** Loads the pre-processed Parquet data and the `model.pkl` file to generate real-time SHAP plots and predictions in the browser.

---

## 💻 How to Run Locally

If you'd like to run the operations dashboard on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   cd dashboard
   streamlit run app.py
   ```
4. The dashboard will automatically open in your browser at `http://localhost:8501`.

---

## 🛠️ Technologies Used
- **Data Manipulation:** `pandas`, `numpy`, `pyarrow`
- **Machine Learning:** `scikit-learn`, `lightgbm`, `xgboost`, `imbalanced-learn` (SMOTE)
- **Hyperparameter Tuning:** `optuna`
- **Explainability:** `shap`
- **Frontend / Deployment:** `streamlit`, `plotly`

---
*Developed as part of the Week 4 AI & Data Analytics Capstone Project.*
