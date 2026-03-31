# Company Bankruptcy Prediction 📉

A Machine Learning project that predicts whether a company is likely to **go bankrupt** based on financial indicators.

This project demonstrates an **end-to-end machine learning workflow**, including data preprocessing, dimensionality reduction using **Principal Component Analysis (PCA)**, model training using **Support Vector Classifier (SVC)**, and deployment through **Streamlit**.

---

# 📊 Dataset

The dataset used in this project is sourced from **Kaggle**:

**Company Bankruptcy Prediction Dataset**
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

The dataset contains financial ratios and indicators used to determine whether a company will go bankrupt.

### Dataset Characteristics

* Financial indicators extracted from company financial statements
* Binary classification problem
* Target variable indicates whether the company will **go bankrupt (1) or remain solvent (0)**
* High dimensional dataset with multiple correlated financial features

---

# 📌 Project Objective

The goal of this project is to build a machine learning model capable of **predicting bankruptcy risk** using financial indicators.

Such models can help:

* Financial institutions assess **credit risk**
* Investors identify **financially unstable companies**
* Businesses evaluate **financial sustainability**

---

# 🧱 Project Workflow

## 1️⃣ Data Loading

* Imported the dataset
* Examined structure, columns, and target variable

---

## 2️⃣ Data Cleaning

* Checked for missing values
* Removed duplicate records
* Verified data types

---

## 3️⃣ Feature Preparation

* Separated **features (X)** and **target variable (y)**
* Standardized numerical features where required

---

## 4️⃣ Dimensionality Reduction (PCA)

The dataset contains a large number of financial indicators that may be highly correlated.

To address this:

* Applied **Principal Component Analysis (PCA)**
* Reduced dimensionality
* Retained maximum variance while reducing feature complexity
* Improved model training efficiency

---

## 5️⃣ Machine Learning Model

The final model uses:

**Support Vector Classifier (SVC)**

SVC works well for high-dimensional datasets and classification problems with complex boundaries.

---

## ⚙️ Machine Learning Pipeline

To ensure reproducibility and prevent data leakage, the model was built using a **Scikit-Learn Pipeline**:

```
StandardScaler → PCA → SVC
```

Pipeline advantages:

* Ensures preprocessing is applied consistently
* Prevents data leakage
* Simplifies model deployment

---

## 🔎 Model Training

The following steps were used during training:

* Train-test split
* Pipeline creation
* Hyperparameter tuning
* Model evaluation

Performance was measured using classification metrics such as:

* Accuracy
* Precision
* Recall
* Confusion Matrix

---

# 🌐 Streamlit Deployment

A **Streamlit application** was built to allow users to:

* Input company financial indicators
* Run the trained model
* Predict bankruptcy risk in real time

---

# 📂 Repository Structure

```
005_Company_Bankruptcy_Prediction/
│
├── app.py
├── experiments.ipynb
│
├── Test Files for streamlitapp
├── outlierbounds.pkl
├── model.pkl
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Tech Stack

Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib / Seaborn
PCA


---

# 🚀 How to Run the Project

Clone the repository

```
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git
cd ML_Project_Deployment/005_Company_Bankruptcy_Prediction
```

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit app

```
streamlit run app.py
```

---

# 📈 Key Highlights

✔ High-dimensional financial dataset
✔ Dimensionality reduction using **PCA**
✔ Classification using **Support Vector Machine**
✔ Pipeline-based ML workflow
✔ Real-time prediction using **Streamlit**

---

# 👨‍💻 Author

**Vikram Vadhirajan**

Data Analyst | Machine Learning | Python | Power BI

GitHub
https://github.com/VikramVadhirajan

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐
