# Kidney Disease Prediction 🩺📊

A Machine Learning-based **Kidney Disease Prediction System** built using Python and deployed with **Streamlit**.  
This project predicts whether a patient is likely to have kidney disease based on clinical parameters.

It demonstrates a **complete end-to-end ML pipeline** including preprocessing, feature engineering, model training, and deployment.

---

## 📌 Project Overview

This project includes:

- 🧠 Classification model for kidney disease prediction  
- ⚙️ Data preprocessing (imputation, encoding, scaling)  
- 📊 Exploratory Data Analysis using notebooks  
- 🌐 Streamlit app for real-time predictions  
- 💾 Saved preprocessing objects and trained model (`.pkl` files)  

---

## 🏗️ Project Structure

```
Kidney_Data/
│
├── app.py                      # Streamlit application
├── kidney_disease.csv         # Dataset
├── Cheatsheet.xlsx            # Feature reference
│
├── experiments.ipynb          # Data exploration
├── prediction.ipynb           # Model building/testing
│
├── encoders.pkl               # Encoding categorical variables
├── simple_imputer.pkl         # Basic missing value handling
├── knn_imputer.pkl            # Advanced imputation
├── scalar.pkl                 # Feature scaling
├── random_forest_model.pkl    # Trained model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- Jupyter Notebook  

---

## 🔄 Workflow

### 1. Data Preprocessing
- Handling missing values (Simple + KNN Imputer)  
- Encoding categorical variables  
- Feature scaling  

### 2. Model Training
- Random Forest classifier trained on dataset  
- Model evaluation using test data  

### 3. Model Saving
- Model and preprocessing steps saved using `.pkl` files  

### 4. Deployment
- Streamlit app loads model and preprocessing objects  
- Accepts user inputs  
- Predicts kidney disease in real-time  

---

## 🚀 How to Run the Project

### Step 1: Clone Repository
```
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git
cd ML_Project_Deployment/Kidney_Data
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```

### Step 3: Run Streamlit App
```
streamlit run app.py
```

---

## 📊 Features

- ✅ End-to-end ML pipeline  
- ✅ Random Forest model for better accuracy  
- ✅ Real-time prediction UI  
- ✅ Model persistence using pickle  
- ✅ Modular preprocessing pipeline  

---

## 📁 Input & Output

### Input
- Patient clinical data (via UI)

### Output
- Kidney Disease Prediction (Yes / No or Probability)

---

## ⚠️ Important Notes

- Ensure all `.pkl` files are present before running the app  
- Input format must match training data  
- Model performance depends on dataset quality  

---

## 🔮 Future Improvements

- Add more models (XGBoost, Neural Networks)  
- Improve UI/UX  
- Add probability confidence score  
- Deploy on cloud (AWS / Streamlit Cloud)  
- Add explainability (SHAP / feature importance)  

---

## 👨‍💻 Author

**Vikram Vadhirajan**  
Data Analyst | Machine Learning | Python | Power BI  

GitHub: https://github.com/VikramVadhirajan  

---

## ⭐ Support

If you found this useful, consider giving the repo a ⭐
