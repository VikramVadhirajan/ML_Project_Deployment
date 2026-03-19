# Patient Survival Prediction 🏥📊

A Machine Learning-based **Patient Survival Prediction System** built using Python and deployed with **Streamlit**.  
This project predicts whether a patient is likely to survive based on clinical and medical input features.

It demonstrates a **complete end-to-end ML pipeline** including preprocessing, feature engineering, model training, and deployment.

---

## 📌 Project Overview

This project includes:

- 🧠 Machine Learning model for survival prediction  
- ⚙️ Data preprocessing pipelines (imputation, encoding, scaling)  
- 📊 Exploratory data analysis using notebooks  
- 🌐 Streamlit web application for real-time predictions  
- 💾 Saved preprocessing objects and trained model (`.pkl` files)  

---

## 🏗️ Project Structure

```
Patient_Survival_Prediction/
│
├── app.py                          # Streamlit application
├── cirrhosis.csv                   # Dataset
├── CheatSheet.xlsx                 # Feature reference
│
├── experiments.ipynb               # Initial experimentation
├── Survival_Prediction_Train.ipynb # Model training
├── Survival_Prediction_Test.ipynb  # Model testing
│
├── dep_encoders.pkl                # Dependent variable encoder
├── indep_encoder.pkl               # Independent encoder
├── simple_imputer.pkl              # Missing value imputer
├── knn_imputer.pkl                 # Advanced imputation
├── outlier_limits.pkl              # Outlier handling
├── zscaler.pkl                     # Feature scaling
├── LRModel.pkl                     # Trained ML model
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
- Handling missing values using Simple & KNN Imputer  
- Encoding categorical variables  
- Outlier treatment  
- Feature scaling (Z-score normalization)  

### 2. Model Training
- Logistic Regression model trained on processed dataset  
- Model evaluation using test dataset  

### 3. Model Saving
- All preprocessing steps and model saved using `.pkl` files  

### 4. Deployment
- Streamlit app loads model and preprocessing objects  
- Accepts user inputs  
- Returns survival prediction in real-time  

---

## 🚀 How to Run the Project

### Step 1: Clone Repository
```
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git
cd ML_Project_Deployment/Patient_Survival_Prediction
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
- ✅ Advanced preprocessing (imputation + scaling)  
- ✅ Model persistence using pickle  
- ✅ Real-time prediction UI  
- ✅ Modular and reusable components  

---

## 📁 Input & Output

### Input
- Patient clinical data (via UI)

### Output
- Survival Prediction (Yes / No or Probability)

---

## ⚠️ Important Notes

- Ensure all `.pkl` files are present before running the app  
- Preprocessing must match training pipeline to ensure accuracy  
- Model performance depends on dataset quality  

---

## 🔮 Future Improvements

- Add more advanced models (Random Forest, XGBoost)  
- Improve UI/UX  
- Add probability confidence score  
- Deploy on cloud (AWS / Streamlit Cloud)  
- Add feature importance visualization  

---

## 👨‍💻 Author

**Vikram Vadhirajan**  
Data Analyst | Machine Learning | Python | Power BI  

GitHub: https://github.com/VikramVadhirajan  

---

## ⭐ Support

If you found this useful, consider giving the repo a ⭐
