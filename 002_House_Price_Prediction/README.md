# House Price Prediction 🏡📈

A Machine Learning-based **House Price Prediction System** built using Python and deployed with **Streamlit**.  
This project predicts real estate property prices based on multiple input features.

It demonstrates a **complete end-to-end ML pipeline** including preprocessing, model training, and deployment.

---

## 📌 Project Overview

This project includes:

- 🧠 Regression model for price prediction  
- ⚙️ Data preprocessing (imputation, outlier handling, scaling)  
- 📊 Exploratory Data Analysis using notebooks  
- 🌐 Streamlit app for real-time predictions  
- 💾 Saved preprocessing objects and trained model (`.pkl` files)  

---

## 🏗️ Project Structure

```
House_Price_Prediction/
│
├── app.py                              # Streamlit application
├── experiments.ipynb                   # Initial analysis
├── Price_Prediction_Train.ipynb        # Model training
├── Price_Prediction_Test.ipynb         # Model testing
│
├── Real estate valuation data set.xlsx # Dataset
│
├── knn_imputer.pkl                     # Missing value handling
├── outlier_limits.pkl                  # Outlier treatment
├── scalar.pkl                          # Feature scaling
├── regr.pkl                            # Trained regression model
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
- Handling missing values using KNN Imputer  
- Outlier detection and treatment  
- Feature scaling  

### 2. Model Training
- Regression model trained on housing dataset  
- Model evaluation using test data  

### 3. Model Saving
- Model and preprocessing steps saved using `.pkl` files  

### 4. Deployment
- Streamlit app loads model and preprocessing objects  
- Accepts user inputs  
- Predicts house price instantly  

---

## 🚀 How to Run the Project

### Step 1: Clone Repository
```
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git
cd ML_Project_Deployment/House_Price_Prediction
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
- ✅ Real-time house price prediction  
- ✅ Model persistence using pickle  
- ✅ Clean and modular structure  
- ✅ Beginner to intermediate friendly  

---

## 📁 Input & Output

### Input
- Property features (via UI)

### Output
- Predicted house price  

---

## ⚠️ Important Notes

- Ensure all `.pkl` files are present before running the app  
- Input features must match training data format  
- Model performance depends on dataset quality  

---

## 🔮 Future Improvements

- Add advanced models (XGBoost, Random Forest)  
- Add feature importance visualization  
- Improve UI/UX  
- Deploy on cloud (AWS / Streamlit Cloud)  
- Add location-based prediction enhancements  

---

## 👨‍💻 Author

**Vikram Vadhirajan**  
Data Analyst | Machine Learning | Python | Power BI  

GitHub: https://github.com/VikramVadhirajan  

---

## ⭐ Support

If you found this useful, consider giving the repo a ⭐
