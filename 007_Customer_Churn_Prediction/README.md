# Customer Churn Prediction 📊

A Machine Learning project that predicts whether a customer is likely to **churn (leave a service)** based on historical customer data.

The project demonstrates a **complete end-to-end machine learning pipeline**, including data preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment using **Streamlit**.

---

# 📌 Project Overview

Customer churn prediction is a critical problem for many businesses because **retaining customers is cheaper than acquiring new ones**.

This project builds a predictive model that helps businesses identify **customers who are likely to leave**, enabling proactive retention strategies.

---

# 🧱 Project Workflow

The following steps were implemented in this project:

### 1️⃣ Data Understanding

* Loaded the dataset
* Identified the **target column**
* Identified **unwanted columns**

### 2️⃣ Data Cleaning

* Removed unnecessary columns
* Detected and removed duplicate records
* Identified and visualized missing values
* Cleaned string symbols from the dataset

### 3️⃣ Feature Engineering

* Separated **numerical** and **categorical** features
* Created mapping for irregular categorical entries
* Stored mappings in **pickle files**

### 4️⃣ Handling Missing Values

* Numerical columns → **KNN Imputer**
* Categorical columns → **Simple Imputer**

### 5️⃣ Outlier Treatment

* Detected outliers
* Applied appropriate treatment methods

### 6️⃣ Encoding

* Applied **Label Encoding** for categorical variables

### 7️⃣ Train-Test Split

* Split dataset into training and testing sets

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

* Naive Bayes (GaussianNB)
* Decision Tree
* Random Forest
* Logistic Regression
* Linear Discriminant Analysis (LDA)
* Support Vector Machine (SVC)

---

# ⚙️ Model Training Strategy

The project uses an optimized model selection approach:

* Created **parameter grids for each model**
* Identified models that **do not require feature scaling**
* Built a **Scikit-Learn Pipeline**
* Applied **GridSearchCV for hyperparameter tuning**
* Evaluated models based on **accuracy**

The **best performing model** was selected and saved.

---

# 💾 Model Persistence

The best model was stored using **Pickle** for deployment.

```python
pickle.dump(best_model, open("model.pkl", "wb"))
```

---

# 🌐 Streamlit Deployment

A **Streamlit web application** was built to allow users to:

* Enter customer details
* Run the trained model
* Predict whether the customer will **churn or stay**

---

# 📂 Repository Structure

```
007_Customer_Churn_Prediction/
│
├── app.py
├── experiments.ipynb
├── dataset/
│
├── encoders.pkl
├── imputers.pkl
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

---

# 🚀 How to Run the Project

### Clone the Repository

```
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git
cd ML_Project_Deployment/007_Customer_Churn_Prediction
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Streamlit App

```
streamlit run app.py
```

---

# 📊 Key Highlights

✔ End-to-End ML pipeline
✔ Data preprocessing and cleaning
✔ Multiple model comparison
✔ Hyperparameter tuning with GridSearchCV
✔ Best model selection
✔ Model deployment using Streamlit

---

# 💡 Business Value

This solution helps businesses:

* Identify customers likely to churn
* Improve customer retention strategies
* Reduce revenue loss
* Make data-driven decisions

---

# 👨‍💻 Author

**Vikram Vadhirajan**

Data Analyst | Machine Learning | Python | Power BI

GitHub
https://github.com/VikramVadhirajan

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐
