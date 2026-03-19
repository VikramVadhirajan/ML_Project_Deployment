# Sentiment Analysis App (Streamlit) 🚀

A simple and interactive **Sentiment Analysis Web App** built using **Python and Streamlit**, where users can analyze sentiments from text or Excel datasets.

This project demonstrates a **complete workflow from experimentation to deployment** without using serialized models (`pickle`), making it easy to understand and extend.

---

## 📌 Project Overview

This project includes:

- 📊 Exploratory analysis in Jupyter Notebook  
- 🧠 Sentiment logic implemented directly in the app  
- 🌐 Streamlit-based UI for real-time predictions  
- 📁 Excel-based input for batch processing  

---

## 🏗️ Project Structure

Sentiment_Analysis/
│
├── Amazon_Comments.xlsx
├── Amazon_Comments_Sentiment.xlsx
├── app.py
├── experiments.ipynb
├── requirements.txt
└── README.md

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- Streamlit  
- Jupyter Notebook  
- NLP (basic preprocessing / rule-based or logic-based approach)

---

## 🔄 Workflow

### 1. Experimentation (`experiments.ipynb`)
- Data exploration  
- Text preprocessing  
- Understanding sentiment patterns  

### 2. Application (`app.py`)
- Loads dataset or user input  
- Applies sentiment logic  
- Displays results in UI  

### 3. Output
- Sentiment classification (Positive / Negative / Neutral)  
- Option to download processed data  

---

## 🚀 How to Run the App

### Step 1: Clone the Repository
git clone https://github.com/VikramVadhirajan/ML_Project_Deployment.git  
cd ML_Project_Deployment/Sentiment_Analysis

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run Streamlit App
streamlit run app.py

---

## 📊 Features

- ✅ Simple and clean UI  
- ✅ Excel file upload support  
- ✅ Real-time sentiment analysis  
- ✅ No model serialization (easy to understand logic)  
- ✅ Beginner-friendly project structure  

---

## 📁 Input & Output

### Input
- Excel file (`.xlsx`) with text data  
- Manual text input (if implemented)

### Output
- Sentiment classification  
- Updated dataset with sentiment column  

---

## ⚠️ Note

- This project does **not use saved ML models (`pickle`)**
- Sentiment logic is applied **directly within the app**
- Ideal for learning and demonstration purposes  

---

## 🔮 Future Improvements

- Add ML model with persistence (pickle/joblib)  
- Improve NLP preprocessing (stemming, lemmatization)  
- Add progress bar for large datasets  
- Deploy on Streamlit Cloud  
- Add visualization dashboard  

---

## 👨‍💻 Author

**Vikram Vadhirajan**  
Data Analyst | Python | Machine Learning | Power BI  

GitHub: https://github.com/VikramVadhirajan  

---

## ⭐ Support

If you found this useful, consider giving the repo a ⭐
