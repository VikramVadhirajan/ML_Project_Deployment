import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import os



BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "best_model_SVC.pkl")
with open(model_path,'rb') as file:
    grid_model=pickle.load(file)

scalar_path = os.path.join(BASE_DIR, "scalar.pkl")
with open(scalar_path,'rb') as file:
    scaler=pickle.load(file)

Outlier_path = os.path.join(BASE_DIR, "outlier_bounds.pkl")
with open(Outlier_path,'rb') as file:
    Outlier=pickle.load(file)

encoders_path = os.path.join(BASE_DIR, "encoder.pkl")
with open(encoders_path,'rb') as file:
    label_encoder = pickle.load(file)


st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Target the 3rd column specifically */
        [data-testid="column"]:nth-child(3) [data-testid="stVerticalBlock"] {
            position: fixed;
            width: 30%; /* Adjust width to match column size */
            height: 100vh;
            overflow-y: auto;
            border-left: 1px solid #ddd;
            padding-left: 20px;
        }
        
        /* Alternative: Sticky approach (often smoother) */
        [data-testid="column"]:nth-child(3) > div {
            position: sticky;
            top: 2rem;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)


st.header("Customer Churn Prediction Pipeline Based model")

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.subheader("Customer Inputs- Numerical")
    Tenure=st.number_input("Tenure",)
    City_Tier=st.number_input("City_Tier",)
    CC_Contacted_LY=st.number_input("CC_Contacted_LY",)
    Service_Score=st.number_input("Service_Score",)
    Account_user_count=st.number_input("Account_user_count",)    
    CC_Agent_Score=st.number_input("CC_Agent_Score",)    
    rev_per_month=st.number_input("rev_per_month",)
    Complain_ly=st.number_input("Complain_ly",)
    rev_growth_yoy=st.number_input("rev_growth_yoy",)
    coupon_used_for_payment=st.number_input("coupon_used_for_payment",)
    Day_Since_CC_connect=st.number_input("Day_Since_CC_connect",)
    cashback=st.number_input("cashback",)
    

with col2:
    st.subheader("Customer Inputs- Categorical")
    Payment=st.selectbox("Payment",label_encoder["Payment"].classes_)
    Gender=st.selectbox("Gender",label_encoder["Gender"].classes_)
    account_segment=st.selectbox("account_segment",label_encoder["account_segment"].classes_)
    Marital_Status=st.selectbox("Marital_Status",label_encoder["Marital_Status"].classes_)
    Login_device=st.selectbox("Login_device",label_encoder["Login_device"].classes_)
    Payment_encoded=label_encoder['Payment'].transform([Payment])[0]
    Gender_encoded=label_encoder['Gender'].transform([Gender])[0]
    account_segment_encoded=label_encoder['account_segment'].transform([account_segment])[0]
    Marital_Status_encoded=label_encoder['Marital_Status'].transform([Marital_Status])[0]
    Login_device_encoded=label_encoder['Login_device'].transform([Login_device])[0]

    input_data= {

    "Tenure":	Tenure	,
    "City_Tier":	City_Tier	,
    "CC_Contacted_LY":	CC_Contacted_LY	,
    "Payment":	Payment_encoded	,
    "Gender":	Gender_encoded	,
    "Service_Score":	Service_Score	,
    "Account_user_count":	Account_user_count	,
    "account_segment":	account_segment_encoded	,
    "CC_Agent_Score":	CC_Agent_Score	,
    "Marital_Status":	Marital_Status_encoded	,
    "rev_per_month":	rev_per_month	,
    "Complain_ly":	Complain_ly	,
    "rev_growth_yoy":	rev_growth_yoy	,
    "coupon_used_for_payment":	coupon_used_for_payment	,
    "Day_Since_CC_connect":	Day_Since_CC_connect	,
    "cashback":	cashback	,
    "Login_device":	Login_device_encoded
    }



with col3:
    st.markdown('<div class="result-column">', unsafe_allow_html=True)
    st.subheader("Churn Prediction Result")
    # if st.button("Predict"): # If predict button to predict after all the variable are punched in
    grid_model.best_estimator_
    SVCmodel = grid_model.best_estimator_
    # Predict churn
    prediction = SVCmodel.predict(pd.DataFrame([input_data]))
    prob = SVCmodel.predict_proba(pd.DataFrame([input_data]))[0][1]   # probability of churn
    prob_percent = round(prob * 100, 2)
    
    if prediction[0] == 1:
        st.error("⚠️ Customer Likely to Churn")
    else:
        st.success("✅ Customer Likely to Stay")

    st.metric("Churn Probability", f"{prob_percent}%")
    st.progress(prob)
    st.markdown('</div>', unsafe_allow_html=True)

