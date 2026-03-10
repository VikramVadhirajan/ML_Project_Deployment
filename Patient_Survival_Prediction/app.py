import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle
import os
# Load the trained model and encoders

BASE_DIR = os.path.dirname(__file__)

dep_encoders = pickle.load(open(os.path.join(BASE_DIR, "dep_encoders.pkl"), "rb"))
indep_encoder = pickle.load(open(os.path.join(BASE_DIR, "indep_encoder.pkl"), "rb"))
knn_imputer = pickle.load(open(os.path.join(BASE_DIR, "knn_imputer.pkl"), "rb"))
simple_imputer = pickle.load(open(os.path.join(BASE_DIR, "simple_imputer.pkl"), "rb"))
zscaler = pickle.load(open(os.path.join(BASE_DIR, "zscaler.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "LRModel.pkl"), "rb"))

st.title("Patient Survival Prediction")

# -----------------------
# Numeric Inputs
# -----------------------

n_days = st.number_input("N_Days", min_value=0)
age = st.number_input("Age", min_value=0)
bilirubin = st.number_input("Bilirubin")
cholesterol = st.number_input("Cholesterol")
albumin = st.number_input("Albumin")
copper = st.number_input("Copper")
alk_phos = st.number_input("Alk_Phos")
sgot = st.number_input("SGOT")
tryglicerides = st.number_input("Tryglicerides")
platelets = st.number_input("Platelets")
prothrombin = st.number_input("Prothrombin")
stage = st.selectbox("Stage", [1,2,3,4])

# -----------------------
# Categorical Inputs
# -----------------------

drug = st.selectbox(
    "Drug",
    ["D-penicillamine", "Placebo"]
)

sex = st.selectbox(
    "Sex",
    ["F", "M"]
)

ascites = st.selectbox(
    "Ascites",
    ["Y", "N"]
)

hepatomegaly = st.selectbox(
    "Hepatomegaly",
    ["Y", "N"]
)

spiders = st.selectbox(
    "Spiders",
    ["Y", "N"]
)

edema = st.selectbox(
    "Edema",
    ["Y", "N", "S"]
)


## Encode Categorical Variable. 

drug_encoded = dep_encoders['Drug'].transform([drug])[0]
sex_encoded = dep_encoders['Sex'].transform([sex])[0]
ascites_encoded = dep_encoders['Ascites'].transform([ascites])[0]
hepatomegaly_encoded = dep_encoders['Hepatomegaly'].transform([hepatomegaly])[0]
spiders_encoded = dep_encoders['Spiders'].transform([spiders])[0]
edema_encoded = dep_encoders['Edema'].transform([edema])[0]


input_data = {
        "N_Days": n_days,
        "Drug": drug_encoded,
        "Age": age,
        "Sex": sex_encoded,
        "Ascites": ascites_encoded,
        "Hepatomegaly": hepatomegaly_encoded,
        "Spiders": spiders_encoded,
        "Edema": edema_encoded,
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": tryglicerides,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Stage": stage
    }


# Scale the input data
input_data_scaled = zscaler.transform(pd.DataFrame([input_data]))

# Predict churn
prediction = model.predict(input_data_scaled)


indep_encoder=pickle.load(open("indep_encoder.pkl","rb"))
result = indep_encoder.inverse_transform(prediction)
prediction_proba =max(model.predict_proba(input_data_scaled)[0])

if result[0]=="D":
    st.write(f'The Patient is likely to die due to the disease with Probability of {prediction_proba:.2f}')
elif result[0]=="C":
    st.write(f'The person is Censored with Probability of {prediction_proba:.2f}')
elif result[0]=="CL":
    st.write(f'The person is censored due to liver transplantation with Probability of {prediction_proba:.2f}')

print("---------------------end of the report ---------------------")



