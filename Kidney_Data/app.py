import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import pickle


#load the model

with open('../Kidney_Data/random_forest_model.pkl','rb') as file:
    rfmodel=pickle.load(file)

with open('../Kidney_Data/knn_imputer.pkl','rb') as file:
    knn_imputer=pickle.load(file)

with open('../Kidney_Data/simple_imputer.pkl','rb') as file:
    simple_imputer=pickle.load(file)

encoders = pickle.load(open("../Kidney_Data/encoders.pkl", "rb"))


with open('../Kidney_Data/scalar.pkl','rb') as file:
    scaler=pickle.load(file)

#Streamlit app
st.title("Kidney Disease Prediction")

#user input
# Numeric Features



# -------------------- NUMERIC FEATURES --------------------

age = st.slider("Age", 2, 100, 45, step=1)

blood_pressure = st.slider("Blood Pressure (mm Hg)", 50, 180, 80, step=1)

specific_gravity = st.slider("Specific Gravity", 1.005, 1.030, 1.020, step=0.005)

albumin = st.slider("Albumin (0-5)", 0, 5, 0, step=1)

sugar = st.slider("Sugar (0-5)", 0, 5, 0, step=1)

blood_glucose_random = st.slider("Blood Glucose Random (mg/dL)", 70, 500, 120, step=1)

blood_urea = st.slider("Blood Urea (mg/dL)", 10, 400, 40, step=1)

serum_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.1, 15.0, 1.2, step=0.1)

sodium = st.slider("Sodium (mEq/L)", 110, 160, 140, step=1)

potassium = st.slider("Potassium (mEq/L)", 2.0, 8.0, 4.5, step=0.1)

hemoglobin = st.slider("Hemoglobin (g/dL)", 3.0, 18.0, 12.0, step=0.1)

packed_cell_volume = st.slider("Packed Cell Volume (%)", 10, 60, 40, step=1)

white_blood_cell_count = st.slider("White Blood Cell Count (cells/cumm)", 2000, 20000, 8000, step=100)

red_blood_cell_count = st.slider("Red Blood Cell Count (millions/cumm)", 2.0, 8.0, 5.0, step=0.1)

# -------------------- CATEGORICAL FEATURES --------------------

red_blood_cells = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
pus_cell = st.selectbox("Pus Cell", ["normal", "abnormal"])
pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
bacteria = st.selectbox("Bacteria", ["present", "notpresent"])
hypertension = st.selectbox("Hypertension", ["yes", "no"])
diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["yes", "no"])
coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["yes", "no"])
appetite = st.selectbox("Appetite", ["good", "poor"])
pedal_edema = st.selectbox("Pedal Edema", ["yes", "no"])
anemia = st.selectbox("Anemia", ["yes", "no"])


# ---------- Encode categorical values first ----------

encoded_rbc = encoders['red_blood_cells'].transform([red_blood_cells])[0]
encoded_pc = encoders['pus_cell'].transform([pus_cell])[0]
encoded_pcc = encoders['pus_cell_clumps'].transform([pus_cell_clumps])[0]
encoded_bacteria = encoders['bacteria'].transform([bacteria])[0]
encoded_htn = encoders['hypertension'].transform([hypertension])[0]
encoded_dm = encoders['diabetes_mellitus'].transform([diabetes_mellitus])[0]
encoded_cad = encoders['coronary_artery_disease'].transform([coronary_artery_disease])[0]
encoded_appetite = encoders['appetite'].transform([appetite])[0]
encoded_pe = encoders['pedal_edema'].transform([pedal_edema])[0]
encoded_anemia = encoders['anemia'].transform([anemia])[0]


# ---------- Create dataframe AFTER encoding ----------

input_data = pd.DataFrame({
    'age':[age],
    'blood_pressure':[blood_pressure],
    'specific_gravity':[specific_gravity],
    'albumin':[albumin],
    'sugar':[sugar],
    'red_blood_cells':[encoded_rbc],
    'pus_cell':[encoded_pc],
    'pus_cell_clumps':[encoded_pcc],
    'bacteria':[encoded_bacteria],
    'blood_glucose_random':[blood_glucose_random],
    'blood_urea':[blood_urea],
    'serum_creatinine':[serum_creatinine],
    'sodium':[sodium],
    'potassium':[potassium],
    'hemoglobin':[hemoglobin],
    'packed_cell_volume':[packed_cell_volume],
    'white_blood_cell_count':[white_blood_cell_count],
    'red_blood_cell_count':[red_blood_cell_count],
    'hypertension':[encoded_htn],
    'diabetes_mellitus':[encoded_dm],
    'coronary_artery_disease':[encoded_cad],
    'appetite':[encoded_appetite],
    'pedal_edema':[encoded_pe],
    'anemia':[encoded_anemia]
})


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = rfmodel.predict(input_data_scaled)
prediction_proba = 1-rfmodel.predict_proba(input_data_scaled)[0][1]


st.write(f'disease Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('Person is likely to have kidney disease.')
else:
    st.write('The person is not likely to have kidney disease.')
