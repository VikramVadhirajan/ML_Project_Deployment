import os
import streamlit as st
import pickle
import pandas as pd
import  warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)

limits = pickle.load(open(os.path.join(BASE_DIR, "outlier_limits.pkl"), "rb"))

scalar=pickle.load(open(os.path.join(BASE_DIR, "scalar.pkl"), "rb"))

model=pickle.load(open(os.path.join(BASE_DIR, "regr.pkl"), "rb"))

knn_imputer=pickle.load(open(os.path.join(BASE_DIR, "knn_imputer.pkl"), "rb"))

outlier_limits = pickle.load(open(os.path.join(BASE_DIR, "outlier_limits.pkl"), "rb"))

#Streamlit app
st.title("House Price Prediction")



Transaction_Date = st.number_input("Enter Transaction Date", min_value=2012.0, max_value=2014.0, value=2013.5, step=0.001)
House_Age = st.number_input("Enter House Age", min_value=0.0, max_value=100.0, value=13.3, step=0.1)
Distance_to_MRT_Station = st.number_input("Enter Distance to the Nearest MRT Station", min_value=0.0, max_value=10000.0, value=561.9845, step=0.1)
Number_of_Convenience_Stores = st.number_input("Enter Number of Convenience Stores", min_value=0, max_value=10, value=5, step=1)
Latitude = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=24.98746, step=0.00001)
Longitude = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=121.54391, step=0.00001)


input_data = {
    'X1 transaction date': Transaction_Date,
    'X2 house age': House_Age,
    'X3 distance to the nearest MRT station': Distance_to_MRT_Station,
    'X4 number of convenience stores': Number_of_Convenience_Stores,
    'X5 latitude': Latitude,
    'X6 longitude': Longitude
}

input_df = pd.DataFrame([input_data])
input_df_scaled = scalar.transform(input_df)
prediction = model.predict(input_df_scaled)
st.subheader("Predicted House Price of Unit Area:")

st.write("$",round(prediction[0], 2))




st.markdown("Check out the github repository (https://github.com/VikramVadhirajan/House_Price_Prediction)")




