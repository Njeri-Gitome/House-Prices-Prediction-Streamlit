import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("House Prediction App")

st.divider()

st.write(" This app uses machine learning for predicting house price based on features of the house. To use this app, you can enter inputs from this UI then click on the predict button")

st.divider()
floors = st.number_input("Number of floors", min_value = 0, value = 0)
bedrooms = st.number_input("Number of bedrooms", min_value = 0, value = 0)
bathrooms = st.number_input("Number of bathrooms", min_value = 0, value = 0)
condition = st.number_input("Condition", min_value = 0, value = 3)
grade = st.number_input("House Grade", min_value = 0, value = 3)
number_of_schools = st.number_input("Number of schools nearby", min_value = 0, value = 0)

st.divider()

X = [floors, bedrooms, bathrooms, condition, grade, number_of_schools]

prediction_button  = st.button("Predict")

if prediction_button:
    st.balloons() # balloons appear when the button is used
    X_array =np.array(X)
    prediction = model.predict(X_array)
    st.write(f"Price prediction is {prediction}")
    
else:
    st.write("Please click on the Predict Button after entering values")
