import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load("model.pkl")

st.title("House Prediction App")

st.markdown("---")

st.write("This app uses machine learning for predicting house prices based on features of the house. To use this app, enter inputs from this UI and then click on the predict button.")

st.markdown("---")

# Collect inputs from the user
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)
floors = st.number_input("Number of floors", min_value=0, value=0)
condition = st.number_input("Condition", min_value=0, value=3)
grade = st.number_input("House Grade", min_value=0, value=3)
number_of_schools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.markdown("---")

# Collect inputs
X = [bedrooms, bathrooms, floors, condition, grade, number_of_schools]

# Define feature names (make sure these match the names used during model training)
feature_names = ["number of bedrooms","number of bathrooms",  "number of floors", "condition of the house", "grade of the house", "Number of schools nearby"]

# Predict button
prediction_button = st.button("Predict")

if prediction_button:
    #st.balloons()  # Balloons appear when the button is used
    X_array = np.array(X).reshape(1, -1)  # Reshape to 2D array as the model expects
    X_df = pd.DataFrame(X_array, columns=feature_names)  # Convert to DataFrame with feature names
    prediction = model.predict(X_df)
    st.write(f"Price prediction is ${prediction[0]:,.2f}")  
else:
    st.write("Please click on the Predict Button after entering values.")
