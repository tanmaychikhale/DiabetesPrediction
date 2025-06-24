import streamlit as st
import numpy as np
import joblib

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Diabetes Prediction System")

preg = st.number_input('Pregnancies', min_value=0, step=1)
glucose = st.number_input('Glucose Level', min_value=0)
bp = st.number_input('Blood Pressure', min_value=0)
skin = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin Level', min_value=0)
bmi = st.number_input('BMI', min_value=0.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0, step=1)

if st.button('Predict'):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error('🔴 High risk of Diabetes!')
    else:
        st.success('🟢 No Diabetes detected.')

