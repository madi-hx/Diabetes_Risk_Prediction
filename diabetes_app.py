
import streamlit as st
import pickle
import numpy as np

# Load saved files
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

st.set_page_config(page_title="Diabetes Risk Prediction")

st.title("🩺 Diabetes Risk Prediction System")

st.write("Enter the patient's health details below.")

# User Inputs

age = st.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.selectbox("Gender", ["Female", "Male"])

bmi = st.number_input("BMI", value=25.0)

blood_pressure = st.number_input("Blood Pressure", value=120)

fasting_glucose = st.number_input("Fasting Glucose Level", value=100)

insulin = st.number_input("Insulin Level", value=15.0)

hba1c = st.number_input("HbA1c Level", value=5.5)

cholesterol = st.number_input("Cholesterol Level", value=180)

triglycerides = st.number_input("Triglycerides Level", value=150)

activity = st.selectbox(
    "Physical Activity Level",
    ["Low", "Medium", "High"]
)

calories = st.number_input("Daily Calorie Intake", value=2000)

sugar = st.number_input("Sugar Intake (grams/day)", value=50.0)

sleep = st.number_input("Sleep Hours", value=7.0)

stress = st.slider("Stress Level", 1, 10, 5)

family = st.selectbox(
    "Family History of Diabetes",
    ["No", "Yes"]
)

waist = st.number_input("Waist Circumference (cm)", value=90.0)

# Convert categorical values

gender = 1 if gender == "Male" else 0

family = 1 if family == "Yes" else 0

# IMPORTANT:
# Change this mapping if your LabelEncoder mapping is different.
activity_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

activity = activity_map[activity]

# Prediction

if st.button("Predict Diabetes Risk"):

    features = np.array([[
        age,
        gender,
        bmi,
        blood_pressure,
        fasting_glucose,
        insulin,
        hba1c,
        cholesterol,
        triglycerides,
        activity,
        calories,
        sugar,
        sleep,
        stress,
        family,
        waist
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    result = target_encoder.inverse_transform(prediction)

    st.success(f"Predicted Diabetes Risk Category: {result[0]}")
