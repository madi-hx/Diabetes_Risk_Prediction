import streamlit as st
import pickle
import numpy as np

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f4f8fb;
}

h1{
    color:#1565C0;
    text-align:center;
}

h3{
    color:#37474F;
}

div.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#1976D2,#42A5F5);
    color:white;
    border-radius:12px;
    height:55px;
    font-size:20px;
    border:none;
    font-weight:bold;
}

div.stButton > button:hover{
    background:linear-gradient(90deg,#1565C0,#1E88E5);
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

with open("diabetes_model.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

with open("target_encoder.pkl","rb") as f:
    target_encoder = pickle.load(f)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("🩺 Diabetes AI")

st.sidebar.markdown("---")

st.sidebar.write("""
### About

This application predicts the **Diabetes Risk Category**
using Machine Learning.

✔ Fast Prediction

✔ Easy to Use

✔ AI Powered
""")

st.sidebar.markdown("---")

st.sidebar.success("Model : Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.info("""
Developer

**Madiha Khan**

B.Tech AI & ML

SRMCEM
""")

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("🩺 Diabetes Risk Prediction")

st.markdown(
"""
### AI Powered Healthcare Assistant

Fill in the patient's health information below to predict the diabetes risk category.
"""
)

st.markdown("---")

# ---------------------------------------------------
# Input Fields
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("👤 Age",1,120,30)

    bmi = st.number_input("⚖ BMI",value=25.0)

    fasting_glucose = st.number_input(
        "🍬 Fasting Glucose Level",
        value=100
    )

    insulin = st.number_input(
        "💉 Insulin Level",
        value=15.0
    )

    triglycerides = st.number_input(
        "🧪 Triglycerides Level",
        value=150
    )

    calories = st.number_input(
        "🍽 Daily Calorie Intake",
        value=2000
    )

    sleep = st.number_input(
        "😴 Sleep Hours",
        value=7.0
    )

    family = st.selectbox(
        "👨‍👩‍👧 Family History",
        ["No","Yes"]
    )

with col2:

    gender = st.selectbox(
        "⚧ Gender",
        ["Female","Male"]
    )

    blood_pressure = st.number_input(
        "🩸 Blood Pressure",
        value=120
    )

    hba1c = st.number_input(
        "🧬 HbA1c Level",
        value=5.5
    )

    cholesterol = st.number_input(
        "❤️ Cholesterol Level",
        value=180
    )

    activity = st.selectbox(
        "🏃 Physical Activity",
        ["Low","Medium","High"]
    )

    sugar = st.number_input(
        "🍫 Sugar Intake (grams/day)",
        value=50.0
    )

    stress = st.slider(
        "😰 Stress Level",
        1,
        10,
        5
    )

    waist = st.number_input(
        "📏 Waist Circumference (cm)",
        value=90.0
    )

# ---------------------------------------------------
# Convert Inputs
# ---------------------------------------------------

gender = 1 if gender == "Male" else 0

family = 1 if family == "Yes" else 0

activity_map = {
    "High":0,
    "Low":1,
    "Medium":2
}

activity = activity_map[activity]

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🔍 Predict Diabetes Risk"):

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

    result = target_encoder.inverse_transform(prediction)[0]

    st.markdown("---")

    st.subheader("Prediction Result")

    if result == "Low":

        st.success("🟢 LOW DIABETES RISK")

        st.info("""
### Recommendation

✔ Maintain a healthy lifestyle.

✔ Exercise regularly.

✔ Eat balanced meals.

✔ Get routine health checkups.
""")

    elif result == "Medium":

        st.warning("🟡 MEDIUM DIABETES RISK")

        st.info("""
### Recommendation

✔ Reduce sugar intake.

✔ Increase physical activity.

✔ Monitor blood glucose regularly.

✔ Maintain a healthy body weight.
""")

    else:

        st.error("🔴 HIGH DIABETES RISK")

        st.info("""
### Recommendation

✔ Consult a healthcare professional.

✔ Monitor blood glucose levels.

✔ Follow a diabetic-friendly diet.

✔ Exercise regularly.

✔ Take prescribed medication if advised.
""")

st.markdown("---")

st.caption("Developed by **Madiha Khan** | B.Tech AI & ML | SRMCEM")
