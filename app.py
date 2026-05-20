import streamlit as st
import pickle
import pandas as pd

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #0083B8;
    text-align: center;
}

.stButton>button {
    background-color: #0083B8;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stNumberInput {
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================

model = pickle.load(open("xgb_top6.pkl", "rb"))

# =========================================
# TITLE
# =========================================

st.title("🩺 Diabetes Risk Prediction System")

st.markdown("""
This AI-powered application predicts the likelihood of diabetes using clinical and health indicators.

⚠️ **Disclaimer:** This tool is for educational purposes only and does not replace professional medical diagnosis.
""")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("About")

st.sidebar.info("""
This application uses an XGBoost machine learning model trained using:

- Glucose
- BMI
- Blood Pressure
- Age
- Insulin
- Diabetes Pedigree Function

The model predicts diabetes risk based on patient health indicators.
""")

# =========================================
# INPUT SECTION
# =========================================

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

with col2:
    bloodpressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict Diabetes Risk"):

    # Create dataframe
    input_data = pd.DataFrame([[
        glucose,
        dpf,
        bmi,
        bloodpressure,
        age,
        insulin
    ]], columns=[
        "Glucose",
        "DiabetesPedigreeFunction",
        "BMI",
        "BloodPressure",
        "Age",
        "Insulin"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    # Progress bar
    st.progress(float(probability))

    # Metric display
    st.metric(
        label="Diabetes Risk Probability",
        value=f"{probability:.2%}"
    )

    # Result message
    if prediction == 1:

        st.error(
            f"⚠️ High Diabetes Risk Detected ({probability:.2%})"
        )

        st.warning("""
        Recommended Actions:
        - Consult a healthcare professional
        - Monitor blood glucose levels
        - Improve diet and physical activity
        - Schedule clinical screening
        """)

    else:

        st.success(
            f"✅ Low Diabetes Risk ({probability:.2%})"
        )

        st.info("""
        Maintain healthy lifestyle habits:
        - Balanced nutrition
        - Regular exercise
        - Routine medical checkups
        """)

# =========================================
# EXPANDER
# =========================================

with st.expander("How does this model work?"):

    st.write("""
    The machine learning model analyzes six important clinical indicators associated with diabetes risk:

    1. Glucose
    2. BMI
    3. Blood Pressure
    4. Age
    5. Insulin
    6. Diabetes Pedigree Function

    The model then estimates the probability of diabetes risk based on patterns learned from training data.
    """)