import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title="Advanced Diabetes Dashboard", page_icon="🩺", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #ffe6e6, #ffcccc);
    font-family: 'Segoe UI', sans-serif;
}

/* Animated Heart */
.heart {
    font-size: 55px;
    text-align: center;
    animation: heartbeat 1.2s infinite;
    color: #b30000;
}

@keyframes heartbeat {
    0% { transform: scale(1); }
    30% { transform: scale(1.15); }
    50% { transform: scale(1); }
    70% { transform: scale(1.15); }
    100% { transform: scale(1); }
}

/* ECG animation */
.ecg {
    width: 100%;
    height: 60px;
    border-top: 2px solid #b30000;
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}

.ecg::before {
    content: "";
    position: absolute;
    width: 200%;
    height: 100%;
    background: repeating-linear-gradient(
        90deg,
        transparent,
        transparent 40px,
        #b30000 40px,
        #b30000 45px
    );
    animation: ecgmove 2s linear infinite;
}

@keyframes ecgmove {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

h1 {
    text-align: center;
    color: #800000;
}

.stButton > button {
    background-color: #cc0000;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# ------------------ HEADER ------------------
st.markdown("<div class='heart'>❤️</div>", unsafe_allow_html=True)
st.markdown("<h1>Advanced Diabetes Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='ecg'></div>", unsafe_allow_html=True)

st.write("### 🩺 Enter Patient Clinical Parameters")

# ------------------ INPUTS ------------------
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('🤰 Pregnancies', min_value=0)
    Glucose = st.number_input('🩸 Glucose Level', min_value=0)
    BloodPressure = st.number_input('💓 Blood Pressure', min_value=0)
    SkinThickness = st.number_input('📏 Skin Thickness', min_value=0)

with col2:
    Insulin = st.number_input('💉 Insulin Level', min_value=0)
    BMI = st.number_input('⚖️ BMI', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('🧬 Pedigree Function', min_value=0.0)
    Age = st.number_input('🎂 Age', min_value=0)

# ------------------ PREDICTION ------------------
if st.button("🔍 Analyze Medical Risk"):

    input_data = np.asarray([
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]).reshape(1, -1)

    std_data = scaler.transform(input_data)

    prediction = loaded_model.predict(std_data)[0]

    # 🔥 Create probability manually using decision function
    decision_score = loaded_model.decision_function(std_data)[0]

    # Convert to probability using sigmoid function
    probability = 1 / (1 + np.exp(-decision_score))
    probability_percent = probability * 100

    # ------------------ RISK GAUGE ------------------
    st.subheader("📈 Risk Probability Meter")
    st.progress(int(probability_percent))
    st.write(f"### 🔎 Estimated Diabetes Risk: {probability_percent:.2f}%")

    # ------------------ RESULT ------------------
    if prediction == 0:
        st.markdown("""
        <div class="result-box" style="background:#fff0f0; border:2px solid #cc0000;">
        ❤️ Patient is NOT Diabetic
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box" style="background:#ffe6e6; border:2px solid #990000;">
        💔 Patient is Diabetic
        </div>
        """, unsafe_allow_html=True)

    # ------------------ EXPLANATION ------------------
    st.subheader("🧠 AI Clinical Insight")

    important_factors = []
    if Glucose > 140:
        important_factors.append("High Glucose Level")
    if BMI > 30:
        important_factors.append("High BMI")
    if Age > 45:
        important_factors.append("Higher Age Risk")
    if Insulin > 150:
        important_factors.append("Elevated Insulin")

    if important_factors:
        st.write("🔬 Key contributing factors detected:")
        for factor in important_factors:
            st.write(f"• {factor}")
    else:
        st.write("No major high-risk indicators detected from provided values.")