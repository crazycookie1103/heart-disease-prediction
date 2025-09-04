
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler



   
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("lg_model.pkl", "rb") as f:
    lg_model = pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("stacking_model.pkl", "rb") as f:
   meta_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
     scaler: StandardScaler = pickle.load(f)


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.markdown("""
Enter your details below and find out your risk of heart disease.
""")


age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[0,1], format_func=lambda x: "female" if x==0 else "male")
cp = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=[0,1])
restecg = st.selectbox("Resting ECG results", options=[0,1,2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina?", options=[0,1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of ST Segment", options=[0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3) colored by fluoroscopy", options=[0,1,2,3,4])
thal = st.selectbox("Thalassemia", options=[0,1,2,3])


if st.button("Predict Heart Disease"):
    
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    
    input_scaled = scaler.transform(input_data)
    
    
    
   
    pred1 = knn_model.predict(input_scaled)
    pred2 = nb_model.predict(input_scaled)
    pred3 = lg_model.predict(input_scaled)
    pred4 = svm_model.predict(input_scaled)

    meta_input = np.column_stack([pred1, pred2, pred3, pred4])
    final_pred = meta_model.predict(meta_input)[0]
    
    
   
    
    # Display result
    if final_pred == 0:#0 heart disease 1 no disease
        st.error("⚠️ High risk of Heart Disease! Consult a doctor.")
    else:
        st.success("✅ Low risk of Heart Disease.")

# ---- Footer ----
st.markdown("---")

