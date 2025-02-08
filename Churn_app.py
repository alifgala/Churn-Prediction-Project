import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load model yang sudah dilatih
model = XGBClassifier()
model.load_model("trained_model.json")  # Ganti path jika perlu

# Fungsi preprocessing
def preprocess_data(input_data):
    input_data = input_data.copy()

    # Konversi TotalCharges ke numerik jika masih string
    input_data["TotalCharges"] = pd.to_numeric(input_data["TotalCharges"], errors="coerce")

    # One-hot encoding untuk fitur kategorikal
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Pastikan semua fitur yang digunakan dalam model tersedia
    expected_features = [
        "tenure", "MonthlyCharges", "TotalCharges", "gender_Male", "Partner_Yes", "Dependents_Yes",
        "PhoneService_Yes", "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_Fiber optic",
        "InternetService_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes", "OnlineBackup_No internet service",
        "OnlineBackup_Yes", "DeviceProtection_No internet service", "DeviceProtection_Yes", "TechSupport_No internet service",
        "TechSupport_Yes", "StreamingTV_No internet service", "StreamingTV_Yes", "StreamingMovies_No internet service",
        "StreamingMovies_Yes", "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes",
        "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
    ]

    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Tambahkan kolom yang hilang dengan nilai 0

    # Normalisasi fitur numerik
    scaler = StandardScaler()
    input_data[expected_features] = scaler.fit_transform(input_data[expected_features])

    return input_data[expected_features]

# Fungsi prediksi churn
def predict_churn(input_data):
    processed_data = preprocess_data(input_data)
    predictions = model.predict(processed_data)
    input_data["Churn Prediction"] = predictions
    input_data["Churn Prediction"] = input_data["Churn Prediction"].map({0: "No", 1: "Yes"})
    return input_data

# Streamlit UI
st.title("📊 Customer Churn Prediction")
st.write("Upload file CSV untuk memprediksi apakah pelanggan akan churn atau tidak.")

# File uploader
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("### 📄 Data yang diunggah:")
    st.dataframe(input_data)

    if st.button("🔍 Prediksi Churn"):
        result = predict_churn(input_data)
        st.write("### 📌 Hasil Prediksi Churn:")
        st.dataframe(result[["customerID", "Churn Prediction"]])  # Tampilkan customerID dan hasil prediksi
        
        # Unduh hasil prediksi
        result_csv = result.to_csv(index=False)
        st.download_button(label="📥 Download Hasil Prediksi", data=result_csv, file_name="churn_predictions.csv", mime="text/csv")
