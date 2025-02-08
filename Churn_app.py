import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Load model, scaler, dan fitur yang dipakai saat training
model = xgb.XGBClassifier()
model.load_model("trained_model.json")

scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Fungsi untuk prediksi churn
def predict_churn(input_data):
    # Konversi TotalCharges ke numerik jika masih string
    input_data["TotalCharges"] = pd.to_numeric(input_data["TotalCharges"], errors="coerce")

    # One-hot encoding untuk fitur kategorikal
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Pastikan fitur input sesuai dengan yang digunakan saat training
    for col in selected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[selected_features]

    # Normalisasi data menggunakan scaler yang sama saat training
    input_data_scaled = scaler.transform(input_data)

    # Prediksi churn
    predictions = model.predict(input_data_scaled)
    input_data["Churn Prediction"] = predictions
    input_data["Churn Prediction"] = input_data["Churn Prediction"].map({0: "No", 1: "Yes"})

    return input_data

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("Upload a CSV file to predict customer churn.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load CSV
    input_data = pd.read_csv(uploaded_file)

    # Display input data
    st.write("Input Data:")
    st.dataframe(input_data)

    # Button untuk prediksi
    if st.button("Predict"):
        result = predict_churn(input_data)

        # Display hasil prediksi
        st.write("Predicted Churn:")
        st.dataframe(result)

        # Download hasil prediksi
        result_csv = result.to_csv(index=False)
        st.download_button(label="Download Result CSV", data=result_csv, file_name="churn_predictions.csv", mime="text/csv")
