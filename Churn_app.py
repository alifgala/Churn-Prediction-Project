import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

# Load model XGBoost yang sudah di-train
model = XGBClassifier()
model.load_model('trained_model.json')  # Pastikan path model benar

# Load scaler jika tersedia (disimpan saat training)
scaler = joblib.load('scaler.pkl')  # Pastikan ada file scaler.pkl

# Fitur yang digunakan dalam model
selected_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
    'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service',
    'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service',
    'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

def predict_churn(input_data):
    original_data = input_data.copy()

    # Konversi TotalCharges ke numerik dengan aman
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0)

    # One-hot encoding untuk fitur kategorikal
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Pastikan input memiliki fitur yang sesuai dengan model
    for col in selected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Tambahkan fitur yang hilang dengan nilai 0

    # Urutkan kolom sesuai dengan model
    input_data = input_data[selected_features]

    # Normalisasi menggunakan scaler yang telah dilatih
    input_data[selected_features] = scaler.transform(input_data[selected_features])

    # Prediksi churn
    predictions = model.predict(input_data)
    original_data['Churn Prediction'] = predictions
    original_data['Churn Prediction'] = original_data['Churn Prediction'].map({0: 'No', 1: 'Yes'})

    return original_data

# Streamlit UI
st.title('Customer Churn Prediction')
st.write('Upload a CSV file to predict customer churn.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)

    st.write("Input Data:")
    st.dataframe(input_data)

    if st.button("Predict"):
        result = predict_churn(input_data)

        st.write("Predicted Churn:")
        st.dataframe(result)

        result_csv = result.to_csv(index=False)
        st.download_button(label="Download Result CSV", data=result_csv, file_name="churn_predictions.csv", mime="text/csv")
