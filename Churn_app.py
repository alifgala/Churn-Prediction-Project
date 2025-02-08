import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

# Load model XGBoost yang sudah di-train dalam format JSON
model = XGBClassifier()
model.load_model('trained_model.json')  # Ganti dengan path model JSON yang benar

def predict_churn(input_data):
    # Salin DataFrame asli untuk mempertahankan semua kolom input
    original_data = input_data.copy()
    
    # Konversi TotalCharges ke numerik jika masih string
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce')
    
    # Mengubah variabel target menjadi biner
    input_data['Churn'] = input_data['Churn'].replace("No", 0).replace("Yes", 1)
    
    # One-hot encoding untuk fitur kategorikal
    input_data = pd.get_dummies(input_data, columns=[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ], drop_first=True)
    
    # Pastikan input memiliki fitur yang sesuai dengan model
    selected_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic',
        'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
        'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service',
        'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service',
        'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    for col in selected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[selected_features]
    
    # Normalisasi fitur numerik
    scaler = StandardScaler()
    input_data[selected_features] = scaler.fit_transform(input_data[selected_features])
    
    # Prediksi churn
    predictions = model.predict(input_data)
    input_data['Churn Prediction'] = predictions
    input_data['Churn Prediction'] = input_data['Churn Prediction'].map({0: 'No', 1: 'Yes'})
    
    # Menggabungkan hasil prediksi dengan data asli
    result_data = pd.concat([original_data, input_data[['Churn Prediction']]], axis=1)
    
    return result_data

# Streamlit UI
st.title('Customer Churn Prediction')
st.write('Upload a CSV file to predict customer churn.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    input_data = pd.read_csv(uploaded_file)
    
    # Display the input data
    st.write("Input Data:")
    st.dataframe(input_data)
    
    # Add a button for predictions
    if st.button("Predict"):
        # Process the input data and make predictions
        result = predict_churn(input_data)
        
        # Display the result in the app
        st.write("Predicted Churn:")
        st.dataframe(result)
        
        # Provide the result as a downloadable CSV
        result_csv = result.to_csv(index=False)
        st.download_button(label="Download Result CSV", data=result_csv, file_name="churn_predictions.csv", mime="text/csv")
