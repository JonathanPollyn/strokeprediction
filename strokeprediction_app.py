import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the model and preprocessing objects
model = joblib.load('stroke_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the app description
st.title('Stroke Prediction')
st.subheader('App by Jonathan Ibifubara Pollyn')
st.write("According to the World Health Organization (WHO), stroke is the second most prevalent cause of mortality worldwide, accounting for approximately 11% of all fatalities. The purpose of this application is to predict the likelihood of a patient experiencing a stroke based on input parameters such as gender, age, presence of various diseases, and smoking status. This application demonstrates the power of machine learning classification. If you have any question about the application, you can contact me via email at j.pollyn@gmail.com")
st.markdown(
    """
        ## Attribute Information

- gender: "Male", "Female" or "Other"
- age: age of the patient
- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- avg_glucose_level: average glucose level in blood
- bmi: body mass index
- smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
- stroke: 1 if the patient had a stroke or 0 if not

""")

# Define mapping for user friendly inputs
hypertension_mapping = {'No': 0, 'Yes':1}
heart_disease_mapping = {'No': 0, 'Yes': 1}

# Collect user input
st.sidebar.header('User Input Parameters')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
hypertension = st.sidebar.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.sidebar.selectbox('Heart Disease', ['No', 'Yes'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level (mg/dL)', min_value=0.0, max_value=1000.0, value=5.0)
bmi = st.sidebar.number_input('BMI (kg/m²)', min_value=0.0, max_value=1000.0, value=5.0)
smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
age_bracket = st.sidebar.selectbox('Age Bracket', ['0-18', '19-35', '36-50', '51-65', '66-80', '80+'])

# Map the user friendly to the user input
hypertension = hypertension_mapping[hypertension]
heart_disease = heart_disease_mapping[heart_disease]

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'gender': [gender],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status],
    'age_bracket': [age_bracket]
})

# Apply the label encoding
for column, le in label_encoders.items():
    input_data[column] = le.transform(input_data[column])

# Define columns to be scaled
columns_to_scale = ['gender', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status', 'age_bracket']

# Apply MinMaxScaler
input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

# Predict the new data
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.sidebar.write('The model predicts that the patient is at risk of having a stroke.')
    else:
        st.sidebar.write('The model predicts that the patient is not at risk of having stroke.')

