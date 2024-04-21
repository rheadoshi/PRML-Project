import streamlit as st
import joblib
import pandas as pd
import numpy as np 

st.title('Stroke Prediction')

# Load the custom decision tree model
loaded_model = joblib.load(r'Scripts\knn_model.pkl')

# Function to make predictions
def make_prediction(data):
    # Make predictions
    prediction = loaded_model.predict(data)
    if prediction[0] == 1:
        return "Stroke Predicted"
    else:
        return "No Stroke Predicted"

# Encode categorical features
def encode_categorical_features(age, gender, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    # Encoding mappings
    age_encoding = {
        '<25': 1,
        '50>age>25': 2,
        '>50': 3
    }
    
    gender_encoding = {
        'Male': 1,
        'Female': 0
    }
    
    ever_married_encoding = {
        'Yes': 1,
        'No': 0
    }
    
    smoking_status_encoding = {
        'formerly smoked': 1,
        'never smoked': 2,
        'smokes': 3,
        'Unknown': 0
    }
    
    residence_type_encoding = {
        'Urban': 1,
        'Rural': 0
    }
    
    work_type_encoding = {
        'Govt_job': 0,
        'Private': 2,
        'Self-employed': 3,
        'children': 4,
        'Never_worked': 1
    }
    
    # Apply encodings
    age_encoded = age_encoding[age]
    gender_encoded = gender_encoding[gender]
    ever_married_encoded = ever_married_encoding[ever_married]
    work_type_encoded = work_type_encoding[work_type]
    residence_type_encoded = residence_type_encoding[residence_type]
    smoking_status_encoded = smoking_status_encoding[smoking_status]
    
    return age_encoded, gender_encoded, hypertension, heart_disease, ever_married_encoded, work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded

# Add input fields for user input
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.selectbox('Age', ['<25', '50>age>25', '>50'])
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart Disease')
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Govt_job', 'Private', 'Self-employed', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.slider('Average Glucose Level', min_value=0, max_value=300, value=100)
bmi = st.slider('BMI', min_value=0, max_value=100, value=25)
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# When the user clicks the 'Predict' button
if st.button('Predict'):
    # Encode categorical features
    age_encoded, gender_encoded, hypertension_encoded, heart_disease_encoded, ever_married_encoded, work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded = encode_categorical_features(age, gender, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status)
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age_encoded],
        'hypertension': [1 if hypertension_encoded else 0],
        'heart_disease': [1 if heart_disease_encoded else 0],
        'ever_married': [ever_married_encoded],
        'work_type': [work_type_encoded],
        'Residence_type': [residence_type_encoded],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status_encoded]
    })
    
    # Make predictions using the input data DataFrame
    prediction = make_prediction(input_data)
    
    # Display the prediction result
    st.write('Prediction:', prediction)