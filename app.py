import os
import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Paths for the model, scaler, PCA, and encoders
model_path = 'loan_status_model.joblib'
scaler_path = 'scaler.joblib'
pca_path = 'pca.joblib'
onehot_encoder_path = 'onehot_encoder.joblib'
label_encoder_path = 'label_encoder.joblib'

# Load objects if they exist, else stop the app
if not all(map(os.path.isfile, [model_path, scaler_path, pca_path, onehot_encoder_path, label_encoder_path])):
    st.error("Error: Necessary model files are missing.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)
onehot_encoder = joblib.load(onehot_encoder_path)
label_encoder = joblib.load(label_encoder_path)

# Load image for title
image = Image.open('credit_img.jpg')

def preprocess_input(input_data):
    # Ensure the input_data columns match the expected training columns
    training_column_order = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                              'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                              'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
                              'person_home_ownership_OWN', 'person_home_ownership_RENT',
                              'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
                              'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                              'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                              'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D',
                              'loan_grade_E', 'loan_grade_F', 'loan_grade_G',
                              'cb_person_default_on_file_encoded']

    # Reorder the columns in the input_data DataFrame to match the training order
    input_data = input_data[training_column_order]

    # Convert 'Y' and 'N' to 1 and 0 for 'cb_person_default_on_file_encoded' column
    input_data['cb_person_default_on_file_encoded'] = input_data['cb_person_default_on_file_encoded'].map({'Y': 1, 'N': 0})

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(input_data)

    # Apply PCA transformation
    pca_features = pca.transform(scaled_features)

    return pca_features


def user_input_features():

    # Numerical features - Adjust the ranges and default values as needed
    person_age = st.sidebar.slider('Age', 18, 100, 30, key='person_age')
    person_income = st.sidebar.number_input('Annual Income', value=50000, min_value=10000, max_value=1000000, step=1000, key='person_income')
    person_emp_length = st.sidebar.slider('Employment Length (years)', 0, 40, 5, key='person_emp_length')
    loan_amnt = st.sidebar.number_input('Loan Amount', value=10000, min_value=1000, max_value=500000, step=1000, key='loan_amnt')
    loan_int_rate = st.sidebar.slider('Loan Interest Rate', 0.0, 30.0, 5.0, key='loan_int_rate')
    loan_percent_income = st.sidebar.slider('Loan Percent of Income', 0.0, 1.0, 0.1, key='loan_percent_income')
    cb_person_cred_hist_length = st.sidebar.slider('Credit History Length (years)', 0, 30, 5, key='cb_person_cred_hist_length')

    # Categorical features - Replace with actual categories from your dataset
    person_home_ownership = st.sidebar.selectbox('Home Ownership', ('Rent', 'Own', 'Mortgage', 'Other'))

    
    loan_intent = st.sidebar.selectbox('Loan Intent', ('Personal', 'Education', 'Medical', 'Venture', 'Home Improvement', 'Debt Consolidation'))
    #loan_intent = st.sidebar.selectbox('Loan Intent', ('Debt Consolidation', 'Credit Card', 'Home Improvement', 'Major Purchase', 'Other'))
    loan_grade = st.sidebar.selectbox('Loan Grade', ('A', 'B', 'C', 'D', 'E', 'F', 'G'))
    cb_person_default_on_file = st.sidebar.selectbox('Default on File', ('Yes', 'No'))

    # Remove spaces and convert to uppercase
    person_home_ownership = person_home_ownership.replace(' ', '').upper()
    loan_intent = loan_intent.replace(' ', '').upper()
    cb_person_default_on_file = {'Yes': 'Y', 'No': 'N'}.get(cb_person_default_on_file, cb_person_default_on_file)

    # Combine the features into a dataframe
    data = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_length': person_emp_length,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_home_ownership_MORTGAGE': 1 if person_home_ownership == 'MORTGAGE' else 0,
        'person_home_ownership_OTHER': 1 if person_home_ownership == 'OTHER' else 0,
        'person_home_ownership_OWN': 1 if person_home_ownership == 'OWN' else 0,
        'person_home_ownership_RENT': 1 if person_home_ownership == 'RENT' else 0,
        'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == 'DEBTCONSOLIDATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
        'loan_grade_A': 1 if loan_grade == 'A' else 0,
        'loan_grade_B': 1 if loan_grade == 'B' else 0,
        'loan_grade_C': 1 if loan_grade == 'C' else 0,
        'loan_grade_D': 1 if loan_grade == 'D' else 0,
        'loan_grade_E': 1 if loan_grade == 'E' else 0,
        'loan_grade_F': 1 if loan_grade == 'F' else 0,
        'loan_grade_G': 1 if loan_grade == 'G' else 0,
        'cb_person_default_on_file_encoded': cb_person_default_on_file
    }

    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.title("Credit Risk Prediction")
    st.image(image, width=400)  # Increase the width as needed

    # Layout for user input features
    with st.sidebar:
        st.header("User Input Features")
        input_df = user_input_features()

    # Layout for the Predict button and prediction display in the main area
    predict_button = st.button('Predict', key='predict_main')

    if predict_button:
        # Preprocess input data
        preprocessed_input = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(preprocessed_input)
        probability = model.predict_proba(preprocessed_input)[0][1] * 100  # Assuming index 1 is the probability of default
        
        # Convert prediction to interpretable output
        risk_level = "high" if prediction[0] == 1 else "low"  # Assuming 1 indicates high risk
        probability_text = f"{probability:.2f}%"

        # Display the prediction with styling
        st.subheader("Prediction Result")
        st.write(f"The probability that you'll have \"something\" is {probability_text}. You have a {risk_level} credit risk.")

    # Custom Styling with a border for the description
    st.markdown("""
        <style>
        .main {
            background-color: #F5F5F5;
        }
        .description {
            border: 1px solid #4F8BF9;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Introduction text with border
    st.markdown("""
    <div class="description">
        <p>Welcome to the Credit Risk Prediction Application. 
        This tool is designed to help you understand the likelihood of a loan defaulting based on various factors 
        such as age, income, employment history, and more. Simply adjust the parameters in the sidebar 
        to match your details and click 'Predict' to see the outcome.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
