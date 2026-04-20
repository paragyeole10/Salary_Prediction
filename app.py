
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Function to preprocess input data
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    df_input = pd.DataFrame([input_data])

    # Apply label encoding using the loaded encoders
    for column, encoder in label_encoders.items():
        # Handle unseen labels by converting them to a known label or NaN, then fill
        # For simplicity, we'll try to transform and catch errors for unseen data
        # In a real-world scenario, more robust handling for unseen data is needed
        try:
            df_input[column] = encoder.transform(df_input[column])
        except ValueError:
            # If an unseen label is encountered, assign a default or handle as appropriate
            # For example, assign a common encoded value or -1 to indicate unseen
            # Here, we'll assign a common category if possible, ensuring scalar output
            if column == 'Gender':
                df_input[column] = encoder.transform(['Male'])[0]
            elif column == 'Education Level':
                df_input[column] = encoder.transform(["Bachelor's"])[0]
            elif column == 'Job Title':
                df_input[column] = encoder.transform(["Software Engineer"])[0]
    return df_input

# Streamlit UI
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields
age = st.slider('Age', 18, 65, 30)

# Debugging: Display loaded classes to check for numerical labels in options
st.write(f"Loaded Gender Classes: {label_encoders['Gender'].classes_}")
st.write(f"Loaded Education Level Classes: {label_encoders['Education Level'].classes_}")
st.write(f"Loaded Job Title Classes: {label_encoders['Job Title'].classes_}")

gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
education_level = st.selectbox('Education Level', label_encoders['Education Level'].classes_)
job_title = st.selectbox('Job Title', label_encoders['Job Title'].classes_)
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

# Create a dictionary for input data
input_data = {
    'Age': age,
    'Gender': gender,
    'Education Level': education_level,
    'Job Title': job_title,
    'Years of Experience': years_of_experience
}

if st.button('Predict Salary'):
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)
    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
