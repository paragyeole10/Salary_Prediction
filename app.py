import streamlit as st
import pandas as pd
import pickle

# Load model safely
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()
label_encoders = load_encoders()

# Preprocessing
def preprocess_input(input_data):
    df = pd.DataFrame([input_data])

    for col, encoder in label_encoders.items():
        if df[col][0] in encoder.classes_:
            df[col] = encoder.transform(df[col])
        else:
            df[col] = -1  # unseen value safe handling

    return df

# UI
st.title("💰 Salary Prediction App")

age = st.slider('Age', 18, 65, 30)
gender = st.selectbox('Gender', label_encoders['Gender'].classes_)
education = st.selectbox('Education Level', label_encoders['Education Level'].classes_)
job = st.selectbox('Job Title', label_encoders['Job Title'].classes_)
experience = st.slider('Years of Experience', 0, 40, 5)

input_data = {
    'Age': age,
    'Gender': gender,
    'Education Level': education,
    'Job Title': job,
    'Years of Experience': experience
}

FEATURE_ORDER = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

if st.button("Predict Salary"):
    with st.spinner("Predicting..."):
        processed = preprocess_input(input_data)
        processed = processed[FEATURE_ORDER]

        prediction = model.predict(processed)

    st.success(f"💰 Predicted Salary: ${prediction[0]:,.2f}")
