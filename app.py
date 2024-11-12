import streamlit as st
import pandas as pd
import pickle

# Load the saved ensemble model
with open('c:/Users/hp/OneDrive/Desktop/mlop/ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)


st.title("Ensemble Model Predictor")

st.sidebar.header("Input Parameters")

# Function to capture user input
def user_input_features():
    # Add inputs for all 7 features used in training
    feature1 = st.sidebar.number_input("Feature 1", value=0.0)
    feature2 = st.sidebar.number_input("Feature 2", value=0.0)
    feature3 = st.sidebar.number_input("Feature 3", value=0.0)
    feature4 = st.sidebar.number_input("Feature 4", value=0.0)
    feature5 = st.sidebar.number_input("Feature 5", value=0.0)
    feature6 = st.sidebar.number_input("Feature 6", value=0.0)
    feature7 = st.sidebar.number_input("Feature 7", value=0.0)
    
    return pd.DataFrame({
        'Feature1': [feature1],
        'Feature2': [feature2],
        'Feature3': [feature3],
        'Feature4': [feature4],
        'Feature5': [feature5],
        'Feature6': [feature6],
        'Feature7': [feature7]
    })

# Capture input from the user
input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Make predictions using the loaded ensemble model
st.subheader("Prediction")
prediction = ensemble_model.predict(input_df)
st.write(f"Predicted Class: {prediction[0]}")

st.write("""
## Conclusion
This app uses an ensemble machine learning model to make predictions based on user inputs.
""")