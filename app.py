import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration for a "Beautiful UI"
st.set_page_config(
    page_title="Heart Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    if os.path.exists("knn_model.pkl"):
        with open("knn_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

def main():
    st.title("❤️ Heart Risk Prediction")
    st.write("Enter the patient's details below to predict the risk of heart disease.")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        bp = st.number_input("Blood Pressure (BP)", min_value=80, max_value=250, value=120)
    
    with col2:
        chol = st.number_input("Cholesterol Level", min_value=100, max_value=500, value=200)

    # Load the model
    model = load_model()

    if st.button("Analyze Risk"):
        if model is not None:
            # Create a DataFrame for prediction to match feature names
            # Note: Using 'Cholestrol' to match the spelling in your training data
            input_df = pd.DataFrame([[bp, chol]], columns=['BP', 'Cholestrol'])
            
            prediction = model.predict(input_df)
            
            st.divider()
            
            if prediction[0] == 1:
                st.error("### Result: Heart Risk Detected")
                st.write("Please consult a medical professional for a detailed checkup.")
            else:
                st.success("### Result: No Heart Risk Detected")
                st.write("The indicators appear to be within the safe range.")
        else:
            st.error("Model file 'knn_model.pkl' not found. Please ensure the model is trained and saved in the same directory.")

    # Sidebar info
    st.sidebar.header("About")
    st.sidebar.info("This app uses a K-Nearest Neighbors (KNN) model trained on blood pressure and cholesterol data to estimate heart health risk.")

if __name__ == "__main__":
    main()
