# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to load and preprocess the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Rename columns for readability
    d1 = {
        'price': 'Price',
        'resid_area': 'Residential_Area',
        'air_qual': 'Air_Quality',
        'room_num': 'Room_Number',
        'age': 'Age',
        'distance': 'Distance',
        'teachers': 'Teachers',
        'poor_prop': 'Proportion_Of_Population',
        'airport': 'Airport',
        'n_hos_beds': 'Number_Of_Hospital_Beds',
        'n_hot_rooms': 'Number_Of_Hotel_Rooms',
        'waterbody': 'Waterbody',
        'rainfall': 'Rainfall',
        'bus_ter': 'Bus_Terminal',
        'Sold': 'Sold'
    }
    df = df.rename(columns=d1)

    # Drop unnecessary column
    df = df.drop("Bus_Terminal", axis=1, errors="ignore")

    # Handle missing values consistently
    df['Waterbody'] = df['Waterbody'].fillna('River')  # Impute categorical column
    df = df.dropna()  # Drop rows with remaining missing values

    # Split the target and independent variables
    y = df['Sold']
    X = df.drop('Sold', axis=1)

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Standardize numerical columns
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, scaler

# Train the model
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Main function to run the Streamlit app
def main():
    st.title("🏠 House Sale Prediction App")
    st.write("Enter the house details to predict whether the house will be sold.")

    # Load and preprocess the data
    file_path = "House_Price1.csv"  # Ensure this file is in the same folder
    X, y, scaler = load_data(file_path)
    model = train_model(X, y)

    # Input form for user data
    st.sidebar.header("Enter House Details")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.sidebar.number_input(col, value=0.0)

    # Convert input to DataFrame
    user_df = pd.DataFrame([user_input])

    # Add a button to trigger the prediction
    if st.sidebar.button("Predict"):
        # Make predictions
        prediction = model.predict(user_df)
        prediction_proba = model.predict_proba(user_df)

        # Output
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("🏡 The house will **be SOLD**!")
        else:
            st.error("🚫 The house will **NOT be sold**.")
        st.write(f"Confidence: {prediction_proba[0][prediction[0]]:.2%}")

if __name__ == "__main__":
    main()
