import streamlit as st
import pandas as pd
import pickle


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="centered"
)


# -----------------------------
# Load model
# -----------------------------
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)


# -----------------------------
# Load label encoders
# -----------------------------
with open("label_encoders.pkl", "rb") as file:
    encoders = pickle.load(file)


# -----------------------------
# Title
# -----------------------------
st.title("🍔 Food Delivery Time Predictor")
st.write("Enter the delivery details to predict delivery time.")

st.divider()


# -----------------------------
# Input fields
# -----------------------------
distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=5.0,
    step=0.1
)

weather = st.selectbox(
    "Weather",
    encoders["Weather"].classes_.tolist()
)

traffic = st.selectbox(
    "Traffic Level",
    encoders["Traffic_Level"].classes_.tolist()
)

time_of_day = st.selectbox(
    "Time of Day",
    encoders["Time_of_Day"].classes_.tolist()
)

vehicle_type = st.selectbox(
    "Vehicle Type",
    encoders["Vehicle_Type"].classes_.tolist()
)

preparation_time = st.number_input(
    "Preparation Time (min)",
    min_value=0,
    value=20,
    step=1
)

courier_experience = st.number_input(
    "Courier Experience (years)",
    min_value=0.0,
    value=2.0,
    step=0.5
)


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Delivery Time"):

    # Encode categorical values
    weather_encoded = encoders["Weather"].transform([weather])[0]

    traffic_encoded = encoders["Traffic_Level"].transform([traffic])[0]

    time_encoded = encoders["Time_of_Day"].transform([time_of_day])[0]

    vehicle_encoded = encoders["Vehicle_Type"].transform([vehicle_type])[0]


    # -----------------------------------------
    # IMPORTANT:
    # Your CURRENT saved model was trained using
    # only these 5 columns.
    # -----------------------------------------
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Weather": [weather_encoded],
        "Traffic_Level": [traffic_encoded],
        "Preparation_Time_min": [preparation_time],
        "Courier_Experience_yrs": [courier_experience]
    })


    # Make sure order is exactly same as training
    input_data = input_data[
        [
            "Distance_km",
            "Weather",
            "Traffic_Level",
            "Preparation_Time_min",
            "Courier_Experience_yrs"
        ]
    ]


    # Prediction
    prediction = model.predict(input_data)[0]


    # -----------------------------
    # Display prediction
    # -----------------------------
    st.success(
        f"Estimated Delivery Time: {prediction:.2f} minutes"
    )

    st.metric(
        "Predicted Delivery Time",
        f"{prediction:.2f} min"
    )