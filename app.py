from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

# Must match the exact column names/order in data/processed/X_train.csv
FEATURES = [
    "Gender",
    "Age",
    "Customer Type",
    "Type of Travel",
    "Class",
    "Flight Distance",
    "Departure Delay",
    "Arrival Delay",
    "Departure and Arrival Time Convenience",
    "Ease of Online Booking",
    "Check-in Service",
    "Online Boarding",
    "Gate Location",
    "On-board Service",
    "Seat Comfort",
    "Leg Room Service",
    "Cleanliness",
    "Food and Drink",
    "In-flight Service",
    "In-flight Wifi Service",
    "In-flight Entertainment",
    "Baggage Handling",
]

st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def pretty(label: int) -> str:
    return "Satisfied" if int(label) == 1 else "Neutral or Dissatisfied"


st.title("Airline Passenger Satisfaction Prediction")

model = load_model()
if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Run `dvc repro` (or `dvc pull`) to generate `models/model.pkl`."
    )
    st.stop()

st.write("Enter passenger, flight, and service rating details below:")

# 1. Passenger & Trip Details
st.subheader("Passenger & Flight Details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_type = st.selectbox("Customer Type", ["First-time", "Returning"])
    travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
    travel_class = st.selectbox("Class", ["Business", "Economy", "Economy Plus"])

with col2:
    age = st.number_input("Age", min_value=7, max_value=85, value=35, step=1)
    flight_distance = st.number_input(
        "Flight Distance (miles)", min_value=30, max_value=5000, value=1000, step=10
    )
    departure_delay = st.number_input(
        "Departure Delay (mins)", min_value=0, max_value=1600, value=0, step=1
    )
    arrival_delay = st.number_input(
        "Arrival Delay (mins)", min_value=0.0, max_value=1600.0, value=0.0, step=1.0
    )

# 2. Pre-Flight & Airport Experience (0 = Not Applicable, 1-5 Rating)
st.subheader("Pre-Flight & Airport Service (0–5)")
col3, col4 = st.columns(2)
with col3:
    time_convenience = st.slider("Departure & Arrival Time Convenience", 0, 5, 3)
    online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
    checkin_service = st.slider("Check-in Service", 0, 5, 3)

with col4:
    online_boarding = st.slider("Online Boarding", 0, 5, 4)
    gate_location = st.slider("Gate Location", 0, 5, 3)
    baggage_handling = st.slider("Baggage Handling", 0, 5, 4)

# 3. In-Flight Experience (0 = Not Applicable, 1-5 Rating)
st.subheader("In-Flight Experience (0–5)")
col5, col6 = st.columns(2)
with col5:
    onboard_service = st.slider("On-board Service", 0, 5, 4)
    seat_comfort = st.slider("Seat Comfort", 0, 5, 4)
    leg_room = st.slider("Leg Room Service", 0, 5, 4)
    cleanliness = st.slider("Cleanliness", 0, 5, 4)

with col6:
    food_drink = st.slider("Food and Drink", 0, 5, 3)
    inflight_service = st.slider("In-flight Service", 0, 5, 4)
    inflight_wifi = st.slider("In-flight Wifi Service", 0, 5, 3)
    inflight_entertainment = st.slider("In-flight Entertainment", 0, 5, 4)

if st.button("Predict Satisfaction", use_container_width=True):
    input_data = pd.DataFrame(
        [[
            gender,
            age,
            customer_type,
            travel_type,
            travel_class,
            flight_distance,
            departure_delay,
            arrival_delay,
            time_convenience,
            online_booking,
            checkin_service,
            online_boarding,
            gate_location,
            onboard_service,
            seat_comfort,
            leg_room,
            cleanliness,
            food_drink,
            inflight_service,
            inflight_wifi,
            inflight_entertainment,
            baggage_handling,
        ]],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_data)[0])
    result_text = pretty(prediction)

    if prediction == 1:
        st.success(f"Prediction: {result_text}")
    else:
        st.warning(f"Prediction: {result_text}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        proba_df = pd.DataFrame(
            {"Probability": proba},
            index=[pretty(c) for c in model.classes_],
        )
        st.bar_chart(proba_df)