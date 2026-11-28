from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from src.monitor_drift import analyze_drift, load_inference_logs, log_inference

BASE_DIR = Path(__file__).parent
MODEL_PATH = (
    BASE_DIR / "model.pkl"
    if (BASE_DIR / "model.pkl").exists()
    else BASE_DIR / "models" / "model.pkl"
)
SELECTION_PATH = (
    BASE_DIR / "streamlit_assets" / "model_selection.json"
    if (BASE_DIR / "streamlit_assets" / "model_selection.json").exists()
    else BASE_DIR / "reports" / "model_selection.json"
)
METRICS_PATH = (
    BASE_DIR / "streamlit_assets" / "metrics.json"
    if (BASE_DIR / "streamlit_assets" / "metrics.json").exists()
    else BASE_DIR / "reports" / "metrics.json"
)
COMP_CSV_PATH = (
    BASE_DIR / "streamlit_assets" / "model_comparison.csv"
    if (BASE_DIR / "streamlit_assets" / "model_comparison.csv").exists()
    else BASE_DIR / "reports" / "model_comparison.csv"
)
REFERENCE_DATA_PATH = (
    BASE_DIR / "streamlit_assets" / "train.csv"
    if (BASE_DIR / "streamlit_assets" / "train.csv").exists()
    else BASE_DIR / "data" / "processed" / "train.csv"
)
INFERENCE_LOG_PATH = BASE_DIR / "data" / "monitoring" / "inference_logs.csv"

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
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports():
    selection, metrics = None, None
    if SELECTION_PATH.exists():
        with open(SELECTION_PATH, "r", encoding="utf-8") as f:
            selection = json.load(f)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return selection, metrics


def load_inference_logs(log_path: Path) -> pd.DataFrame:
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame()


def pretty(label: int) -> str:
    return "Satisfied" if int(label) == 1 else "Neutral or Dissatisfied"


st.title("Airline Passenger Satisfaction Prediction")

model = load_model()
selection, metrics = load_reports()

if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Ensure `model.pkl` is present in the project root or run `dvc repro` to generate `models/model.pkl`."
    )
    st.stop()

tab_predict, tab_metrics, tab_monitoring = st.tabs([
    "Live Inference",
    "Model Lineage & Metrics",
    "Data Drift & Monitoring",
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE INFERENCE
# -----------------------------------------------------------------------------
with tab_predict:
    st.write("Enter passenger, flight, and service rating details below:")

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

    if st.button("Predict Satisfaction", type="primary", use_container_width=True):
        row_values = [
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
        ]
        input_data = pd.DataFrame([row_values], columns=FEATURES)

        prediction = int(model.predict(input_data)[0])
        result_text = pretty(prediction)

        proba_dict = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            classes = [pretty(c) for c in model.classes_]
            proba_dict = {classes[i]: float(proba[i]) for i in range(len(classes))}

        # Matches src/monitor_drift.py signature: log_inference(input_df, prediction, proba_dict)
        log_inference(input_data, prediction, proba_dict)

        if prediction == 1:
            st.success(f"Prediction: {result_text}")
        else:
            st.warning(f"Prediction: {result_text}")

        if proba_dict:
            proba_df = pd.DataFrame(
                list(proba_dict.items()),
                columns=["Satisfaction", "Probability"],
            ).set_index("Satisfaction")
            st.bar_chart(proba_df)

# -----------------------------------------------------------------------------
# TAB 2: MODEL LINEAGE & METRICS
# -----------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Champion Model Summary")
    if selection and "champion" in selection:
        champ = selection["champion"]
        crit = champ.get("criteria", {})
        st.info(f"**Active Champion:** {champ.get('display_name', champ.get('model_name', 'N/A'))}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Val Accuracy", f"{crit.get('validation_accuracy_pct', 0.0):.2f}%")
        test_acc = (metrics or {}).get("test_accuracy", (metrics or {}).get("accuracy", 0.0))
        c2.metric("Test Accuracy", f"{test_acc * 100:.2f}%")
        c3.metric("Latency", f"{crit.get('prediction_time_ms', 0.0):.2f} ms")
        c4.metric("Model Size", f"{crit.get('model_size_mb', 0.0):.3f} MB")

        if champ.get("explanation"):
            st.caption(champ["explanation"])
    elif metrics:
        st.json(metrics)

    st.subheader("Candidate Model Benchmark")
    if COMP_CSV_PATH.exists():
        st.dataframe(pd.read_csv(COMP_CSV_PATH), use_container_width=True)
    else:
        st.info("Run `dvc repro` to generate `reports/model_comparison.csv`.")

# -----------------------------------------------------------------------------
# TAB 3: DATA DRIFT & MONITORING
# -----------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Production Data Drift & Quality Monitoring")
    st.write(
        "Compares live production inference logs against the training reference distribution "
        "using the **Kolmogorov-Smirnov (KS) two-sample test**."
    )

    log_df = load_inference_logs(INFERENCE_LOG_PATH)
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.metric("Logged Production Inferences", len(log_df))
    with col_m2:
        run_check = st.button("Run Drift Check", type="primary", use_container_width=True)

    if not log_df.empty:
        if run_check or len(log_df) >= 5:
            if REFERENCE_DATA_PATH.exists():
                ref_df = pd.read_csv(REFERENCE_DATA_PATH)
                report = analyze_drift(ref_df, log_df)

                drift_detected = report.get("dataset_drift_detected", report.get("dataset_drift", False))
                drift_pct = report.get("drift_share_pct", report.get("drifted_percentage", 0.0))

                if drift_detected:
                    st.error(
                        f"DRIFT DETECTED — {report['drifted_features']} / {report['total_features']} "
                        f"features drifted ({drift_pct}%)"
                    )
                else:
                    st.success(
                        f"HEALTHY DISTRIBUTION — {report['drifted_features']} / {report['total_features']} "
                        f"features drifted ({drift_pct}%)"
                    )

                drift_rows = [
                    {
                        "Feature": feat,
                        "Status": "DRIFT" if d.get("drift_detected", d.get("drifted", False)) else "OK",
                        "p-value": round(d["p_value"], 4),
                        "KS Statistic": round(d["ks_statistic"], 4),
                        "Reference Mean": round(d.get("reference_mean", d.get("ref_mean", 0.0)), 2),
                        "Production Mean": round(d.get("current_mean", d.get("cur_mean", 0.0)), 2),
                        "Mean Shift": round(d["mean_shift"], 2),
                    }
                    for feat, d in report["features"].items()
                ]
                st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
            else:
                st.warning(f"Reference training file not found at `{REFERENCE_DATA_PATH}`.")
        else:
            st.info("Log at least 5 predictions (or click **Run Drift Check**) to compute KS drift statistics.")

        with st.expander("View Recent Production Logs (Last 10 Requests)"):
            st.dataframe(log_df.tail(10), use_container_width=True)
    else:
        st.info("No production inferences logged yet. Submit a prediction in the **Live Inference** tab first.")