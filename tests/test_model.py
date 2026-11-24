from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


AIRLINE_FEATURES = [
    "Gender",
    "Customer Type",
    "Age",
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


@pytest.fixture
def model():
    model_path = Path("models/model.pkl")
    assert model_path.exists(), f"Champion model not found at {model_path}"
    return joblib.load(model_path)


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame([
        [
            "Female", "Loyal Customer", 35, "Business travel", "Business",
            1200, 0, 0.0, 4, 4, 4, 5, 4, 4, 5, 4, 5, 4, 4, 4, 5, 4,
        ],
        [
            "Male", "disloyal Customer", 24, "Personal Travel", "Eco",
            450, 25, 30.0, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 2, 3,
        ],
        [
            "Female", "Loyal Customer", 48, "Business travel", "Eco Plus",
            850, 5, 5.0, 3, 3, 4, 4, 3, 3, 4, 3, 4, 3, 3, 3, 4, 4,
        ],
    ], columns=AIRLINE_FEATURES)


def test_model_pipeline_structure(model):
    """Ensure the champion is a scikit-learn Pipeline with preprocessor and classifier stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "preprocessor" in step_names, f"Pipeline missing 'preprocessor' stage; found {step_names}"
    assert "classifier" in step_names, f"Pipeline missing 'classifier' stage; found {step_names}"


def test_single_row_inference(model):
    """Test inference on a single passenger record."""
    sample = _sample_rows().iloc[[0]]
    pred = model.predict(sample)
    assert len(pred) == 1
    assert pred[0] in (0, 1)


def test_batch_inference_and_probabilities(model):
    """Test batch inference and verify probability distributions sum to 1.0."""
    batch = _sample_rows()
    preds = model.predict(batch)
    assert len(preds) == 3

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(batch)
        assert probs.shape == (3, 2)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), atol=1e-5)


def test_prediction_determinism(model):
    """Ensure identical passenger inputs yield identical outputs."""
    sample = _sample_rows().iloc[[0]]
    pred_1 = model.predict(sample)[0]
    pred_2 = model.predict(sample)[0]
    assert pred_1 == pred_2