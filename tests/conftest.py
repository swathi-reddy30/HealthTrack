"""Shared test fixtures.

Tests train their own small model in memory instead of using the file in
models/, so they are fast and don't depend on the training script being run.
"""

import io
import os
import tempfile

# Send test logs to a temporary folder so running pytest never fills the
# real logs/healthtrack.log. This must happen before "app" is imported,
# because the settings are read once at import time.
os.environ["LOG_DIR"] = tempfile.mkdtemp(prefix="healthtrack-test-logs-")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.model import RiskModel, train_model  # noqa: E402
from app.preprocessing import TARGET_COLUMN, prepare_features  # noqa: E402
from scripts.generate_mock_data import generate_patients  # noqa: E402

TEST_MODEL_VERSION = "test-1.0"


def make_patient(**overrides) -> dict:
    """A valid patient record; pass keyword arguments to change any field."""
    patient = {
        "patient_id": "T001",
        "age": 52,
        "sex": "female",
        "bmi": 31.5,
        "systolic_bp": 142,
        "diastolic_bp": 90,
        "resting_hr": 78,
        "glucose": 118,
        "cholesterol": 245,
        "smoker": "current",
        "alcohol_units_week": 6,
        "exercise_days_week": 1,
        "sleep_hours": 5.5,
        "family_history": "yes",
    }
    patient.update(overrides)
    return patient


def to_csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


@pytest.fixture(scope="session")
def training_data() -> pd.DataFrame:
    return generate_patients(n_patients=400, seed=7)


@pytest.fixture(scope="session")
def trained_pipeline(training_data):
    return train_model(prepare_features(training_data), training_data[TARGET_COLUMN])


@pytest.fixture(scope="session")
def risk_model(trained_pipeline) -> RiskModel:
    return RiskModel(trained_pipeline, version=TEST_MODEL_VERSION)


@pytest.fixture
def client(risk_model):
    # Entering the TestClient runs the app's startup (logging, fresh cache).
    # We then swap in the in-memory test model.
    with TestClient(app) as test_client:
        app.state.model = risk_model
        yield test_client
