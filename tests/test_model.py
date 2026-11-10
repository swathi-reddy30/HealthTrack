import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from app.model import RANDOM_SEED, RISK_LEVELS, TOP_FACTOR_COUNT, evaluate_model, load_model, save_model, train_model
from app.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, prepare_features
from scripts.generate_mock_data import generate_patients
from scripts.train_model import TARGET_ACCURACY, VALIDATION_SET_FRACTION
from tests.conftest import make_patient


@pytest.fixture
def sample_features(training_data) -> pd.DataFrame:
    return prepare_features(training_data.head(20))


def test_model_trains_and_supports_all_three_classes(trained_pipeline):
    assert sorted(trained_pipeline.classes_) == sorted(RISK_LEVELS)


def test_model_generates_predictions(risk_model, sample_features):
    results = risk_model.predict(sample_features)

    assert len(results) == len(sample_features)
    assert all(result["risk_level"] in RISK_LEVELS for result in results)


def test_probabilities_sum_to_one(risk_model, sample_features):
    for result in risk_model.predict(sample_features):
        assert set(result["probabilities"]) == set(RISK_LEVELS)
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predicted_level_has_highest_probability(risk_model, sample_features):
    for result in risk_model.predict(sample_features):
        probabilities = result["probabilities"]
        assert probabilities[result["risk_level"]] == max(probabilities.values())


def test_top_factors_are_ranked_original_features(risk_model, sample_features):
    for result in risk_model.predict(sample_features):
        factors = result["top_factors"]
        contributions = [factor["contribution"] for factor in factors]

        assert len(factors) == TOP_FACTOR_COUNT
        assert all(factor["feature"] in FEATURE_COLUMNS for factor in factors)
        assert contributions == sorted(contributions, reverse=True)
        for factor in factors:
            assert factor["direction"] == ("increases" if factor["contribution"] > 0 else "decreases")


def test_high_risk_profile_is_predicted_high(risk_model):
    high_risk_patient = make_patient(
        age=75, bmi=38, systolic_bp=170, diastolic_bp=100, glucose=180,
        cholesterol=290, smoker="current", exercise_days_week=0, sleep_hours=5,
    )
    low_risk_patient = make_patient(
        age=25, bmi=22, systolic_bp=110, diastolic_bp=70, glucose=85,
        cholesterol=160, smoker="never", exercise_days_week=6, sleep_hours=8, family_history="no",
    )

    results = risk_model.predict(prepare_features(pd.DataFrame([high_risk_patient, low_risk_patient])))

    assert results[0]["risk_level"] == "high"
    assert results[1]["risk_level"] == "low"


def test_model_can_be_saved_and_loaded(tmp_path, trained_pipeline, risk_model, sample_features):
    model_path = tmp_path / "model.joblib"

    save_model(trained_pipeline, model_path, version="9.9.9")
    loaded_model = load_model(model_path)

    assert loaded_model.version == "9.9.9"
    assert loaded_model.predict(sample_features) == risk_model.predict(sample_features)


@pytest.mark.parametrize("seed", [RANDOM_SEED, 7, 123])
def test_model_meets_accuracy_target_on_validation_set(seed):
    # The brief requires at least 75% accuracy on a validation set. Seed 42 is
    # the real training data; the other seeds check the result isn't a lucky split.
    data = generate_patients(seed=seed)
    features, labels = prepare_features(data), data[TARGET_COLUMN]
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        features, labels, test_size=VALIDATION_SET_FRACTION, stratify=labels, random_state=RANDOM_SEED
    )

    metrics = evaluate_model(train_model(train_features, train_labels), validation_features, validation_labels)

    assert metrics["accuracy"] >= TARGET_ACCURACY


def test_evaluate_model_returns_expected_metrics(trained_pipeline, training_data):
    features = prepare_features(training_data)

    metrics = evaluate_model(trained_pipeline, features, training_data[TARGET_COLUMN])

    assert 0 <= metrics["accuracy"] <= 1
    assert set(metrics["per_class"]) == set(RISK_LEVELS)
    matrix = metrics["confusion_matrix"]["matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert sum(sum(row) for row in matrix) == len(training_data)
