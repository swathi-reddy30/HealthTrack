import numpy as np
import pandas as pd
import pytest

from app.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, prepare_features
from tests.conftest import make_patient


@pytest.fixture
def training_features(training_data) -> pd.DataFrame:
    return prepare_features(training_data)


@pytest.fixture
def fitted_preprocessor(training_features):
    return build_preprocessor().fit(training_features)


def one_patient_frame(**overrides) -> pd.DataFrame:
    return prepare_features(pd.DataFrame([make_patient(**overrides)]))


def test_numeric_features_are_standardised(fitted_preprocessor, training_features):
    transformed = fitted_preprocessor.transform(training_features)
    numeric_part = transformed[:, : len(NUMERIC_FEATURES)]

    # After StandardScaler, each numeric column has mean ~0 and std ~1 on the training data.
    assert np.allclose(numeric_part.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_part.std(axis=0), 1, atol=1e-6)


def test_missing_numeric_value_is_imputed_with_median(fitted_preprocessor, training_features):
    transformed = fitted_preprocessor.transform(one_patient_frame(glucose=None))

    glucose_column = NUMERIC_FEATURES.index("glucose")
    scaler = fitted_preprocessor.named_transformers_["num"].named_steps["scaler"]
    median_glucose = training_features["glucose"].median()
    expected = (median_glucose - scaler.mean_[glucose_column]) / scaler.scale_[glucose_column]

    assert not np.isnan(transformed).any()
    assert transformed[0, glucose_column] == pytest.approx(expected)


def test_categorical_features_are_one_hot_encoded(fitted_preprocessor):
    transformed = fitted_preprocessor.transform(one_patient_frame(smoker="current"))
    encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    categorical_part = transformed[0, len(NUMERIC_FEATURES):]

    # Exactly one "1" per categorical feature (sex, smoker, family_history).
    assert categorical_part.sum() == len(CATEGORICAL_FEATURES)
    output_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    assert categorical_part[output_names.index("smoker_current")] == 1


def test_missing_category_is_imputed_with_most_frequent(fitted_preprocessor, training_features):
    transformed = fitted_preprocessor.transform(one_patient_frame(smoker=None))
    encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    output_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    most_common_status = training_features["smoker"].mode()[0]

    assert transformed[0, len(NUMERIC_FEATURES) + output_names.index(f"smoker_{most_common_status}")] == 1


def test_unknown_category_does_not_crash(fitted_preprocessor):
    # handle_unknown="ignore" encodes an unseen category as all zeros.
    transformed = fitted_preprocessor.transform(one_patient_frame(sex="other"))
    encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    output_names = list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    sex_columns = [len(NUMERIC_FEATURES) + i for i, name in enumerate(output_names) if name.startswith("sex_")]

    assert transformed[0, sex_columns].sum() == 0


def test_transformations_are_consistent(fitted_preprocessor):
    patient = one_patient_frame()
    first = fitted_preprocessor.transform(patient)
    second = fitted_preprocessor.transform(patient)

    assert np.array_equal(first, second)
    assert first.shape[1] == fitted_preprocessor.transform(one_patient_frame(sex="male")).shape[1]


def test_prepare_features_orders_columns_and_marks_missing_values():
    raw = pd.DataFrame([make_patient(smoker=None, bmi=None, extra_column="ignored")])

    features = prepare_features(raw)

    assert list(features.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert np.isnan(features.loc[0, "bmi"])
    assert features.loc[0, "smoker"] is np.nan
