"""Feature definitions and the scikit-learn preprocessing pipeline.

The preprocessor is part of the saved model pipeline, so exactly the same
imputation, scaling and encoding is applied at training time and at
prediction time. It is only ever fitted on the training split, which avoids
leaking information from the test data into the model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "patient_id"
TARGET_COLUMN = "risk_level"

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "resting_hr",
    "glucose",
    "cholesterol",
    "alcohol_units_week",
    "exercise_days_week",
    "sleep_hours",
]
CATEGORICAL_FEATURES = ["sex", "smoker", "family_history"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            # Median is less affected by extreme values than the mean.
            ("imputer", SimpleImputer(strategy="median")),
            # Scaling puts every feature on a similar range, which helps Logistic
            # Regression train well and makes its coefficients comparable.
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": an unseen category becomes all zeros
            # instead of crashing the prediction. Dense output keeps the
            # contribution maths in model.py simple.
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Select the model's feature columns in a fixed order with consistent types."""
    features = data[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        # astype(float) turns None into NaN, which SimpleImputer recognises.
        features[column] = features[column].astype(float)
    for column in CATEGORICAL_FEATURES:
        # SimpleImputer only treats NaN (not None) as missing in text columns.
        features[column] = features[column].astype(object).where(features[column].notna(), np.nan)
    return features
