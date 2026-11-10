"""Model training, evaluation, persistence, prediction and explanations.

The model is a scikit-learn Pipeline: preprocessing (see preprocessing.py)
followed by a multiclass Logistic Regression.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from app.preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_preprocessor

logger = logging.getLogger(__name__)

# Bump this whenever the training data or model changes. It is stored inside
# the model file and is part of every cache key, so old cached predictions
# are never served for a new model.
MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42

RISK_LEVELS = ["low", "medium", "high"]
RISK_LABELS = {"low": "Low Risk", "medium": "Medium Risk", "high": "High Risk"}
TOP_FACTOR_COUNT = 3


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            # With three classes, scikit-learn's default solver fits a
            # multinomial model: one coefficient vector per risk level.
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def train_model(features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    pipeline = build_model_pipeline()
    pipeline.fit(features, labels)
    return pipeline


def evaluate_model(pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict:
    predicted_labels = pipeline.predict(features)
    report = classification_report(labels, predicted_labels, labels=RISK_LEVELS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(labels, predicted_labels)), 4),
        "per_class": {
            level: {
                "precision": round(report[level]["precision"], 4),
                "recall": round(report[level]["recall"], 4),
                "f1_score": round(report[level]["f1-score"], 4),
                "support": int(report[level]["support"]),
            }
            for level in RISK_LEVELS
        },
        "confusion_matrix": {
            "labels": RISK_LEVELS,
            "note": "Rows are the true risk level, columns are the predicted risk level.",
            "matrix": confusion_matrix(labels, predicted_labels, labels=RISK_LEVELS).tolist(),
        },
    }


def save_model(pipeline: Pipeline, path: Path, version: str = MODEL_VERSION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the version and feature list next to the pipeline so a loaded
    # model always knows which version it is.
    bundle = {
        "pipeline": pipeline,
        "model_version": version,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, path)
    logger.info("Saved model version %s to %s", version, path)


def load_model(path: Path) -> "RiskModel":
    # SECURITY: joblib uses pickle, which can execute code while loading.
    # That is why we only ever load the model from the server's own
    # configured path, never from a file supplied by a user.
    bundle = joblib.load(path)
    return RiskModel(pipeline=bundle["pipeline"], version=bundle["model_version"])


class RiskModel:
    """A trained pipeline plus its version, with explainable predictions."""

    def __init__(self, pipeline: Pipeline, version: str):
        self.pipeline = pipeline
        self.version = version
        self.classifier: LogisticRegression = pipeline.named_steps["classifier"]
        self.class_names = list(self.classifier.classes_)
        # For each column the preprocessor outputs, which original feature it came from.
        self.column_feature_names = _original_feature_for_each_column(pipeline)

    def predict(self, features: pd.DataFrame) -> list[dict]:
        """Return risk level, probabilities and top factors for each row."""
        all_probabilities = self.pipeline.predict_proba(features)
        transformed_rows = self.pipeline.named_steps["preprocessor"].transform(features)

        # Original (untransformed) values, so each factor can show the patient's own value.
        original_rows = features.astype(object).where(features.notna(), None).to_dict(orient="records")

        results = []
        for probabilities, transformed_row, original_row in zip(all_probabilities, transformed_rows, original_rows):
            predicted_index = int(np.argmax(probabilities))
            top_factors = self.explain(transformed_row, predicted_index)
            for factor in top_factors:
                factor["value"] = original_row[factor["feature"]]  # None means it was missing and imputed
            results.append(
                {
                    "risk_level": self.class_names[predicted_index],
                    "probabilities": {
                        level: round(float(probabilities[self.class_names.index(level)]), 4) for level in RISK_LEVELS
                    },
                    "top_factors": top_factors,
                }
            )
        return results

    def explain(self, transformed_row: np.ndarray, class_index: int) -> list[dict]:
        """Approximate how much each feature pushed this patient towards the predicted class.

        For Logistic Regression a class score is: intercept + sum(coefficient * value).
        So coefficient * value is each feature's share of that score. Because
        numeric values are standardised, 0 means "an average patient", and a
        contribution says how this patient differs from average.
        These are model contributions, NOT medical causes.
        """
        class_coefficients = self.classifier.coef_[class_index]
        column_contributions = transformed_row * class_coefficients

        # One-hot encoding splits e.g. "smoker" into several columns. Add them
        # back together so the coach sees one entry per original feature.
        contribution_by_feature: dict[str, float] = {}
        for feature_name, contribution in zip(self.column_feature_names, column_contributions):
            contribution_by_feature[feature_name] = contribution_by_feature.get(feature_name, 0.0) + float(contribution)

        # Largest positive contributions first: the factors that most support the prediction.
        ranked = sorted(contribution_by_feature.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                "feature": feature_name,
                "contribution": round(contribution, 4),
                "direction": "increases" if contribution > 0 else "decreases",
            }
            for feature_name, contribution in ranked[:TOP_FACTOR_COUNT]
        ]


def _original_feature_for_each_column(pipeline: Pipeline) -> list[str]:
    # The ColumnTransformer outputs numeric columns first (one each), then the
    # one-hot columns (one per category) in the order of CATEGORICAL_FEATURES.
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    names = list(NUMERIC_FEATURES)
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        names.extend([feature] * len(categories))
    return names
