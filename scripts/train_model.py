"""Train and evaluate the risk model, then save it and its metrics.

Run from the project root (after generate_mock_data):
    python -m scripts.train_model

Note: the metrics measure how well the model re-learns our own synthetic
labelling rule. They say nothing about real-world medical performance.
"""

import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from app.config import PROJECT_ROOT, settings
from app.ingestion import ingest_csv
from app.model import (
    MODEL_VERSION,
    RANDOM_SEED,
    RISK_LEVELS,
    build_model_pipeline,
    evaluate_model,
    save_model,
    train_model,
)
from app.preprocessing import TARGET_COLUMN, prepare_features

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_patients.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
VALIDATION_SET_FRACTION = 0.2
CROSS_VALIDATION_FOLDS = 5
TARGET_ACCURACY = 0.75  # minimum validation accuracy required by the project brief


def count_classes(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    return {level: int(counts.get(level, 0)) for level in RISK_LEVELS}


def print_metrics(metrics: dict) -> None:
    print(f"\nModel version: {metrics['model_version']}")
    print(f"Train rows: {metrics['train_rows']}, validation rows: {metrics['validation_rows']}")
    print(f"Class distribution (all data): {metrics['class_distribution']['all']}")

    cross_validation = metrics["cross_validation"]
    print(f"\nValidation accuracy: {metrics['accuracy']:.3f} (target >= {TARGET_ACCURACY})")
    print(
        f"{cross_validation['folds']}-fold cross-validation accuracy on the training set: "
        f"{cross_validation['mean_accuracy']:.3f} +/- {cross_validation['std_accuracy']:.3f}"
    )
    print(f"Baseline accuracy (always predict the most common class): {metrics['baseline_accuracy']:.3f}")

    print("\nPer-class results on the validation set:")
    for level, scores in metrics["per_class"].items():
        print(f"  {level:<7} precision={scores['precision']:.3f} recall={scores['recall']:.3f} support={scores['support']}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("          " + "  ".join(f"{level:>7}" for level in RISK_LEVELS))
    for level, row in zip(RISK_LEVELS, metrics["confusion_matrix"]["matrix"]):
        print(f"  {level:<7} " + "  ".join(f"{count:>7}" for count in row))


def main() -> None:
    # The training data goes through the same ingestion and validation as
    # uploaded files, so the model is trained on the same kind of cleaned data
    # it will see at prediction time.
    ingestion = ingest_csv(TRAINING_DATA_PATH.read_bytes(), with_labels=True)
    if ingestion.row_errors:
        print(f"Warning: {len(ingestion.row_errors)} invalid training rows were skipped.")
    data = ingestion.to_dataframe()
    features = prepare_features(data)
    labels = data[TARGET_COLUMN]

    # Hold back 20% as a validation set that the model never trains on.
    # stratify=labels keeps the same low/medium/high mix in both parts, and
    # the fixed seed makes the split reproducible.
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        features, labels, test_size=VALIDATION_SET_FRACTION, stratify=labels, random_state=RANDOM_SEED
    )

    # Cross-validation trains and scores 5 models on different slices of the
    # training set. It shows how stable the accuracy is, which a single
    # 200-row validation set can't show on its own.
    folds = StratifiedKFold(n_splits=CROSS_VALIDATION_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = cross_val_score(build_model_pipeline(), train_features, train_labels, cv=folds, scoring="accuracy")

    # The pipeline (including the imputer and scaler) only ever sees training
    # data while fitting, which prevents validation-set information leaking in.
    pipeline = train_model(train_features, train_labels)
    metrics = evaluate_model(pipeline, validation_features, validation_labels)

    # A trivial baseline gives the accuracy number some context.
    most_common_share = validation_labels.value_counts(normalize=True).max()
    metrics.update(
        {
            "model_version": MODEL_VERSION,
            "train_rows": len(train_features),
            "validation_rows": len(validation_features),
            "target_accuracy": TARGET_ACCURACY,
            "meets_target": metrics["accuracy"] >= TARGET_ACCURACY,
            "baseline_accuracy": round(float(most_common_share), 4),
            "cross_validation": {
                "folds": CROSS_VALIDATION_FOLDS,
                "mean_accuracy": round(float(fold_scores.mean()), 4),
                "std_accuracy": round(float(fold_scores.std()), 4),
            },
            "class_distribution": {
                "all": count_classes(labels),
                "train": count_classes(train_labels),
                "validation": count_classes(validation_labels),
            },
            "data_note": "Synthetic data with rule-based labels. Not evidence of real-world medical performance.",
        }
    )

    print_metrics(metrics)
    if not metrics["meets_target"]:
        print(f"\nWARNING: validation accuracy is below the {TARGET_ACCURACY} target.")

    save_model(pipeline, settings.model_path, MODEL_VERSION)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model to {settings.model_path}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
