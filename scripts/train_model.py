"""Train and evaluate the risk model, then save it and its metrics.

Run from the project root (after generate_mock_data):
    python -m scripts.train_model

Note: the metrics measure how well the model re-learns our own synthetic
labelling rule. They say nothing about real-world medical performance.
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import PROJECT_ROOT, settings
from app.model import MODEL_VERSION, RANDOM_SEED, RISK_LEVELS, evaluate_model, save_model, train_model
from app.preprocessing import TARGET_COLUMN, prepare_features

TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_patients.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
TEST_SET_FRACTION = 0.2


def count_classes(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    return {level: int(counts.get(level, 0)) for level in RISK_LEVELS}


def print_metrics(metrics: dict) -> None:
    print(f"\nModel version: {metrics['model_version']}")
    print(f"Train rows: {metrics['train_rows']}, test rows: {metrics['test_rows']}")
    print(f"Class distribution (all data): {metrics['class_distribution']['all']}")
    print(f"\nTest accuracy: {metrics['accuracy']:.3f}")
    print(f"Baseline accuracy (always predict the most common class): {metrics['baseline_accuracy']:.3f}")
    print("\nPer-class results on the test set:")
    for level, scores in metrics["per_class"].items():
        print(f"  {level:<7} precision={scores['precision']:.3f} recall={scores['recall']:.3f} support={scores['support']}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("          " + "  ".join(f"{level:>7}" for level in RISK_LEVELS))
    for level, row in zip(RISK_LEVELS, metrics["confusion_matrix"]["matrix"]):
        print(f"  {level:<7} " + "  ".join(f"{count:>7}" for count in row))


def main() -> None:
    data = pd.read_csv(TRAINING_DATA_PATH)
    features = prepare_features(data)
    labels = data[TARGET_COLUMN]

    # stratify=labels keeps the same low/medium/high mix in both splits, so the
    # test set is a fair sample. The fixed seed makes the split reproducible.
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SET_FRACTION, stratify=labels, random_state=RANDOM_SEED
    )

    # The pipeline (including the imputer and scaler) only ever sees training
    # data while fitting, which prevents test-set information leaking in.
    pipeline = train_model(train_features, train_labels)
    metrics = evaluate_model(pipeline, test_features, test_labels)

    # A trivial baseline gives the accuracy number some context.
    most_common_share = test_labels.value_counts(normalize=True).max()
    metrics.update(
        {
            "model_version": MODEL_VERSION,
            "train_rows": len(train_features),
            "test_rows": len(test_features),
            "baseline_accuracy": round(float(most_common_share), 4),
            "class_distribution": {
                "all": count_classes(labels),
                "train": count_classes(train_labels),
                "test": count_classes(test_labels),
            },
            "data_note": "Synthetic data with rule-based labels. Not evidence of real-world medical performance.",
        }
    )

    print_metrics(metrics)
    save_model(pipeline, settings.model_path, MODEL_VERSION)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved model to {settings.model_path}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
