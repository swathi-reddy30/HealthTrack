"""Generate a reproducible synthetic patient dataset.

IMPORTANT: every value and every label here is made up. The labels come from
the simple points-based rule in calculate_risk_points(), not from real
clinical outcomes. The dataset only exists to demonstrate the software.

Run from the project root:
    python -m scripts.generate_mock_data
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_patients.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_patients.csv"

NUMBER_OF_PATIENTS = 1000
RANDOM_SEED = 42
SAMPLE_RANDOM_SEED = 2024  # a different seed so the demo file isn't a copy of training rows
NOISE_STANDARD_DEVIATION = 1.0
MISSING_VALUE_RATE = 0.02

# Total points below LOW_RISK_MAX_POINTS -> "low"; at or above HIGH_RISK_MIN_POINTS -> "high".
LOW_RISK_MAX_POINTS = 3.5
HIGH_RISK_MIN_POINTS = 6.0

# Columns that get a few blank values, so the model's imputation is exercised.
COLUMNS_WITH_MISSING_VALUES = ["bmi", "glucose", "cholesterol", "sleep_hours", "smoker"]


def generate_patient_features(n_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    age = rng.integers(18, 86, size=n_patients)
    bmi = np.clip(rng.normal(27, 5, n_patients), 16, 50).round(1)
    # Some values depend on others (e.g. blood pressure rises with age) so the
    # data looks a little more realistic than independent random numbers.
    systolic_bp = np.clip(rng.normal(105 + 0.45 * age, 14), 90, 200).round()
    diastolic_bp = np.clip(systolic_bp * 0.55 + rng.normal(12, 6, n_patients), 55, 120).round()
    diastolic_bp = np.minimum(diastolic_bp, systolic_bp - 20)
    glucose = np.clip(rng.normal(80 + 0.7 * bmi, 18), 65, 250).round()
    cholesterol = np.clip(rng.normal(170 + 0.6 * age, 32), 120, 320).round()

    return pd.DataFrame(
        {
            "patient_id": [f"P{number:04d}" for number in range(1, n_patients + 1)],
            "age": age,
            "sex": rng.choice(["male", "female"], size=n_patients),
            "bmi": bmi,
            "systolic_bp": systolic_bp.astype(int),
            "diastolic_bp": diastolic_bp.astype(int),
            "resting_hr": np.clip(rng.normal(72, 10, n_patients), 45, 120).round().astype(int),
            "glucose": glucose.astype(int),
            "cholesterol": cholesterol.astype(int),
            "smoker": rng.choice(["never", "former", "current"], size=n_patients, p=[0.6, 0.2, 0.2]),
            "alcohol_units_week": np.clip(rng.gamma(2.0, 3.0, n_patients), 0, 40).round().astype(int),
            "exercise_days_week": rng.integers(0, 8, size=n_patients),
            "sleep_hours": np.clip(rng.normal(7, 1.1, n_patients), 4, 10).round(1),
            "family_history": rng.choice(["yes", "no"], size=n_patients, p=[0.3, 0.7]),
        }
    )


def calculate_risk_points(patients: pd.DataFrame) -> pd.Series:
    """A transparent, made-up scoring rule. More points = higher synthetic risk."""
    points = pd.Series(0.0, index=patients.index)
    points += np.select([patients["age"] >= 60, patients["age"] >= 45], [2, 1], default=0)
    points += np.select([patients["bmi"] >= 30, patients["bmi"] >= 25], [2, 1], default=0)
    points += np.select([patients["systolic_bp"] >= 140, patients["systolic_bp"] >= 130], [2, 1], default=0)
    points += np.select([patients["glucose"] >= 126, patients["glucose"] >= 100], [2, 1], default=0)
    points += (patients["cholesterol"] >= 240).astype(int)
    points += patients["smoker"].map({"current": 2, "former": 1, "never": 0})
    points += (patients["alcohol_units_week"] > 14).astype(int)
    points += np.select([patients["exercise_days_week"] < 2, patients["exercise_days_week"] >= 5], [1, -1], default=0)
    points += (patients["sleep_hours"] < 6).astype(int)
    points += (patients["family_history"] == "yes").astype(int)
    points += (patients["resting_hr"] > 90).astype(int)
    return points


def assign_risk_levels(points: pd.Series, rng: np.random.Generator) -> pd.Series:
    # Random noise means two similar patients can get different labels, so the
    # model can't simply memorise the rule (real data is never perfectly clean).
    noisy_points = points + rng.normal(0, NOISE_STANDARD_DEVIATION, len(points))
    levels = np.select(
        [noisy_points < LOW_RISK_MAX_POINTS, noisy_points >= HIGH_RISK_MIN_POINTS],
        ["low", "high"],
        default="medium",
    )
    return pd.Series(levels, index=points.index)


def add_missing_values(patients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patients = patients.astype(object)
    for column in COLUMNS_WITH_MISSING_VALUES:
        is_missing = rng.random(len(patients)) < MISSING_VALUE_RATE
        patients.loc[is_missing, column] = None
    return patients


def generate_patients(n_patients: int = NUMBER_OF_PATIENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate labelled synthetic patients. The same seed always gives the same data."""
    rng = np.random.default_rng(seed)
    patients = generate_patient_features(n_patients, rng)
    # Labels are calculated from the complete values *before* blanks are added,
    # like a real measurement that exists but wasn't recorded.
    patients["risk_level"] = assign_risk_levels(calculate_risk_points(patients), rng)
    return add_missing_values(patients, rng)


def build_sample_upload() -> pd.DataFrame:
    """A small unlabelled demo file, including some deliberately invalid rows."""
    sample = generate_patients(n_patients=12, seed=SAMPLE_RANDOM_SEED).drop(columns=["risk_level"])
    sample["patient_id"] = [f"S{number:03d}" for number in range(1, len(sample) + 1)]
    sample.loc[2, "glucose"] = None  # a blank cell: still valid, filled in by imputation

    template = sample.iloc[0].to_dict()
    invalid_rows = [
        {**template, "patient_id": "BAD001", "age": 400},  # impossible age
        {**template, "patient_id": "BAD002", "sex": "unknown"},  # invalid category
        {**template, "patient_id": "BAD003", "bmi": "abc"},  # not a number
        {**template, "patient_id": "BAD004", "systolic_bp": 80, "diastolic_bp": 95},  # diastolic above systolic
        {**template, "patient_id": None},  # missing patient ID
        {**template, "patient_id": "S001"},  # duplicate ID
    ]
    return pd.concat([sample, pd.DataFrame(invalid_rows)], ignore_index=True)


def main() -> None:
    patients = generate_patients()
    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"Saved {len(patients)} synthetic patients to {TRAINING_DATA_PATH}")
    print("Class distribution:")
    print(patients["risk_level"].value_counts().to_string())

    sample = build_sample_upload()
    sample.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Saved demo upload file with {len(sample)} rows to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
