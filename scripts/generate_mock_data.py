"""Generate a reproducible synthetic patient dataset.

IMPORTANT: every value and every label here is made up. The labels come from
the simple weighted scoring rule in calculate_risk_score(), not from real
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
NOISE_STANDARD_DEVIATION = 0.5
MISSING_VALUE_RATE = 0.02

# The made-up risk score: for each numeric feature, (reference value, points
# per unit above the reference). Below the reference the points are negative,
# e.g. age 60 adds (60 - 50) * 0.05 = +0.5 and age 40 adds -0.5.
# Negative weights mean "more is healthier" (exercise, sleep).
NUMERIC_RISK_WEIGHTS = {
    "age": (50, 0.05),
    "bmi": (25, 0.12),
    "systolic_bp": (120, 0.04),
    "glucose": (95, 0.025),
    "cholesterol": (200, 0.01),
    "resting_hr": (72, 0.02),
    "alcohol_units_week": (7, 0.04),
    "exercise_days_week": (3.5, -0.15),
    "sleep_hours": (7, -0.3),
}
SMOKER_POINTS = {"never": 0.0, "former": 0.4, "current": 1.0}
FAMILY_HISTORY_POINTS = 0.5

# Score below LOW_RISK_MAX_SCORE -> "low"; at or above HIGH_RISK_MIN_SCORE -> "high".
# Chosen to give roughly 38% low, 38% medium and 24% high.
LOW_RISK_MAX_SCORE = 0.5
HIGH_RISK_MIN_SCORE = 2.5

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


def calculate_risk_score(patients: pd.DataFrame) -> pd.Series:
    """A transparent, made-up weighted score. Higher score = higher synthetic risk."""
    score = pd.Series(0.0, index=patients.index)
    for feature, (reference_value, points_per_unit) in NUMERIC_RISK_WEIGHTS.items():
        score += (patients[feature] - reference_value) * points_per_unit
    score += patients["smoker"].map(SMOKER_POINTS)
    score += (patients["family_history"] == "yes") * FAMILY_HISTORY_POINTS
    return score


def assign_risk_levels(score: pd.Series, rng: np.random.Generator) -> pd.Series:
    # Random noise means two similar patients can get different labels, so the
    # model can't simply memorise the rule (real data is never perfectly clean).
    noisy_score = score + rng.normal(0, NOISE_STANDARD_DEVIATION, len(score))
    levels = np.select(
        [noisy_score < LOW_RISK_MAX_SCORE, noisy_score >= HIGH_RISK_MIN_SCORE],
        ["low", "high"],
        default="medium",
    )
    return pd.Series(levels, index=score.index)


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
    patients["risk_level"] = assign_risk_levels(calculate_risk_score(patients), rng)
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
    # lineterminator="\n" makes the file byte-for-byte identical on Windows, macOS and Linux.
    patients.to_csv(TRAINING_DATA_PATH, index=False, lineterminator="\n")
    print(f"Saved {len(patients)} synthetic patients to {TRAINING_DATA_PATH}")
    print("Class distribution:")
    print(patients["risk_level"].value_counts().to_string())

    sample = build_sample_upload()
    sample.to_csv(SAMPLE_DATA_PATH, index=False, lineterminator="\n")
    print(f"Saved demo upload file with {len(sample)} rows to {SAMPLE_DATA_PATH}")


if __name__ == "__main__":
    main()
