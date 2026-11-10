"""Rule-based, personalised recommendations.

Recommendations are built from three things:
1. The predicted risk level -> general guidance (RISK_LEVEL_GUIDANCE).
2. The model's top contributing factors -> listed first, because they are
   most relevant to this particular prediction.
3. The patient's own values -> a factor rule only fires if the patient's
   value is actually a concern (e.g. bmi >= 30).

To add a new recommendation, add one FactorRule to FACTOR_RULES.
"""

from dataclasses import dataclass
from typing import Callable

DISCLAIMER = (
    "This application is an educational demonstration and does not provide medical advice, "
    "diagnosis, or treatment. Consult a qualified healthcare professional for medical decisions."
)

RISK_LEVEL_GUIDANCE = {
    "low": [
        "Keep up the current healthy habits and continue routine check-ups.",
    ],
    "medium": [
        "Consider setting one or two small, achievable lifestyle goals and reviewing progress regularly.",
        "Consider a routine check-up with a qualified healthcare professional to review these results.",
    ],
    "high": [
        "Encourage the patient to arrange a review with a qualified healthcare professional soon.",
        "Agree on a structured plan with regular follow-ups to track progress.",
    ],
}


@dataclass(frozen=True)
class FactorRule:
    feature: str
    applies_to: Callable[[dict], bool]  # True if this patient's value is a concern
    message: str


def _value_at_least(feature: str, threshold: float) -> Callable[[dict], bool]:
    return lambda patient: patient.get(feature) is not None and patient[feature] >= threshold


def _value_below(feature: str, threshold: float) -> Callable[[dict], bool]:
    return lambda patient: patient.get(feature) is not None and patient[feature] < threshold


def _value_equals(feature: str, expected: str) -> Callable[[dict], bool]:
    return lambda patient: patient.get(feature) == expected


def _sleep_out_of_range(patient: dict) -> bool:
    hours = patient.get("sleep_hours")
    return hours is not None and (hours < 7 or hours > 9)


BLOOD_PRESSURE_MESSAGE = (
    "Blood pressure is elevated: recommend discussing blood pressure monitoring with a qualified healthcare professional."
)

FACTOR_RULES = [
    FactorRule("bmi", _value_at_least("bmi", 30), "Encourage appropriate physical activity and healthy, balanced nutrition."),
    FactorRule("systolic_bp", _value_at_least("systolic_bp", 130), BLOOD_PRESSURE_MESSAGE),
    FactorRule("diastolic_bp", _value_at_least("diastolic_bp", 85), BLOOD_PRESSURE_MESSAGE),
    FactorRule(
        "glucose",
        _value_at_least("glucose", 100),
        "Blood glucose is above the typical range: suggest discussing a glucose check with a qualified healthcare professional.",
    ),
    FactorRule(
        "cholesterol",
        _value_at_least("cholesterol", 240),
        "Cholesterol is high: encourage a heart-healthy diet and discussing cholesterol with a healthcare professional.",
    ),
    FactorRule("smoker", _value_equals("smoker", "current"), "Suggest seeking professional smoking cessation support."),
    FactorRule(
        "alcohol_units_week",
        _value_at_least("alcohol_units_week", 15),
        "Alcohol intake is above 14 units a week: suggest gradually reducing alcohol consumption.",
    ),
    FactorRule(
        "exercise_days_week",
        _value_below("exercise_days_week", 3),
        "Exercise frequency is low: suggest gradually increasing physical activity, building towards most days of the week.",
    ),
    FactorRule("sleep_hours", _sleep_out_of_range, "Encourage consistent sleep habits, aiming for 7-9 hours per night."),
    FactorRule(
        "resting_hr",
        _value_at_least("resting_hr", 100),
        "Resting heart rate is high: suggest discussing it with a qualified healthcare professional.",
    ),
    FactorRule(
        "family_history",
        _value_equals("family_history", "yes"),
        "Family history is relevant: encourage sharing it with a healthcare professional when planning check-ups.",
    ),
]


def generate_recommendations(risk_level: str, top_factors: list[dict], patient: dict) -> list[str]:
    recommendations = list(RISK_LEVEL_GUIDANCE[risk_level])

    # Rules for the model's top factors go first, then any other rules that
    # apply to this patient. Python's sort is stable, so the original order
    # of FACTOR_RULES is kept within each group.
    top_feature_names = [factor["feature"] for factor in top_factors if factor["direction"] == "increases"]
    ordered_rules = sorted(FACTOR_RULES, key=lambda rule: rule.feature not in top_feature_names)

    for rule in ordered_rules:
        # Systolic and diastolic share a message, so skip duplicates.
        if rule.applies_to(patient) and rule.message not in recommendations:
            recommendations.append(rule.message)
    return recommendations
