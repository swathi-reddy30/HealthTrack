from app.recommendations import BLOOD_PRESSURE_MESSAGE, DISCLAIMER, RISK_LEVEL_GUIDANCE, generate_recommendations

HEALTHY_PATIENT = {
    "age": 30, "sex": "female", "bmi": 22, "systolic_bp": 115, "diastolic_bp": 75,
    "resting_hr": 65, "glucose": 85, "cholesterol": 170, "smoker": "never",
    "alcohol_units_week": 3, "exercise_days_week": 5, "sleep_hours": 8, "family_history": "no",
}


def factor(feature: str, direction: str = "increases") -> dict:
    return {"feature": feature, "contribution": 1.0 if direction == "increases" else -1.0, "direction": direction}


def test_low_risk_healthy_patient_gets_only_general_guidance():
    recommendations = generate_recommendations("low", [], HEALTHY_PATIENT)

    assert recommendations == RISK_LEVEL_GUIDANCE["low"]


def test_medium_risk_includes_medium_guidance():
    recommendations = generate_recommendations("medium", [], HEALTHY_PATIENT)

    assert recommendations[: len(RISK_LEVEL_GUIDANCE["medium"])] == RISK_LEVEL_GUIDANCE["medium"]


def test_high_risk_includes_high_guidance_and_attribute_rules():
    patient = {**HEALTHY_PATIENT, "smoker": "current", "bmi": 34}

    recommendations = generate_recommendations("high", [], patient)

    assert recommendations[: len(RISK_LEVEL_GUIDANCE["high"])] == RISK_LEVEL_GUIDANCE["high"]
    assert any("smoking cessation" in text for text in recommendations)
    assert any("physical activity and healthy" in text for text in recommendations)


def test_top_contributing_factors_come_first():
    patient = {**HEALTHY_PATIENT, "bmi": 34, "exercise_days_week": 1, "sleep_hours": 5}

    recommendations = generate_recommendations("medium", [factor("sleep_hours")], patient)
    specific = recommendations[len(RISK_LEVEL_GUIDANCE["medium"]):]

    assert "sleep" in specific[0]
    assert len(specific) == 3  # sleep, then bmi and exercise from the patient's values


def test_rule_only_fires_when_value_is_a_concern():
    # sleep_hours is a top factor, but 8 hours is healthy, so no sleep advice.
    recommendations = generate_recommendations("low", [factor("sleep_hours")], HEALTHY_PATIENT)

    assert not any("sleep" in text for text in recommendations)


def test_blood_pressure_message_is_not_duplicated():
    patient = {**HEALTHY_PATIENT, "systolic_bp": 150, "diastolic_bp": 95}

    recommendations = generate_recommendations("high", [factor("systolic_bp"), factor("diastolic_bp")], patient)

    assert recommendations.count(BLOOD_PRESSURE_MESSAGE) == 1


def test_missing_values_do_not_trigger_rules():
    patient = {name: None for name in HEALTHY_PATIENT}

    assert generate_recommendations("low", [], patient) == RISK_LEVEL_GUIDANCE["low"]


def test_disclaimer_text():
    assert "educational demonstration" in DISCLAIMER
    assert "does not provide medical advice" in DISCLAIMER
