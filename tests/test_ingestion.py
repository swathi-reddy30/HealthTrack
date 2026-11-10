import pytest

from app.ingestion import CSVValidationError, ingest_csv
from tests.conftest import make_patient, to_csv_bytes


def test_valid_csv_is_fully_accepted():
    content = to_csv_bytes([make_patient(patient_id="A1"), make_patient(patient_id="A2")])

    result = ingest_csv(content)

    assert result.total_rows == 2
    assert [patient.patient_id for patient in result.patients] == ["A1", "A2"]
    assert result.row_errors == []


def test_column_names_and_categories_are_normalised():
    content = b"Patient_ID, Age ,sex,bmi,systolic_bp,diastolic_bp,resting_hr,glucose,cholesterol,smoker,alcohol_units_week,exercise_days_week,sleep_hours,family_history\n" \
              b"A1,40, Male ,25,120,80,70,90,180,NEVER,2,3,7,No\n"

    result = ingest_csv(content)

    patient = result.patients[0]
    assert patient.sex == "male"
    assert patient.smoker == "never"
    assert patient.family_history == "no"


def test_missing_required_columns_raises_error():
    rows = [make_patient()]
    for row in rows:
        del row["glucose"]
        del row["smoker"]

    with pytest.raises(CSVValidationError, match="glucose, smoker"):
        ingest_csv(to_csv_bytes(rows))


@pytest.mark.parametrize(
    "content",
    [b"", b"   \n", b"\xff\xfe\x00\x01binary-data"],
    ids=["empty", "whitespace", "not-utf8"],
)
def test_unreadable_files_raise_error(content):
    with pytest.raises(CSVValidationError):
        ingest_csv(content)


def test_header_without_rows_raises_error():
    header_only = to_csv_bytes([make_patient()]).split(b"\n")[0] + b"\n"
    with pytest.raises(CSVValidationError, match="no patient rows"):
        ingest_csv(header_only)


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"age": 400}, "age"),
        ({"bmi": "abc"}, "bmi"),
        ({"sex": "unknown"}, "sex"),
        ({"exercise_days_week": 9}, "exercise_days_week"),
        ({"sleep_hours": -1}, "sleep_hours"),
        ({"systolic_bp": 80, "diastolic_bp": 95}, "diastolic_bp must be lower than systolic_bp"),
        ({"patient_id": None}, "patient_id: Field required"),
    ],
)
def test_invalid_values_are_rejected_with_reason(overrides, expected_reason):
    result = ingest_csv(to_csv_bytes([make_patient(**overrides)]))

    assert result.patients == []
    assert len(result.row_errors) == 1
    assert expected_reason in result.row_errors[0].reason


def test_invalid_rows_are_reported_and_valid_rows_kept():
    rows = [
        make_patient(patient_id="GOOD1"),
        make_patient(patient_id="BAD1", age=400),
        make_patient(patient_id="GOOD2"),
        make_patient(patient_id="GOOD1"),  # duplicate ID
    ]

    result = ingest_csv(to_csv_bytes(rows))

    assert [patient.patient_id for patient in result.patients] == ["GOOD1", "GOOD2"]
    # Row numbers match the CSV file: the header is line 1, so the second data row is line 3.
    assert [(error.row_number, error.patient_id) for error in result.row_errors] == [(3, "BAD1"), (5, "GOOD1")]
    assert "Duplicate" in result.row_errors[1].reason


def test_blank_values_are_kept_as_missing_for_imputation():
    result = ingest_csv(to_csv_bytes([make_patient(glucose=None, smoker=None)]))

    assert result.row_errors == []
    assert result.patients[0].glucose is None
    assert result.patients[0].smoker is None


def test_extra_columns_are_ignored():
    result = ingest_csv(to_csv_bytes([make_patient(risk_level="high", notes="anything")]))

    assert len(result.patients) == 1
    assert result.row_errors == []
