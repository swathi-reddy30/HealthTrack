"""CSV ingestion: read the uploaded file, check its structure, validate each row.

There are two kinds of problems:
- File-level problems (not a CSV, missing columns, no rows) stop processing
  and raise CSVValidationError, which the API turns into HTTP 400.
- Row-level problems (impossible or invalid values) only reject that row.
  Valid rows are still processed and every rejected row is reported back
  with its row number and reason.
"""

import io
import logging
from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from app.preprocessing import FEATURE_COLUMNS, ID_COLUMN
from app.schemas import PatientInput, RowError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS
# pandas counts data rows from 0, and line 1 of the file is the header,
# so data row 0 is line 2 in the file the coach is looking at.
FIRST_DATA_LINE_NUMBER = 2
MAX_REPORTED_ID_LENGTH = 50


class CSVValidationError(ValueError):
    """The file as a whole cannot be processed."""


@dataclass
class IngestionResult:
    patients: list[PatientInput]
    row_errors: list[RowError]
    total_rows: int


def read_csv(content: bytes) -> pd.DataFrame:
    try:
        # "utf-8-sig" also removes the invisible BOM character Excel adds.
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CSVValidationError("File is not valid UTF-8 text. Please upload a CSV file.") from error

    if not text.strip():
        raise CSVValidationError("The uploaded file is empty.")

    try:
        # Read every value as text. Pydantic then does the type conversion,
        # so a bad value is reported for that row instead of failing the file.
        data = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise CSVValidationError("The file could not be parsed as CSV. Check that every row has the same columns.") from error

    data.columns = [str(column).strip().lower() for column in data.columns]
    return data


def check_required_columns(data: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise CSVValidationError(f"Missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise CSVValidationError("The CSV has a header row but no patient rows.")


def validate_rows(data: pd.DataFrame) -> IngestionResult:
    patients: list[PatientInput] = []
    row_errors: list[RowError] = []
    seen_patient_ids: set[str] = set()

    # Blank cells arrive from pandas as NaN. Converting them to None lets
    # Pydantic treat them as "missing" (imputed later) rather than invalid.
    required_data = data[REQUIRED_COLUMNS]
    records = required_data.astype(object).where(required_data.notna(), None).to_dict(orient="records")

    for index, record in enumerate(records):
        row_number = index + FIRST_DATA_LINE_NUMBER
        raw_patient_id = record.get(ID_COLUMN)
        reported_id = str(raw_patient_id)[:MAX_REPORTED_ID_LENGTH] if raw_patient_id else None

        # Leave blank cells out entirely: optional fields then default to
        # "missing", and a blank patient_id is reported as "Field required".
        filled_values = {column: value for column, value in record.items() if value is not None}
        try:
            patient = PatientInput(**filled_values)
        except ValidationError as error:
            row_errors.append(RowError(row_number=row_number, patient_id=reported_id, reason=format_validation_error(error)))
            continue

        # A repeated ID is almost always a copy-paste mistake; keep the first one.
        if patient.patient_id in seen_patient_ids:
            row_errors.append(
                RowError(
                    row_number=row_number,
                    patient_id=patient.patient_id,
                    reason="Duplicate patient_id; only the first occurrence is used.",
                )
            )
            continue

        seen_patient_ids.add(patient.patient_id)
        patients.append(patient)

    return IngestionResult(patients=patients, row_errors=row_errors, total_rows=len(records))


def ingest_csv(content: bytes) -> IngestionResult:
    data = read_csv(content)
    check_required_columns(data)
    result = validate_rows(data)

    # Only counts and row numbers are logged, never the patients' health values.
    logger.info(
        "CSV ingestion complete: %d rows, %d valid, %d rejected",
        result.total_rows,
        len(result.patients),
        len(result.row_errors),
    )
    if result.row_errors:
        rejected_row_numbers = [error.row_number for error in result.row_errors]
        logger.warning("Rejected CSV rows: %s", rejected_row_numbers)
    return result


def format_validation_error(error: ValidationError) -> str:
    """Turn a Pydantic error into a short, readable reason such as 'age: Input should be ...'."""
    messages = []
    for issue in error.errors():
        field_name = ".".join(str(part) for part in issue["loc"])
        message = issue["msg"].removeprefix("Value error, ")
        messages.append(f"{field_name}: {message}" if field_name else message)
    return "; ".join(messages)
