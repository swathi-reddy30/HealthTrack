"""Pydantic models for API input validation and response formats.

PatientInput is the single source of truth for what a valid patient looks
like. It validates JSON requests *and* each row of an uploaded CSV, so both
paths reject the same impossible values with the same messages.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sex = Literal["male", "female"]
SmokerStatus = Literal["never", "former", "current"]
YesNo = Literal["yes", "no"]
RiskLevel = Literal["low", "medium", "high"]


class PatientInput(BaseModel):
    """One patient's health data.

    Health fields are optional: a missing value is filled in later by the
    model's imputer. A value that *is* present must be within a plausible
    range (these bounds reject impossible data such as age = 400; they are
    not clinical reference ranges).
    """

    # extra="ignore": extra CSV columns (for example a risk_level label) are skipped.
    # allow_inf_nan=False: NaN/infinity are never valid health measurements.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, allow_inf_nan=False)

    # A restricted character set keeps IDs safe to display in the UI and in logs.
    patient_id: str = Field(min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_-]+$")
    age: Optional[int] = Field(default=None, ge=18, le=110)
    sex: Optional[Sex] = None
    bmi: Optional[float] = Field(default=None, ge=10, le=80)
    systolic_bp: Optional[float] = Field(default=None, ge=70, le=250)
    diastolic_bp: Optional[float] = Field(default=None, ge=40, le=150)
    resting_hr: Optional[float] = Field(default=None, ge=30, le=220)
    glucose: Optional[float] = Field(default=None, ge=40, le=500)
    cholesterol: Optional[float] = Field(default=None, ge=80, le=500)
    smoker: Optional[SmokerStatus] = None
    alcohol_units_week: Optional[float] = Field(default=None, ge=0, le=150)
    exercise_days_week: Optional[int] = Field(default=None, ge=0, le=7)
    sleep_hours: Optional[float] = Field(default=None, ge=0, le=24)
    family_history: Optional[YesNo] = None

    @field_validator("patient_id", mode="before")
    @classmethod
    def convert_numeric_id_to_text(cls, value):
        # JSON clients may send IDs like 1001; treat every ID as text.
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("sex", "smoker", "family_history", mode="before")
    @classmethod
    def normalise_category(cls, value):
        # Accept "Male" or " YES " so small formatting differences don't reject a row.
        if isinstance(value, str):
            cleaned = value.strip().lower()
            return cleaned or None  # an empty cell counts as missing
        return value

    @model_validator(mode="after")
    def check_blood_pressure_order(self):
        if self.systolic_bp is not None and self.diastolic_bp is not None:
            if self.diastolic_bp >= self.systolic_bp:
                raise ValueError("diastolic_bp must be lower than systolic_bp")
        return self


class LabelledPatientInput(PatientInput):
    """A training row: patient data plus the known risk level."""

    risk_level: RiskLevel

    @field_validator("risk_level", mode="before")
    @classmethod
    def normalise_label(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class PredictRequest(BaseModel):
    patients: list[PatientInput] = Field(min_length=1, max_length=1000)


class ContributingFactor(BaseModel):
    feature: str
    # The patient's own value (None if it was missing and filled in by imputation).
    value: Optional[float | str] = None
    contribution: float
    # Relative to the *predicted* class: "increases" means this feature pushed
    # the score for that risk level up.
    direction: Literal["increases", "decreases"]


class RiskProbabilities(BaseModel):
    low: float
    medium: float
    high: float


class PatientPrediction(BaseModel):
    patient_id: str
    risk_level: RiskLevel
    risk_label: str
    probabilities: RiskProbabilities
    top_factors: list[ContributingFactor]
    recommendations: list[str]
    cached: bool


class RowError(BaseModel):
    row_number: int = Field(description="Line number in the CSV file (the header is line 1).")
    patient_id: Optional[str] = None
    reason: str


class PredictionResponse(BaseModel):
    model_version: str
    total_rows: int
    # Server-side time for this request, so the speed-up from caching is visible.
    processing_time_ms: float
    predictions: list[PatientPrediction]
    row_errors: list[RowError] = []
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
