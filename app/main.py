"""FastAPI application: API routes, startup and error handling.

Start the server from the project root:
    uvicorn app.main:app --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app.cache import PredictionCache
from app.config import settings
from app.ingestion import CSVValidationError, ingest_csv
from app.logging_config import setup_logging
from app.model import RISK_LABELS, RiskModel, load_model
from app.preprocessing import FEATURE_COLUMNS, prepare_features
from app.recommendations import DISCLAIMER, generate_recommendations
from app.schemas import (
    HealthResponse,
    PatientInput,
    PatientPrediction,
    PredictionResponse,
    PredictRequest,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# Browsers report CSV files inconsistently (Windows often says
# "application/vnd.ms-excel"), so we accept these and then check the content itself.
ALLOWED_UPLOAD_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel", "text/plain", "application/octet-stream"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: the model is loaded a single time, not per request.
    setup_logging(settings.log_dir, settings.log_level)
    app.state.model = load_model_at_startup()
    app.state.cache = PredictionCache(max_size=settings.cache_max_size, ttl_seconds=settings.cache_ttl_seconds)
    yield


app = FastAPI(
    title="HealthTrack API",
    description="Educational demo: predicts synthetic patient risk levels with explanations and recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


def load_model_at_startup() -> Optional[RiskModel]:
    # If the model is missing, keep the server running so /health can report
    # the problem, instead of crashing with an unclear error.
    if not settings.model_path.exists():
        logger.warning("Model file not found at %s. Run 'python -m scripts.train_model'.", settings.model_path)
        return None
    try:
        model = load_model(settings.model_path)
    except Exception:
        logger.exception("Failed to load model from %s", settings.model_path)
        return None
    logger.info("Loaded model version %s", model.version)
    return model


def get_model(request: Request) -> RiskModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is not loaded. Train it with 'python -m scripts.train_model'.",
        )
    return model


def run_predictions(patients: list[PatientInput], model: RiskModel, cache: PredictionCache) -> list[PatientPrediction]:
    """Predict every patient, using the cache where possible.

    Cache misses are collected and sent to the model in one batch, which is
    much faster than calling the model once per patient.
    """
    start_time = time.perf_counter()
    patient_features = [patient.model_dump(include=set(FEATURE_COLUMNS)) for patient in patients]
    cache_keys = [PredictionCache.make_key(features, model.version) for features in patient_features]

    results: list[Optional[dict]] = [cache.get(key) for key in cache_keys]
    cached_flags = [result is not None for result in results]
    missing_indexes = [index for index, result in enumerate(results) if result is None]

    if missing_indexes:
        feature_table = pd.DataFrame([patient_features[index] for index in missing_indexes], columns=FEATURE_COLUMNS)
        model_outputs = model.predict(prepare_features(feature_table))
        for index, output in zip(missing_indexes, model_outputs):
            output["recommendations"] = generate_recommendations(
                output["risk_level"], output["top_factors"], patient_features[index]
            )
            cache.set(cache_keys[index], output)
            results[index] = output

    predictions = [
        PatientPrediction(
            patient_id=patient.patient_id,
            risk_label=RISK_LABELS[result["risk_level"]],
            cached=was_cached,
            **result,
        )
        for patient, result, was_cached in zip(patients, results, cached_flags)
    ]

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    cache_hits = sum(cached_flags)
    logger.info(
        "Predicted %d patients in %.1f ms (cache hits: %d, misses: %d)",
        len(patients),
        elapsed_ms,
        cache_hits,
        len(patients) - cache_hits,
    )
    return predictions


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=model.version if model is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest, request: Request) -> PredictionResponse:
    # FastAPI has already validated the JSON body against PredictRequest;
    # any invalid patient produces a 422 before this function runs.
    model = get_model(request)
    predictions = run_predictions(payload.patients, model, request.app.state.cache)
    return PredictionResponse(
        model_version=model.version,
        total_rows=len(payload.patients),
        predictions=predictions,
        row_errors=[],
        disclaimer=DISCLAIMER,
    )


@app.post("/predict/upload", response_model=PredictionResponse)
def predict_upload(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    model = get_model(request)
    check_upload_type(file)

    # Read at most one byte more than the limit: enough to know the file is
    # too large without reading a huge file into memory.
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        logger.warning("Rejected upload larger than %d bytes", settings.max_upload_bytes)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File is too large. The maximum size is {settings.max_upload_bytes // (1024 * 1024)} MB.",
        )

    try:
        ingestion = ingest_csv(content)
    except CSVValidationError as error:
        logger.warning("Rejected CSV upload: %s", error)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    # Partial success is still a 200: valid rows get predictions and the
    # rejected rows are listed in row_errors.
    predictions = run_predictions(ingestion.patients, model, request.app.state.cache) if ingestion.patients else []
    return PredictionResponse(
        model_version=model.version,
        total_rows=ingestion.total_rows,
        predictions=predictions,
        row_errors=ingestion.row_errors,
        disclaimer=DISCLAIMER,
    )


def check_upload_type(file: UploadFile) -> None:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if not filename.endswith(".csv") or content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files (.csv) can be uploaded.",
        )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    # A consistent, readable format. We don't echo back the submitted values.
    errors = [
        {"location": ".".join(str(part) for part in issue["loc"]), "message": issue["msg"]}
        for issue in error.errors()
    ]
    logger.info("Rejected request to %s with %d validation error(s)", request.url.path, len(errors))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Invalid request data.", "errors": errors},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # The full stack trace goes to our logs only; the user gets a safe message.
    logger.exception("Unexpected error while handling %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )
