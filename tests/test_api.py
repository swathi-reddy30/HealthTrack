import logging

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.recommendations import DISCLAIMER
from tests.conftest import TEST_MODEL_VERSION, make_patient, to_csv_bytes


def upload(client, content: bytes, filename: str = "patients.csv", content_type: str = "text/csv"):
    return client.post("/predict/upload", files={"file": (filename, content, content_type)})


def test_health_reports_model_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": TEST_MODEL_VERSION}


def test_frontend_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "HealthTrack" in response.text


def test_json_prediction_success(client):
    response = client.post("/predict", json={"patients": [make_patient(patient_id="J1"), make_patient(patient_id="J2")]})

    assert response.status_code == 200
    body = response.json()
    assert body["disclaimer"] == DISCLAIMER
    assert body["model_version"] == TEST_MODEL_VERSION
    assert body["row_errors"] == []
    assert [prediction["patient_id"] for prediction in body["predictions"]] == ["J1", "J2"]

    prediction = body["predictions"][0]
    assert prediction["risk_level"] in {"low", "medium", "high"}
    assert set(prediction["probabilities"]) == {"low", "medium", "high"}
    assert len(prediction["top_factors"]) == 3
    assert prediction["recommendations"]
    assert prediction["cached"] is False


def test_repeated_prediction_is_served_from_cache(client):
    payload = {"patients": [make_patient()]}

    first = client.post("/predict", json=payload).json()["predictions"][0]
    second = client.post("/predict", json=payload).json()["predictions"][0]

    assert first["cached"] is False
    assert second["cached"] is True
    assert {**second, "cached": False} == first


def test_invalid_patient_input_returns_422(client):
    response = client.post("/predict", json={"patients": [make_patient(age=400)]})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Invalid request data."
    assert body["errors"][0]["location"] == "body.patients.0.age"


def test_empty_patient_list_returns_422(client):
    assert client.post("/predict", json={"patients": []}).status_code == 422


def test_csv_upload_success(client):
    response = upload(client, to_csv_bytes([make_patient(patient_id="C1"), make_patient(patient_id="C2", age=30)]))

    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 2
    assert len(body["predictions"]) == 2
    assert body["row_errors"] == []
    assert body["disclaimer"] == DISCLAIMER


def test_partial_success_with_invalid_rows(client):
    rows = [make_patient(patient_id="OK1"), make_patient(patient_id="BAD1", age=400), make_patient(patient_id="OK2", glucose=None)]

    response = upload(client, to_csv_bytes(rows))

    assert response.status_code == 200
    body = response.json()
    assert [prediction["patient_id"] for prediction in body["predictions"]] == ["OK1", "OK2"]
    assert body["row_errors"] == [
        {"row_number": 3, "patient_id": "BAD1", "reason": "age: Input should be less than or equal to 110"}
    ]


def test_all_rows_invalid_still_returns_200_with_errors(client):
    response = upload(client, to_csv_bytes([make_patient(age=400)]))

    assert response.status_code == 200
    assert response.json()["predictions"] == []
    assert len(response.json()["row_errors"]) == 1


def test_invalid_csv_file_returns_400_and_is_logged(client, caplog):
    with caplog.at_level(logging.WARNING):
        response = upload(client, b"\xff\xfe\x00\x01 not text")

    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]
    assert "Rejected CSV upload" in caplog.text


def test_non_csv_file_type_returns_400_and_is_logged(client, caplog):
    with caplog.at_level(logging.WARNING):
        response = upload(client, b"%PDF-1.4", filename="report.pdf", content_type="application/pdf")

    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]
    assert "unsupported file type" in caplog.text


def test_response_reports_processing_time(client):
    body = client.post("/predict", json={"patients": [make_patient()]}).json()

    assert body["processing_time_ms"] >= 0


def test_missing_required_columns_returns_400(client):
    response = upload(client, b"patient_id,age\nA1,40\n")

    assert response.status_code == 400
    assert "Missing required columns" in response.json()["detail"]


def test_oversized_file_returns_413(client):
    oversized = b"a" * (settings.max_upload_bytes + 1)

    response = upload(client, oversized)

    assert response.status_code == 413


def test_missing_model_returns_503(client):
    app.state.model = None

    assert client.post("/predict", json={"patients": [make_patient()]}).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False


def test_unexpected_error_returns_safe_500(risk_model, monkeypatch):
    def broken_predict(features):
        raise RuntimeError("secret internal detail")

    # raise_server_exceptions=False makes the TestClient return the 500
    # response (like a real server) instead of raising the error in the test.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        app.state.model = risk_model
        monkeypatch.setattr(risk_model, "predict", broken_predict)
        response = test_client.post("/predict", json={"patients": [make_patient()]})

    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["detail"] == "An unexpected error occurred. Please try again later."
