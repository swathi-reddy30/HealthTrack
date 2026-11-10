"""Application settings, read from environment variables (or a local .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load a local .env file if one exists. Real environment variables still take
# priority, so the same code works on a laptop or in a container unchanged.
load_dotenv(PROJECT_ROOT / ".env")


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        # Fail fast at startup rather than running with a silently wrong setting.
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}") from error


def _read_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    path = Path(raw_value) if raw_value else default
    # Relative paths are resolved from the project root (not the current folder),
    # so the app behaves the same no matter where it is started from.
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    log_dir: Path
    log_level: str
    max_upload_bytes: int
    cache_max_size: int
    cache_ttl_seconds: int


def load_settings() -> Settings:
    return Settings(
        model_path=_read_path("MODEL_PATH", PROJECT_ROOT / "models" / "healthtrack_model.joblib"),
        log_dir=_read_path("LOG_DIR", PROJECT_ROOT / "logs"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_upload_bytes=_read_int("MAX_UPLOAD_BYTES", 2 * 1024 * 1024),  # 2 MB
        cache_max_size=_read_int("CACHE_MAX_SIZE", 1000),
        cache_ttl_seconds=_read_int("CACHE_TTL_SECONDS", 3600),  # 1 hour
    )


settings = load_settings()
