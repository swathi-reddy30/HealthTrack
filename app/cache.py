"""A small in-memory LRU (least recently used) cache for predictions, with a TTL.

Good enough for this single-process demo. With several server instances each
would have its own cache, so production would use a shared cache such as Redis.
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PredictionCache:
    def __init__(self, max_size: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # The clock can be swapped in tests to check expiry without waiting.
        self._clock = clock
        # OrderedDict remembers order, so the least recently used entry is first.
        # Values are stored as JSON text: every get() then returns a fresh copy, so
        # callers can never change the cached value (and it is faster than deepcopy).
        self._entries: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
        # FastAPI runs normal (def) endpoints in a thread pool, so two requests
        # can touch the cache at the same time; the lock keeps it consistent.
        self._lock = threading.Lock()

    @staticmethod
    def make_key(patient_features: dict, model_version: str) -> str:
        """Hash the normalised features plus model version.

        patient_id is deliberately left out: two patients with identical
        values get the same prediction. Numbers are converted to float so
        45 and 45.0 produce the same key, and sort_keys makes key order
        irrelevant. Including the model version means a new model never
        reuses old predictions.
        """
        normalised = {
            name: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            for name, value in patient_features.items()
        }
        payload = json.dumps({"model_version": model_version, "features": normalised}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                logger.debug("Cache miss for key %s", key[:12])
                return None

            stored_at, value = entry
            if self._clock() - stored_at > self.ttl_seconds:
                del self._entries[key]
                logger.debug("Cache entry expired for key %s", key[:12])
                return None

            self._entries.move_to_end(key)  # mark as most recently used
            logger.debug("Cache hit for key %s", key[:12])
            return json.loads(value)

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._entries[key] = (self._clock(), json.dumps(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)  # evict the least recently used entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
