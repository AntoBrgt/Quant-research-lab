"""Deterministic, content-addressed filesystem cache for expensive operations.

This is intentionally a plain JSON-file cache under `data/cache/<namespace>/` --
no Redis, no database. It is designed so the *key derivation* (the important,
easy-to-get-wrong part) can be swapped to a different backend later without
touching any caller.

A ticker alone is never a valid cache key: the key is derived from the actual
input content plus the model/prompt/schema versions that could change the
output, so a prompt or model change invalidates exactly what it should and
nothing more.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import config


def normalize_text(text: str) -> str:
    """Collapse whitespace so trivial formatting differences hash identically."""
    return re.sub(r"\s+", " ", text or "").strip()


def content_hash(text: str) -> str:
    """SHA256 of the normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_cache_key(
    operation: str,
    input_text: str,
    model: str,
    prompt_version: str = config.PROMPT_VERSION,
    schema_version: str = config.SCHEMA_VERSION,
) -> str:
    """Build a deterministic cache key.

    key = SHA256(operation + "|" + SHA256(normalized_input) + "|" + model
                 + "|" + prompt_version + "|" + schema_version)

    Same input/model/versions -> same key. Any change to any component
    (including the prompt or schema version) -> a different key, which is
    exactly the invalidation behavior we want without deleting the cache.
    """
    payload = "|".join(
        [operation, content_hash(input_text), model, prompt_version, schema_version]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_for(namespace: str, key: str) -> Path:
    return config.CACHE_DIR / namespace / f"{key}.json"


def exists(namespace: str, key: str) -> bool:
    if not config.CACHE_ENABLED:
        return False
    return _path_for(namespace, key).exists()


def get(namespace: str, key: str) -> Optional[dict]:
    """Return the cached value dict, or None on a miss / cache disabled."""
    if not config.CACHE_ENABLED:
        return None

    path = _path_for(namespace, key)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt cache entry should behave like a miss, not crash the run.
        return None


def set(namespace: str, key: str, value: dict[str, Any]) -> None:
    """Persist a value under the given namespace/key with an atomic write."""
    if not config.CACHE_ENABLED:
        return

    directory = config.CACHE_DIR / namespace
    directory.mkdir(parents=True, exist_ok=True)

    target = _path_for(namespace, key)

    fd, tmp_name = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, default=str)
        Path(tmp_name).replace(target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def stats(namespace: str) -> dict:
    """Cheap diagnostics: how many entries exist in a cache namespace."""
    directory = config.CACHE_DIR / namespace
    if not directory.exists():
        return {"namespace": namespace, "entries": 0}
    return {"namespace": namespace, "entries": sum(1 for _ in directory.glob("*.json"))}
