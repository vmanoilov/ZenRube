"""Caching utilities for zenrube-mcp."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, cast

LOGGER = logging.getLogger("zenrube.cache")


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    def __init__(self) -> None:
        self._store: Dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store[key]
            if expiry and expiry < time.time():
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expiry = time.time() + ttl if ttl else None
        with self._lock:
            self._store[key] = (value, expiry)


class FileCache(CacheBackend):
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe_key = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        if len(safe_key) > 100:
            safe_key = safe_key[:100]
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        if not safe_key:
            safe_key = "entry"
        return self.directory / f"{safe_key}_{digest}.json"

    def get(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            expires_at = payload.get("expires_at")
            if expires_at and expires_at < time.time():
                path.unlink(missing_ok=True)
                return None
            return payload.get("value")
        except Exception as exc:  # pragma: no cover - logging side effect
            LOGGER.warning("Failed reading cache file %s: %s", path, exc)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        path = self._path(key)
        payload = {
            "value": value,
            "expires_at": time.time() + ttl if ttl else None,
        }
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed writing cache file %s: %s", path, exc)


class RedisCache(CacheBackend):
    def __init__(self, url: str) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise RuntimeError("redis package not installed") from exc
        self._client = redis.from_url(url)

    def get(self, key: str) -> Any:  # pragma: no cover - integration only
        value = self._client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> None:  # pragma: no cover
        payload = json.dumps(value)
        if ttl:
            self._client.setex(key, ttl, payload)
        else:
            self._client.set(key, payload)


class CacheManager:
    _backend: CacheBackend = cast(CacheBackend, InMemoryCache())
    _ttl: Optional[int] = None

    @classmethod
    def configure(
        cls,
        backend: CacheBackend,
        ttl: Optional[int] = None,
    ) -> None:
        cls._backend = backend
        cls._ttl = ttl

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> None:
        backend_name = config.get("backend", "memory")
        ttl = config.get("ttl")
        backend: CacheBackend
        if backend_name == "memory":
            if isinstance(cls._backend, InMemoryCache):
                backend = cls._backend
            else:
                backend = InMemoryCache()
        elif backend_name == "file":
            directory = Path(config.get("directory", ".zenrube-cache"))
            backend = FileCache(directory)
        elif backend_name == "redis":
            backend = RedisCache(config.get("url", "redis://localhost:6379/0"))
        else:
            raise ValueError(f"Unknown cache backend: {backend_name}")
        cls.configure(backend, ttl)

    @classmethod
    def get(cls, key: str) -> Any:
        return cls._backend.get(key)

    @classmethod
    def set(cls, key: str, value: Any, ttl: Optional[int] = None) -> None:
        effective_ttl = ttl if ttl is not None else cls._ttl
        cls._backend.set(key, value, effective_ttl)


def build_cache_key(*parts: str) -> str:
    return "::".join(parts)
