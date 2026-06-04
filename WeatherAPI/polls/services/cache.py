import copy
import logging
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

from django.conf import settings

from .nws_client import NWSNotModified, get_json_response

logger = logging.getLogger(__name__)

_CACHE_TTL = timedelta(
    seconds=getattr(settings, "NWS_CACHE_TTL_SECONDS", 300)
)
_CACHE_STALE_TTL = timedelta(
    seconds=getattr(settings, "NWS_CACHE_STALE_TTL_SECONDS", 1800)
)
_CACHE_MAX_ITEMS = int(getattr(settings, "NWS_CACHE_MAX_ITEMS", 512))


@dataclass
class CacheEntry:
    data: Any
    stored_at: float
    expires_at: float
    stale_expires_at: float
    etag: str | None = None
    last_modified: str | None = None
    cache_control: str | None = None
    expires_header: str | None = None
    update_time: str | None = None


_req_cache: Dict[str, CacheEntry] = {}
_cache_lock = threading.RLock()


def _now() -> float:
    return time.monotonic()


def _clone(data: Any) -> Any:
    try:
        return copy.deepcopy(data)
    except Exception:
        return data


def cache_clear() -> None:
    with _cache_lock:
        _req_cache.clear()


def cache_delete(url: str) -> None:
    with _cache_lock:
        _req_cache.pop(url, None)


def cache_size() -> int:
    with _cache_lock:
        return len(_req_cache)


def _get_entry(url: str) -> Optional[CacheEntry]:
    with _cache_lock:
        entry = _req_cache.get(url)

        if entry is None:
            return None

        now = _now()

        if now >= entry.stale_expires_at:
            _req_cache.pop(url, None)
            return None

        return entry


def _prune_if_needed() -> None:
    if _CACHE_MAX_ITEMS <= 0:
        _req_cache.clear()
        return

    if len(_req_cache) <= _CACHE_MAX_ITEMS:
        return

    now = _now()

    expired_keys = [
        key
        for key, entry in _req_cache.items()
        if now >= entry.stale_expires_at
    ]

    for key in expired_keys:
        _req_cache.pop(key, None)

    if len(_req_cache) <= _CACHE_MAX_ITEMS:
        return

    overflow = len(_req_cache) - _CACHE_MAX_ITEMS
    oldest_keys = sorted(
        _req_cache,
        key=lambda key: _req_cache[key].stored_at,
    )[:overflow]

    for key in oldest_keys:
        _req_cache.pop(key, None)


def _extract_update_time(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None

    props = data.get("properties")

    if not isinstance(props, dict):
        return None

    update_time = props.get("updateTime")

    if update_time:
        return str(update_time)

    generated_at = props.get("generatedAt")

    if generated_at:
        return str(generated_at)

    return None


def _store(url: str, data: Any, meta: dict[str, Any] | None = None) -> None:
    ttl_seconds = max(_CACHE_TTL.total_seconds(), 0.0)
    stale_seconds = max(_CACHE_STALE_TTL.total_seconds(), ttl_seconds)
    now = _now()
    meta = meta or {}

    with _cache_lock:
        _req_cache[url] = CacheEntry(
            data=_clone(data),
            stored_at=now,
            expires_at=now + ttl_seconds,
            stale_expires_at=now + stale_seconds,
            etag=meta.get("etag"),
            last_modified=meta.get("last_modified"),
            cache_control=meta.get("cache_control"),
            expires_header=meta.get("expires"),
            update_time=_extract_update_time(data),
        )
        _prune_if_needed()


def _refresh_entry_timestamps(url: str, entry: CacheEntry, meta: dict[str, Any] | None = None) -> None:
    ttl_seconds = max(_CACHE_TTL.total_seconds(), 0.0)
    stale_seconds = max(_CACHE_STALE_TTL.total_seconds(), ttl_seconds)
    now = _now()
    meta = meta or {}

    with _cache_lock:
        entry.stored_at = now
        entry.expires_at = now + ttl_seconds
        entry.stale_expires_at = now + stale_seconds

        if meta.get("etag"):
            entry.etag = meta.get("etag")

        if meta.get("last_modified"):
            entry.last_modified = meta.get("last_modified")

        if meta.get("cache_control"):
            entry.cache_control = meta.get("cache_control")

        if meta.get("expires"):
            entry.expires_header = meta.get("expires")

        _req_cache[url] = entry
        _prune_if_needed()


def _conditional_headers(entry: CacheEntry | None) -> dict[str, str]:
    if entry is None:
        return {}

    headers = {}

    if entry.etag:
        headers["If-None-Match"] = entry.etag

    if entry.last_modified:
        headers["If-Modified-Since"] = entry.last_modified

    return headers


def cached_get_json(
    url: str,
    force_refresh: bool = False,
    allow_stale_on_error: bool = True,
):
    if not url or not isinstance(url, str):
        raise ValueError("url must be a non-empty string")

    if not force_refresh:
        entry = _get_entry(url)

        if entry is not None and _now() < entry.expires_at:
            return _clone(entry.data)

    stale_entry = _get_entry(url)
    headers = {} if force_refresh else _conditional_headers(stale_entry)

    try:
        data, meta = get_json_response(url, extra_headers=headers)
    except NWSNotModified:
        if stale_entry is not None:
            _refresh_entry_timestamps(url, stale_entry)
            return _clone(stale_entry.data)

        logger.warning("NWS returned 304 without a cached entry for %s", url)
        data, meta = get_json_response(url)
    except Exception as exc:
        if allow_stale_on_error and stale_entry is not None:
            logger.warning(
                "Fetch failed for %s; returning stale cached response: %s",
                url,
                exc,
            )
            return _clone(stale_entry.data)

        logger.warning("Fetch failed for %s: %s", url, exc)
        raise

    _store(url, data, meta)
    return _clone(data)