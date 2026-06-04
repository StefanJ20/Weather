import json
import logging
from typing import Any

import requests
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

USER_AGENT = getattr(settings, "NWS_USER_AGENT", "MyApp/1.0 (contact@example.com)")
NWS_CONNECT_TIMEOUT_SECONDS = getattr(settings, "NWS_CONNECT_TIMEOUT_SECONDS", 5)
NWS_READ_TIMEOUT_SECONDS = getattr(settings, "NWS_READ_TIMEOUT_SECONDS", 20)
NWS_MAX_RETRIES = getattr(settings, "NWS_MAX_RETRIES", 3)
NWS_BACKOFF_FACTOR = getattr(settings, "NWS_BACKOFF_FACTOR", 0.5)
NWS_MAX_LOG_BODY_CHARS = getattr(settings, "NWS_MAX_LOG_BODY_CHARS", 1000)

_session: requests.Session | None = None


class NWSAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        url: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


class NWSNotModified(RuntimeError):
    def __init__(self, url: str):
        super().__init__("NWS response was not modified")
        self.url = url


def _build_session() -> requests.Session:
    retry = Retry(
        total=NWS_MAX_RETRIES,
        connect=NWS_MAX_RETRIES,
        read=NWS_MAX_RETRIES,
        status=NWS_MAX_RETRIES,
        backoff_factor=NWS_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session() -> requests.Session:
    global _session

    if _session is None:
        _session = _build_session()

    return _session


def _headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": str(USER_AGENT),
        "Accept": "application/geo+json, application/json",
    }

    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v})

    return headers


def _safe_body(response: requests.Response) -> str:
    text = response.text or ""

    if len(text) > NWS_MAX_LOG_BODY_CHARS:
        return text[:NWS_MAX_LOG_BODY_CHARS] + "..."

    return text


def _response_meta(response: requests.Response) -> dict[str, Any]:
    return {
        "status_code": response.status_code,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "cache_control": response.headers.get("Cache-Control"),
        "expires": response.headers.get("Expires"),
        "content_type": response.headers.get("Content-Type"),
    }


def get_json_response(
    url: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    timeout = (NWS_CONNECT_TIMEOUT_SECONDS, NWS_READ_TIMEOUT_SECONDS)

    try:
        response = get_session().get(
            url,
            headers=_headers(extra_headers),
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        logger.warning("NWS request timed out", extra={"url": url})
        raise NWSAPIError("NWS request timed out", url=url) from exc
    except requests.exceptions.RequestException as exc:
        logger.warning("NWS request failed", extra={"url": url, "error": str(exc)})
        raise NWSAPIError("NWS request failed", url=url) from exc

    if response.status_code == 304:
        raise NWSNotModified(url)

    if response.status_code == 204:
        return {}, _response_meta(response)

    if response.status_code >= 400:
        body = _safe_body(response)
        logger.warning(
            "NWS returned error response",
            extra={
                "url": url,
                "status_code": response.status_code,
                "response_body": body,
            },
        )
        raise NWSAPIError(
            f"NWS request returned HTTP {response.status_code}",
            url=url,
            status_code=response.status_code,
            response_text=body,
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        body = _safe_body(response)
        logger.warning(
            "NWS returned invalid JSON",
            extra={
                "url": url,
                "status_code": response.status_code,
                "response_body": body,
            },
        )
        raise NWSAPIError(
            "NWS returned invalid JSON",
            url=url,
            status_code=response.status_code,
            response_text=body,
        ) from exc

    if not isinstance(data, dict):
        logger.warning(
            "NWS returned unexpected JSON type",
            extra={"url": url, "json_type": type(data).__name__},
        )
        raise NWSAPIError(
            f"NWS returned unexpected JSON type: {type(data).__name__}",
            url=url,
            status_code=response.status_code,
        )

    return data, _response_meta(response)


def get_json(url: str) -> dict[str, Any]:
    data, _ = get_json_response(url)
    return data