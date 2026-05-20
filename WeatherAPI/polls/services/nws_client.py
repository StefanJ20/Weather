import requests  # type: ignore
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

USER_AGENT = getattr(settings, "NWS_USER_AGENT", "MyApp/1.0 (contact@example.com)")
NWS_TIMEOUT_SECONDS = getattr(settings, "NWS_TIMEOUT_SECONDS", 30)


def get_json(url: str):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json,application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=NWS_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for URL {url}: {str(e)}")
        raise
