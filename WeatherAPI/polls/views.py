from datetime import date, timedelta

from django.http import JsonResponse #type: ignore
from django.shortcuts import render #type: ignore
from django.views.decorators.http import require_GET #type: ignore
from django.views.decorators.csrf import csrf_exempt #type: ignore
from django.views.decorators.http import require_POST #type: ignore
from .services.cache import cache_clear
from .services.weather_summary import highest_temp_for_day


def index(request):
    return render(request, "index.html", {})


def _bad_request(message: str, details: dict | None = None):
    payload = {
        "error": message,
    }

    if details:
        payload["details"] = details

    return JsonResponse(payload, status=400)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float_param(value: str | None, name: str):
    if value is None:
        raise ValueError(f"Missing required '{name}' query parameter.")

    cleaned = value.replace("°", "").strip()

    if not cleaned:
        raise ValueError(f"Missing required '{name}' query parameter.")

    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid '{name}' format.") from exc


def _parse_target_date(value: str | None):
    if not value:
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.") from exc


@require_GET
def api_highest_full(request):
    try:
        lat_f = _parse_float_param(request.GET.get("lat"), "lat")
        lon_f = _parse_float_param(request.GET.get("lon"), "lon")
    except ValueError as exc:
        return _bad_request(str(exc))

    if not (-90 <= lat_f <= 90):
        return _bad_request("Latitude must be between -90 and 90 degrees.")

    if not (-180 <= lon_f <= 180):
        return _bad_request("Longitude must be between -180 and 180 degrees.")

    try:
        target_date = _parse_target_date(request.GET.get("date"))
    except ValueError as exc:
        return _bad_request(str(exc))

    today = date.today()
    min_supported_date = today - timedelta(days=14)
    max_supported_date = today + timedelta(days=7)

    if target_date is not None and target_date < min_supported_date:
        return _bad_request(
            "Date is too far in the past for reliable NWS station/API data.",
            {
                "min_supported_date": min_supported_date.isoformat(),
                "requested_date": target_date.isoformat(),
            },
        )

    if target_date is not None and target_date > max_supported_date:
        return _bad_request(
            "Date is too far in the future for reliable NWS forecast data.",
            {
                "max_supported_date": max_supported_date.isoformat(),
                "requested_date": target_date.isoformat(),
            },
        )

    refresh = _parse_bool(request.GET.get("refresh"))

    if refresh:
        cache_clear()

    try:
        data = highest_temp_for_day(
            lat=lat_f,
            lon=lon_f,
            target_date=target_date,
            tz_name=None,
        )
    except Exception as exc:
        return JsonResponse(
            {
                "error": "Error processing weather data.",
                "details": str(exc),
            },
            status=502,
        )

    if not data:
        return JsonResponse(
            {
                "error": "No weather data available for the specified location and date.",
            },
            status=404,
        )

    return JsonResponse(data, json_dumps_params={"indent": 2})

from .llm import weather_impression # type: ignore
import json

@csrf_exempt
@require_POST
def weather_ai_impression(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    weather = payload.get("weather")

    if not isinstance(weather, dict):
        return JsonResponse({"error": "Missing weather object"}, status=400)

    result = weather_impression(weather)
    return JsonResponse(result)