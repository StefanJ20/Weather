from datetime import date

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render

from .services.cache import cache_clear
from .services.weather_summary import highest_temp_for_day


def index(request):
    return render(request, "index.html", {})


def api_highest_full(request):
    cache_clear()

    lat = request.GET.get("lat")
    lon = request.GET.get("lon")
    date_str = request.GET.get("date")

    if lat is None or lon is None:
        return HttpResponseBadRequest("Missing required 'lat' and 'lon' query parameters.")

    lat = lat.replace("°", "").strip()
    lon = lon.replace("°", "").strip()

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except ValueError:
        return HttpResponseBadRequest("Invalid 'lat' or 'lon' format.")

    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            return HttpResponseBadRequest("Invalid date format. Use YYYY-MM-DD.")

    data = highest_temp_for_day(lat_f, lon_f, target_date, tz_name=None)
    return JsonResponse(data)