from datetime import date

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render

from .services.cache import cache_clear
from .services.weather_summary import highest_temp_for_day


def index(request):
    return render(request, "index.html", {})


def api_highest_full(request):
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

    # Validate coordinate ranges
    if not (-90 <= lat_f <= 90):
        return HttpResponseBadRequest("Latitude must be between -90 and 90 degrees.")
    
    if not (-180 <= lon_f <= 180):
        return HttpResponseBadRequest("Longitude must be between -180 and 180 degrees.")

    target_date = None
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            return HttpResponseBadRequest("Invalid date format. Use YYYY-MM-DD.")

    # Clear cache only if we're going to proceed with the operation
    cache_clear()

    try:
        data = highest_temp_for_day(lat_f, lon_f, target_date, tz_name=None)
        if data is None:
            return HttpResponseBadRequest("No weather data available for the specified location and date.")
        return JsonResponse(data)
    except Exception as e:
        return HttpResponseBadRequest(f"Error processing weather data: {str(e)}")
