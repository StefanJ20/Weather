
from urllib.parse import urlencode
from django.conf import settings
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render

import requests #type: ignore
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder # type: ignore
from urllib.parse import urlencode 
import re
from typing import Optional, Dict, Any

tf = TimezoneFinder()


USER_AGENT = getattr(settings, "NWS_USER_AGENT", "MyApp/1.0 (youremail@example.com)")
TIMEZONE = getattr(settings, "NWS_DEFAULT_TIMEZONE", "America/Los_Angeles")


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0

def tz_for_latlon(lat: float, lon: float) -> str:
    return tf.timezone_at(lat=lat, lng=lon) or "UTC"

def iso_to_dt(s: str) -> datetime:
    # NWS sometimes uses Z; normalize for fromisoformat
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def get_json(url: str):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json,application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def day_range_local(target_date: date, tz_name: str):
    tz = ZoneInfo(tz_name)
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def max_from_forecast_periods(forecast_json, start_utc: datetime, end_utc: datetime):
    """
    /forecast periods are usually 12-hour blocks (day/night) with temps in F.
    We'll take the max temperature among periods that overlap the target day window.
    """
    max_temp = None
    periods = forecast_json.get("properties", {}).get("periods", [])
    for p in periods:
        st = iso_to_dt(p["startTime"])
        et = iso_to_dt(p["endTime"])
        if et <= start_utc or st >= end_utc:
            continue

        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if temp is None:
            continue
        temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)

        if max_temp is None or temp_f > max_temp:
            max_temp = temp_f
    return max_temp


def station_max_last_hours(station_id: str, now_utc: datetime, hours: int):
    return max_from_station_observations(
        station_id,
        now_utc - timedelta(hours=hours),
        now_utc
    )

def max_from_hourly(hourly_json, start_utc: datetime, end_utc: datetime):
    """
    /forecastHourly returns hourly periods with temperature and startTime.
    We'll take the max temperature for hours starting inside the day window.
    """
    max_temp = None
    periods = hourly_json.get("properties", {}).get("periods", [])
    for p in periods:
        st = iso_to_dt(p["startTime"])
        if not (start_utc <= st < end_utc):
            continue

        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if temp is None:
            continue
        temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)

        if max_temp is None or temp_f > max_temp:
            max_temp = temp_f
    return max_temp

def iso_z(dt: datetime) -> str:

    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def max_from_station_observations(station_id: str, start_utc: datetime, end_utc: datetime):
    base = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": iso_z(start_utc),
        "end": iso_z(end_utc),
        "limit": 500,
    }
    url = f"{base}?{urlencode(params)}"

    max_temp = None

    while url:
        j = get_json(url)

        for f in j.get("features", []):
            props = (f or {}).get("properties", {}) or {}
            ts = props.get("timestamp")
            if not ts:
                continue
            t = iso_to_dt(ts)
            if not (start_utc <= t < end_utc):
                continue

            temp_obj = props.get("temperature") or {}
            val = temp_obj.get("value")
            unit_code = temp_obj.get("unitCode") or ""
            if val is None:
                continue

            val = float(val)
            if "degC" in unit_code or unit_code.endswith("C"):
                val_f = c_to_f(val)
            else:
                val_f = val

            if max_temp is None or val_f > max_temp:
                max_temp = val_f

        # pagination
        url = (j.get("pagination") or {}).get("next")

    return max_temp

import math

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def pick_closest_station_id(point_json, lat: float, lon: float, max_candidates: int = 20):
    stations_url = point_json.get("properties", {}).get("observationStations")
    if not stations_url:
        return None
    stations_json = get_json(stations_url)
    best = None
    best_d = None
    for f in stations_json.get("features", [])[:max_candidates]:
        sid = (f.get("properties") or {}).get("stationIdentifier")
        geom = (f.get("geometry") or {})
        coords = geom.get("coordinates")
        if not sid or not coords or len(coords) < 2:
            continue
        slon, slat = coords[0], coords[1]
        d = haversine_km(lat, lon, slat, slon)
        if best_d is None or d < best_d:
            best_d = d
            best = sid
    return best


def window_max_from_hourly(hourly_json, now_utc: datetime, hours: int):
    """
    Returns max temp/dewpoint/wind across the next `hours` hourly periods starting
    from the first period whose startTime >= now_utc.
    """
    periods = hourly_json.get("properties", {}).get("periods", [])
    window = []

    # pick periods
    for p in periods:
        st = iso_to_dt(p["startTime"])
        if st >= now_utc:
            window.append(p)
            if len(window) >= hours:
                break

    # compute maxes
    max_temp_f = None
    max_dew_f = None
    max_wind_mph = None

    for p in window:
        temp = p.get("temperature")
        unit = (p.get("temperatureUnit") or "F").upper()
        if temp is not None:
            temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)
            max_temp_f = temp_f if max_temp_f is None else max(max_temp_f, temp_f)

        dp_f = extract_value_f(p.get("dewpoint"))
        if dp_f is not None:
            max_dew_f = dp_f if max_dew_f is None else max(max_dew_f, dp_f)

        wind_mph = parse_wind_speed_mph(p.get("windSpeed"))
        if wind_mph is not None:
            max_wind_mph = wind_mph if max_wind_mph is None else max(max_wind_mph, wind_mph)

    start_utc = None
    end_utc = None
    if window:
        start_utc = iso_to_dt(window[0]["startTime"]).astimezone(timezone.utc).isoformat()
        # end is start of last period + 1 hour (since each is 1h)
        end_dt = iso_to_dt(window[-1]["startTime"]) + timedelta(hours=1)
        end_utc = end_dt.astimezone(timezone.utc).isoformat()

    return {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "max_temp_f": max_temp_f,
        "max_dewpoint_f": max_dew_f,
        "max_wind_speed_mph": max_wind_mph,
        "period_count": len(window),
    }



def highest_temp_for_day(lat: float, lon: float, target_date: date = None, tz_name: str = TIMEZONE):
    if tz_name is None:
        tz_name = tz_for_latlon(lat, lon)

    tz = ZoneInfo(tz_name)

    if target_date is None:
        target_date = datetime.now(tz).date()

    pts_url = f"https://api.weather.gov/points/{lat},{lon}"
    pts = get_json(pts_url)

    start_utc, end_utc = day_range_local(target_date, tz_name)

    forecast_url = pts.get("properties", {}).get("forecast")
    hourly_url = pts.get("properties", {}).get("forecastHourly")
    station_id = pick_closest_station_id(pts, lat, lon)
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    today_local = datetime.now(ZoneInfo(tz_name)).date()
    future_date = target_date > today_local

    end_utc_full = end_utc 

    if target_date == today_local and now_utc < end_utc:
        end_utc = now_utc.replace(microsecond=0)

    results = {
        "lat": lat,
        "lon": lon,
        "date": target_date.isoformat(),
        "timezone": tz_name,
        "station_id": station_id,
        "window_start_utc": start_utc.isoformat(),
        "window_end_utc": end_utc.isoformat(),
        "window_end_utc_full": end_utc_full.isoformat(),
        "is_future_date": future_date,
        
    }


    # forecast (day/night blocks)
    if forecast_url:
        try:
            fjson = get_json(forecast_url)
            results["forecast_max_f"] = max_from_forecast_periods(fjson, start_utc, end_utc_full)
        except Exception as e:
            results["forecast_error"] = str(e)
            results["forecast_max_f"] = None
    else:
        results["forecast_max_f"] = None

    # hourly
    if hourly_url:
        try:
            hj = get_json(hourly_url)
            results["hourly_max_f"] = max_from_hourly(hj, start_utc, end_utc_full)

            now_utc = datetime.now(timezone.utc).replace(microsecond=0)
            horizons = []
            for h in range(1, 13):
                item = temp_at_horizon_from_hourly(hj, now_utc, h)
                if item:
                    horizons.append(item)

            results["best_temp_next_1_12h"] = horizons
            nxt = next_hour_from_hourly(hj, now_utc)
            results.update(nxt)

            try:
                nowcast = build_nowcast_layer(
                    station_id=station_id,
                    now_utc=now_utc,
                    forecast_next_hour_temp_f=results.get("next_hour_temp_f"),
                    lookback_minutes=90,
                    freshness_minutes=40,
                )
                results.update(nowcast)
            except Exception as e:
                results["nowcast_error"] = str(e)

            w3 = window_max_from_hourly(hj, now_utc, 3)
            w6 = window_max_from_hourly(hj, now_utc, 6)

            results["station_max_last_6h_f"]  = station_max_last_hours(station_id, now_utc, 6)
            results["station_max_last_24h_f"] = station_max_last_hours(station_id, now_utc, 24)

            results["next_3h_start_utc"] = w3["start_utc"]
            results["next_3h_end_utc"] = w3["end_utc"]
            results["next_3h_max_temp_f"] = w3["max_temp_f"]
            results["next_3h_max_dewpoint_f"] = w3["max_dewpoint_f"]
            results["next_3h_max_wind_speed_mph"] = w3["max_wind_speed_mph"]
            results["next_3h_period_count"] = w3["period_count"]

            results["next_6h_start_utc"] = w6["start_utc"]
            results["next_6h_end_utc"] = w6["end_utc"]
            results["next_6h_max_temp_f"] = w6["max_temp_f"]
            results["next_6h_max_dewpoint_f"] = w6["max_dewpoint_f"]
            results["next_6h_max_wind_speed_mph"] = w6["max_wind_speed_mph"]
            results["next_6h_period_count"] = w6["period_count"]

            results["debug_now_utc"] = datetime.now(timezone.utc).isoformat()
            results["debug_now_local"] = datetime.now(ZoneInfo(tz_name)).isoformat()
            results["debug_midnight_local"] = datetime(target_date.year, target_date.month, target_date.day, tzinfo=ZoneInfo(tz_name)).isoformat()
            results["debug_window_start_utc"] = start_utc.isoformat()
            results["debug_window_end_utc"] = end_utc.isoformat()


        except Exception as e:
            results["hourly_error"] = str(e)
            results["hourly_max_f"] = None
            results["next_hour_start_utc"] = None
            results["next_hour_temp_f"] = None
            results["next_hour_dewpoint_f"] = None
            results["next_hour_wind_speed_mph"] = None
            results["next_hour_wind_direction"] = None
            results["next_3h_start_utc"] = None
            results["next_3h_end_utc"] = None
            results["next_3h_max_temp_f"] = None
            results["next_3h_max_dewpoint_f"] = None
            results["next_3h_max_wind_speed_mph"] = None
            results["next_3h_period_count"] = 0

            results["next_6h_start_utc"] = None
            results["next_6h_end_utc"] = None
            results["next_6h_max_temp_f"] = None
            results["next_6h_max_dewpoint_f"] = None
            results["next_6h_max_wind_speed_mph"] = None
            results["next_6h_period_count"] = 0

    else:
        results["hourly_max_f"] = None
        results["next_hour_start_utc"] = None
        results["next_hour_temp_f"] = None
        results["next_hour_dewpoint_f"] = None
        results["next_hour_wind_speed_mph"] = None
        results["next_hour_wind_direction"] = None


    # station observations
    if station_id and not future_date:
        try:
            results["station_running_max_f"] = max_from_station_observations(station_id, start_utc, end_utc)
        except Exception as e:
            results["station_error"] = str(e)
            results["station_running_max_f"] = None
    else:
        results["station_running_max_f"] = None

    results["observed_max_f"] = results.get("station_running_max_f")

    forecast_candidates = [results.get("forecast_max_f"), results.get("hourly_max_f")]
    forecast_candidates = [x for x in forecast_candidates if x is not None]
    results["forecasted_max_f"] = max(forecast_candidates) if forecast_candidates else None

    best_candidates = [results.get("observed_max_f"), results.get("forecasted_max_f")]
    best_candidates = [x for x in best_candidates if x is not None]
    results["best_estimate_max_f"] = max(best_candidates) if best_candidates else None
    results["station_max_f"] = results.get("observed_max_f")
    results["overall_max_f"] = results.get("best_estimate_max_f")

    # --- next-hour best estimate (prefer recent station obs, else hourly forecast) ---
    results["best_next_hour_temp_f"] = results.get("next_hour_temp_f")
    results["best_next_hour_dewpoint_f"] = results.get("next_hour_dewpoint_f")
    results["best_next_hour_wind_speed_mph"] = results.get("next_hour_wind_speed_mph")
    results["best_next_hour_wind_direction"] = results.get("next_hour_wind_direction")

     # --- best estimate (next 3h / 6h): start from forecast-window maxes ---
    results["best_next_3h_max_temp_f"] = results.get("next_3h_max_temp_f")
    results["best_next_3h_max_dewpoint_f"] = results.get("next_3h_max_dewpoint_f")
    results["best_next_3h_max_wind_speed_mph"] = results.get("next_3h_max_wind_speed_mph")

    results["best_next_6h_max_temp_f"] = results.get("next_6h_max_temp_f")
    results["best_next_6h_max_dewpoint_f"] = results.get("next_6h_max_dewpoint_f")
    results["best_next_6h_max_wind_speed_mph"] = results.get("next_6h_max_wind_speed_mph")

    obs_max = max_from_station_observations(station_id, start_utc, end_utc)
    if obs_max is None:
        latest = get_latest_station_observation(station_id)
        obs_max = latest.get("obs_temp_f")
    results["station_running_max_f"] = obs_max

    # If station exists, try latest obs and use it if it's fresh
    if station_id and not future_date:
        try:
            obs = get_latest_station_observation(station_id)
            results.update(obs)

            # freshness check (90 minutes)
            if obs.get("obs_time_utc"):
                obs_time = datetime.fromisoformat(obs["obs_time_utc"])
                now_utc = datetime.now(timezone.utc).replace(microsecond=0)
                age_seconds = (now_utc - obs_time).total_seconds()
                results["obs_age_minutes"] = round(age_seconds / 60.0, 1)
                results["best_temp_now_obs_f"] = obs.get("obs_temp_f")
                results["best_temp_now_obs_time_utc"] = obs.get("obs_time_utc")

                FRESH_MINUTES = 20

                if 0 <= age_seconds <= FRESH_MINUTES * 60:
                    # Use obs as best estimate “right now into next hour”
                    if obs.get("obs_temp_f") is not None:
                        results["best_next_hour_temp_f"] = obs["obs_temp_f"]
                    if obs.get("obs_dewpoint_f") is not None:
                        results["best_next_hour_dewpoint_f"] = obs["obs_dewpoint_f"]
                    if obs.get("obs_wind_speed_mph") is not None:
                        results["best_next_hour_wind_speed_mph"] = obs["obs_wind_speed_mph"]
                    # wind direction degrees vs string; keep both if you want
                    if obs.get("obs_wind_dir_deg") is not None:
                        results["best_next_hour_wind_direction"] = f"{obs['obs_wind_dir_deg']:.0f}°"
                                        # Inject obs into 3h/6h window maxes (so real observed can override forecast)
                    if obs.get("obs_temp_f") is not None:
                        results["best_next_3h_max_temp_f"] = (
                            obs["obs_temp_f"] if results.get("best_next_3h_max_temp_f") is None
                            else max(results["best_next_3h_max_temp_f"], obs["obs_temp_f"])
                        )
                        results["best_next_6h_max_temp_f"] = (
                            obs["obs_temp_f"] if results.get("best_next_6h_max_temp_f") is None
                            else max(results["best_next_6h_max_temp_f"], obs["obs_temp_f"])
                        )
                    if obs.get("obs_dewpoint_f") is not None:
                        results["best_next_3h_max_dewpoint_f"] = (
                            obs["obs_dewpoint_f"] if results.get("best_next_3h_max_dewpoint_f") is None
                            else max(results["best_next_3h_max_dewpoint_f"], obs["obs_dewpoint_f"])
                        )
                        results["best_next_6h_max_dewpoint_f"] = (
                            obs["obs_dewpoint_f"] if results.get("best_next_6h_max_dewpoint_f") is None
                            else max(results["best_next_6h_max_dewpoint_f"], obs["obs_dewpoint_f"])
                        )
                    if obs.get("obs_wind_speed_mph") is not None:
                        results["best_next_3h_max_wind_speed_mph"] = (
                            obs["obs_wind_speed_mph"] if results.get("best_next_3h_max_wind_speed_mph") is None
                            else max(results["best_next_3h_max_wind_speed_mph"], obs["obs_wind_speed_mph"])
                        )
                        results["best_next_6h_max_wind_speed_mph"] = (
                            obs["obs_wind_speed_mph"] if results.get("best_next_6h_max_wind_speed_mph") is None
                            else max(results["best_next_6h_max_wind_speed_mph"], obs["obs_wind_speed_mph"])
                        )

        except Exception as e:
            results["latest_obs_error"] = str(e)
            results["obs_age_minutes"] = None
            results["obs_time_utc"] = None
            results["obs_temp_f"] = None
            results["obs_dewpoint_f"] = None
            results["obs_wind_speed_mph"] = None
            results["obs_wind_dir_deg"] = None


    return results


# ---------------- Django views ----------------

def index(request):
    # HTML page
    return render(request, "index.html", {})

def api_highest_full(request):
    lat = request.GET.get("lat")
    lon = request.GET.get("lon")
    date_str = request.GET.get("date")

    if lat is None or lon is None:
        return HttpResponseBadRequest("Missing required 'lat' and 'lon' query parameters.")

    # strip degree symbols just in case
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

def wind_to_mph(qv: Optional[dict]) -> Optional[float]:
    if not qv or qv.get("value") is None:
        return None

    val = float(qv["value"])
    unit = (qv.get("unitCode") or "").lower()

    # Common NWS unitCodes:
    # wmoUnit:m_s-1, wmoUnit:km_h-1, wmoUnit:kn
    if "m_s-1" in unit or unit.endswith(":m_s-1"):
        return val * 2.2369362920544
    if "km_h-1" in unit or "km/h" in unit:
        return val * 0.621371
    if unit.endswith(":kn") or "knot" in unit or unit.endswith(":kt"):
        return val * 1.150779
    # If they ever return mph directly (rare), just return it:
    if "mi_h-1" in unit or "mph" in unit:
        return val

    # Unknown: return None or treat as m/s, but better to fail loudly:
    return None

def parse_wind_speed_mph(wind_speed_str: Optional[str]) -> Optional[float]:
    if not wind_speed_str:
        return None
    s = wind_speed_str.strip().lower()
    if "calm" in s:
        return 0.0

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None

    return max(float(x) for x in nums)



def extract_value_f(qv: Optional[dict]) -> Optional[float]:
    """
    NWS obs fields like temperature/dewpoint come as:
      {"value": 12.3, "unitCode": "wmoUnit:degC"}
    Convert to °F when value is °C.
    """
    if not qv or qv.get("value") is None:
        return None

    val = float(qv["value"])
    unit_code = (qv.get("unitCode") or "").lower()
    if "degc" in unit_code or unit_code.endswith(":degc"):
        return c_to_f(val)
    return val


def get_latest_station_observation(station_id: str) -> Dict[str, Any]:
    """
    Pull latest station obs for temp/dewpoint/wind.
    """
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    j = get_json(url)
    props = (j or {}).get("properties", {}) or {}

    ts = props.get("timestamp")
    obs_time = iso_to_dt(ts) if ts else None

    temp_f = extract_value_f(props.get("temperature"))
    dew_f = extract_value_f(props.get("dewpoint"))

    wind_dir = props.get("windDirection", {}).get("value")
    wind_speed_mps = props.get("windSpeed", {}).get("value")  # often m/s
    wind_speed_mph = None
    if wind_speed_mps is not None:
        wind_speed_mph = wind_to_mph(props.get("windSpeed"))


    return {
        "obs_time_utc": obs_time.astimezone(timezone.utc).isoformat() if obs_time else None,
        "obs_temp_f": temp_f,
        "obs_dewpoint_f": dew_f,
        "obs_wind_dir_deg": float(wind_dir) if wind_dir is not None else None,
        "obs_wind_speed_mph": wind_speed_mph,
    }

def next_hour_from_hourly(hourly_json, now_utc: datetime):
    """
    Pick the first hourly period whose startTime is >= now_utc.
    """
    periods = hourly_json.get("properties", {}).get("periods", [])
    for p in periods:
        st = iso_to_dt(p["startTime"])
        if st >= now_utc:
            temp = p.get("temperature")
            unit = (p.get("temperatureUnit") or "F").upper()
            temp_f = None
            if temp is not None:
                temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)

            # dewpoint in hourly forecast is usually object {value, unitCode}
            dp_f = extract_value_f(p.get("dewpoint"))

            wind_mph = parse_wind_speed_mph(p.get("windSpeed"))
            wind_dir = p.get("windDirection")

            return {
                "next_hour_start_utc": st.astimezone(timezone.utc).isoformat(),
                "next_hour_temp_f": temp_f,
                "next_hour_dewpoint_f": dp_f,
                "next_hour_wind_speed_mph": wind_mph,
                "next_hour_wind_direction": wind_dir,
            }
    return {
        "next_hour_start_utc": None,
        "next_hour_temp_f": None,
        "next_hour_dewpoint_f": None,
        "next_hour_wind_speed_mph": None,
        "next_hour_wind_direction": None,
    }

def temp_at_horizon_from_hourly(hourly_json, now_utc: datetime, hours_ahead: int):
    """
    Return the hourly-forecast temperature for the period whose startTime is
    closest to (now_utc + hours_ahead). This is a point-in-time estimate, not a max.
    """
    target = now_utc + timedelta(hours=hours_ahead)
    periods = hourly_json.get("properties", {}).get("periods", [])

    best_p = None
    best_delta = None

    for p in periods:
        st = iso_to_dt(p["startTime"])
        if st < now_utc:
            continue
        delta = abs((st - target).total_seconds())
        if best_p is None or delta < best_delta:
            best_p = p
            best_delta = delta

    if not best_p:
        return None

    temp = best_p.get("temperature")
    unit = (best_p.get("temperatureUnit") or "F").upper()
    if temp is None:
        temp_f = None
    else:
        temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)

    st = iso_to_dt(best_p["startTime"]).astimezone(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "hours_ahead": hours_ahead,
        "start_utc": st,
        "temp_f": temp_f,
    }

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_recent_station_observations(station_id: str, now_utc: datetime, lookback_minutes: int = 60, limit: int = 200):
    """
    Pull station observations in the last lookback_minutes.
    Returns list of dicts sorted ascending by time: [{"t": datetime, "temp_f": float}, ...]
    """
    start_utc = now_utc - timedelta(minutes=lookback_minutes)

    base = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": iso_z(start_utc),
        "end": iso_z(now_utc),
        "limit": limit,
    }
    url = f"{base}?{urlencode(params)}"

    rows = []
    while url:
        j = get_json(url)
        for f in j.get("features", []):
            props = (f or {}).get("properties", {}) or {}
            ts = props.get("timestamp")
            if not ts:
                continue
            t = iso_to_dt(ts)
            if not (start_utc <= t <= now_utc):
                continue

            temp_obj = props.get("temperature") or {}
            val = temp_obj.get("value")
            unit_code = temp_obj.get("unitCode") or ""
            if val is None:
                continue

            val = float(val)
            temp_f = c_to_f(val) if ("degC" in unit_code or unit_code.lower().endswith("c")) else val

            rows.append({"t": t.astimezone(timezone.utc), "temp_f": temp_f})

        url = (j.get("pagination") or {}).get("next")

    rows.sort(key=lambda r: r["t"])
    return rows

def linear_trend_nowcast_temps(recent_obs, now_utc: datetime, horizons_minutes=(15, 30, 45, 60)):
    """
    Fit a simple line temp(t) using last N points; return predicted temps at horizons.
    Uses seconds-from-now as x (negative in the past), predicts positive horizons.
    """
    if not recent_obs or len(recent_obs) < 3:
        return None

    # Use only the most recent points (avoid super old flattening)
    recent = recent_obs[-12:]  # up to last 12 obs points

    xs = []
    ys = []
    for r in recent:
        dt = (r["t"] - now_utc).total_seconds()  # negative
        xs.append(dt)
        ys.append(r["temp_f"])

    # Simple least squares for y = a + b*x
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None

    b = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n)) / denom
    a = y_mean - b * x_mean

    out = []
    for m in horizons_minutes:
        x = m * 60.0
        y = a + b * x
        out.append({"minutes_ahead": m, "temp_f_trend": y})

    return out

def build_nowcast_layer(
    station_id: str,
    now_utc: datetime,
    forecast_next_hour_temp_f: Optional[float],
    lookback_minutes: int = 90,        # increased window
    freshness_minutes: int = 40,      # be a bit more forgiving
):
    """
    Nowcast = trend-based prediction blended toward NWS next-hour forecast.
    More tolerant: longer lookback and fallback behavior if trend can't be fit.
    """
    if not station_id:
        return {"nowcast_error": "no_station_id"}

    recent = get_recent_station_observations(station_id, now_utc, lookback_minutes=lookback_minutes)
    if not recent:
        return {"nowcast_error": "no_recent_obs"}

    latest_t = recent[-1]["t"]
    age_sec = (now_utc - latest_t).total_seconds()
    if age_sec < 0:
        age_sec = 0

    # If obs are stale, fall back to forecast (if available) instead of erroring
    if age_sec > freshness_minutes * 60:
        if forecast_next_hour_temp_f is not None:
            blended = []
            for m in (15, 30, 45, 60):
                blended.append({
                    "minutes_ahead": m,
                    "temp_f": round(float(forecast_next_hour_temp_f), 2),
                    "temp_f_trend": round(float(forecast_next_hour_temp_f), 2),
                    "weight_trend": 0.0
                })
            return {
                "nowcast_method": "forecast_fallback_stale_obs",
                "nowcast_generated_utc": now_utc.replace(microsecond=0).isoformat(),
                "nowcast_obs_count": len(recent),
                "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
                "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
                "nowcast_next_0_60m": blended,
            }
        return {"nowcast_error": f"stale_obs ({round(age_sec/60,1)} min)"}

    trend_preds = linear_trend_nowcast_temps(recent, now_utc)
    if not trend_preds:
        # fallback: use forecast if available, else repeat last observation
        if forecast_next_hour_temp_f is not None:
            blended = []
            for m in (15, 30, 45, 60):
                blended.append({
                    "minutes_ahead": m,
                    "temp_f": round(float(forecast_next_hour_temp_f), 2),
                    "temp_f_trend": None,
                    "weight_trend": 0.0
                })
            return {
                "nowcast_method": "forecast_fallback_no_trend",
                "nowcast_generated_utc": now_utc.replace(microsecond=0).isoformat(),
                "nowcast_obs_count": len(recent),
                "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
                "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
                "nowcast_next_0_60m": blended,
            }
        # repeat last observed temp
        last_temp = recent[-1]["temp_f"]
        blended = []
        for m in (15, 30, 45, 60):
            blended.append({
                "minutes_ahead": m,
                "temp_f": round(float(last_temp), 2),
                "temp_f_trend": round(float(last_temp), 2),
                "weight_trend": 1.0
            })
        return {
            "nowcast_method": "repeat_latest_obs",
            "nowcast_generated_utc": now_utc.replace(microsecond=0).isoformat(),
            "nowcast_obs_count": len(recent),
            "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
            "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
            "nowcast_next_0_60m": blended,
        }

    # existing blend logic
    blended = []
    method = "trend_only"
    for p in trend_preds:
        m = p["minutes_ahead"]
        temp_trend = p["temp_f_trend"]

        w_trend = clamp(0.95 - (m / 120.0), 0.45, 0.90)

        if forecast_next_hour_temp_f is None:
            temp_best = temp_trend
            method = "trend_only"
        else:
            temp_mix = w_trend * temp_trend + (1.0 - w_trend) * forecast_next_hour_temp_f
            cap = 3.0 if m <= 30 else 2.0
            temp_best = clamp(temp_mix, forecast_next_hour_temp_f - cap, forecast_next_hour_temp_f + cap)
            method = "trend_blend_capped"

        blended.append({
            "minutes_ahead": m,
            "temp_f": round(float(temp_best), 2),
            "temp_f_trend": round(float(temp_trend), 2),
            "weight_trend": round(float(w_trend), 2),
        })

    return {
        "nowcast_method": method,
        "nowcast_generated_utc": now_utc.replace(microsecond=0).isoformat(),
        "nowcast_obs_count": len(recent),
        "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
        "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
        "nowcast_next_0_60m": blended,
    }


