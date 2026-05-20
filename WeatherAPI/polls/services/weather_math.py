import math
import re
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from django.conf import settings
from timezonefinder import TimezoneFinder  # type: ignore

from .cache import cached_get_json


TIMEZONE = getattr(settings, "NWS_DEFAULT_TIMEZONE", "America/Los_Angeles")
tf = TimezoneFinder()


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def tz_for_latlon(lat: float, lon: float) -> str:
    return tf.timezone_at(lat=lat, lng=lon) or "UTC"


def iso_to_dt(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def iso_z(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def day_range_local(target_date: date, tz_name: str):
    tz = ZoneInfo(tz_name)
    start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )

    return 2 * radius_km * math.asin(math.sqrt(a))


def extract_value_f(qv: Optional[dict]) -> Optional[float]:
    if not qv or qv.get("value") is None:
        return None

    val = float(qv["value"])
    unit_code = (qv.get("unitCode") or "").lower()

    if "degc" in unit_code or unit_code.endswith(":degc"):
        return c_to_f(val)

    return val


def wind_to_mph(qv: Optional[dict]) -> Optional[float]:
    if not qv or qv.get("value") is None:
        return None

    val = float(qv["value"])
    unit = (qv.get("unitCode") or "").lower()

    if "m_s-1" in unit or unit.endswith(":m_s-1"):
        return val * 2.2369362920544

    if "km_h-1" in unit or "km/h" in unit:
        return val * 0.621371

    if unit.endswith(":kn") or "knot" in unit or unit.endswith(":kt"):
        return val * 1.150779

    if "mi_h-1" in unit or "mph" in unit:
        return val

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


def max_from_forecast_periods(forecast_json, start_utc: datetime, end_utc: datetime):
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


def max_from_hourly(hourly_json, start_utc: datetime, end_utc: datetime):
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


def min_from_hourly(hourly_json, start_utc: datetime, end_utc: datetime):
    min_temp = None
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

        if min_temp is None or temp_f < min_temp:
            min_temp = temp_f

    return min_temp


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
        try:
            j = cached_get_json(url)
        except Exception:
            return max_temp

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

            if "degC" in unit_code or unit_code.lower().endswith("c"):
                val_f = c_to_f(val)
            else:
                val_f = val

            if max_temp is None or val_f > max_temp:
                max_temp = val_f

        url = (j.get("pagination") or {}).get("next")

    return max_temp


def min_from_station_observations(station_id: str, start_utc: datetime, end_utc: datetime):
    base = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": iso_z(start_utc),
        "end": iso_z(end_utc),
        "limit": 500,
    }
    url = f"{base}?{urlencode(params)}"

    min_temp = None

    while url:
        try:
            j = cached_get_json(url)
        except Exception:
            return min_temp

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

            if "degC" in unit_code or unit_code.lower().endswith("c"):
                val_f = c_to_f(val)
            else:
                val_f = val

            if min_temp is None or val_f < min_temp:
                min_temp = val_f

        url = (j.get("pagination") or {}).get("next")

    return min_temp


def station_max_last_hours(station_id: str, now_utc: datetime, hours: int):
    return max_from_station_observations(
        station_id,
        now_utc - timedelta(hours=hours),
        now_utc,
    )


def pick_closest_station_id(point_json, lat: float, lon: float, max_candidates: int = 200):
    stations_url = point_json.get("properties", {}).get("observationStations")

    if not stations_url:
        return None, None

    try:
        stations_json = cached_get_json(stations_url)
    except Exception:
        return None, None

    best = None
    best_d = None

    for f in stations_json.get("features", [])[:max_candidates]:
        sid = (f.get("properties") or {}).get("stationIdentifier")
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates")

        if not sid or not coords or len(coords) < 2:
            continue

        slon, slat = coords[0], coords[1]
        d = haversine_km(lat, lon, slat, slon)

        if best_d is None or d < best_d:
            best_d = d
            best = sid

    return best, best_d


def window_max_from_hourly(hourly_json, now_utc: datetime, hours: int):
    periods = hourly_json.get("properties", {}).get("periods", [])
    window = []

    for p in periods:
        st = iso_to_dt(p["startTime"])

        if st >= now_utc:
            window.append(p)

            if len(window) >= hours:
                break

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


def next_hour_from_hourly(hourly_json, now_utc: datetime):
    periods = hourly_json.get("properties", {}).get("periods", [])

    for p in periods:
        st = iso_to_dt(p["startTime"])

        if st >= now_utc:
            temp = p.get("temperature")
            unit = (p.get("temperatureUnit") or "F").upper()
            temp_f = None

            if temp is not None:
                temp_f = c_to_f(float(temp)) if unit == "C" else float(temp)

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

    st = (
        iso_to_dt(best_p["startTime"])
        .astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    return {
        "hours_ahead": hours_ahead,
        "start_utc": st,
        "temp_f": temp_f,
    }


def get_latest_station_observation(station_id: str) -> Dict[str, Any]:
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    try:
        j = cached_get_json(url, force_refresh=True)
    except Exception:
        return {
            "obs_time_utc": None,
            "obs_temp_f": None,
            "obs_dewpoint_f": None,
            "obs_wind_dir_deg": None,
            "obs_wind_speed_mph": None,
        }

    props = (j or {}).get("properties", {}) or {}

    ts = props.get("timestamp")
    obs_time = iso_to_dt(ts) if ts else None

    temp_f = extract_value_f(props.get("temperature"))
    dew_f = extract_value_f(props.get("dewpoint"))

    wind_dir = props.get("windDirection", {}).get("value")
    wind_speed_mph = wind_to_mph(props.get("windSpeed"))

    return {
        "obs_time_utc": obs_time.astimezone(timezone.utc).isoformat() if obs_time else None,
        "obs_temp_f": temp_f,
        "obs_dewpoint_f": dew_f,
        "obs_wind_dir_deg": float(wind_dir) if wind_dir is not None else None,
        "obs_wind_speed_mph": wind_speed_mph,
    }


def get_recent_station_observations(
    station_id: str,
    now_utc: datetime,
    lookback_minutes: int = 60,
    limit: int = 200,
):
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
        try:
            j = cached_get_json(url)
        except Exception:
            break

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
