import math
import re
from datetime import datetime, date, timedelta, timezone
from typing import Optional, Dict, Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from django.conf import settings # type: ignore
from timezonefinder import TimezoneFinder # type: ignore

from .cache import cached_get_json


TIMEZONE = getattr(settings, "NWS_DEFAULT_TIMEZONE", "America/Los_Angeles")
tf = TimezoneFinder()

import re
from typing import Optional


def metar_signed_tenths_c(code: str) -> Optional[float]:
    """
    Converts METAR signed tenths-C codes.

    Examples:
    0294  -> 29.4 C
    1028  -> -2.8 C
    """
    if not code or len(code) != 4 or not code.isdigit():
        return None

    sign = -1.0 if code[0] == "1" else 1.0
    value = float(code[1:]) / 10.0
    return sign * value


def metar_remark_6h_max_f(raw_message: str) -> Optional[float]:
    """
    Finds 6-hour max temp from METAR remarks.

    Example:
    10294 -> +29.4 C -> 84.92 F
    11028 -> -2.8 C
    """
    if not raw_message:
        return None

    m = re.search(r"\b1([01]\d{3})\b", raw_message)

    if not m:
        return None

    temp_c = metar_signed_tenths_c(m.group(1))

    if temp_c is None:
        return None

    return valid_temp_f(c_to_f(temp_c))


def metar_remark_6h_min_f(raw_message: str) -> Optional[float]:
    """
    Finds 6-hour min temp from METAR remarks.

    Example:
    20228 -> +22.8 C
    21028 -> -2.8 C
    """
    if not raw_message:
        return None

    m = re.search(r"\b2([01]\d{3})\b", raw_message)

    if not m:
        return None

    temp_c = metar_signed_tenths_c(m.group(1))

    if temp_c is None:
        return None

    return valid_temp_f(c_to_f(temp_c))

def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def meters_to_miles(m: float) -> float:
    return m * 0.000621371


def meters_to_feet(m: float) -> float:
    return m * 3.280839895


def pa_to_hpa(pa: float) -> float:
    return pa / 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))

def finite_float(value) -> Optional[float]:
    if value is None:
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def valid_temp_f(value) -> Optional[float]:
    x = finite_float(value)

    if x is None:
        return None

    # Broad enough for CONUS extremes and bad-source rejection.
    if -120.0 <= x <= 160.0:
        return x

    return None


def valid_dewpoint_f(value, temp_f: Optional[float] = None) -> Optional[float]:
    dew = finite_float(value)

    if dew is None:
        return None

    # Dew points above ~100F or below -120F are usually bad data for this app.
    if not (-120.0 <= dew <= 100.0):
        return None

    temp = valid_temp_f(temp_f)

    # Dew point should not materially exceed air temp.
    # Allow +2F because station/grid rounding can create small violations.
    if temp is not None and dew > temp + 2.0:
        return None

    return dew


def valid_wind_speed_mph(value) -> Optional[float]:
    wind = finite_float(value)

    if wind is None:
        return None

    # Allows extreme severe-weather gusts but rejects broken values.
    if 0.0 <= wind <= 250.0:
        return wind

    return None


def valid_wind_gust_mph(value, sustained_mph: Optional[float] = None) -> Optional[float]:
    gust = valid_wind_speed_mph(value)

    if gust is None:
        return None

    sustained = valid_wind_speed_mph(sustained_mph)

    # Gust below sustained is not useful as a "gust" value.
    if sustained is not None and gust < sustained:
        return None

    return gust


def valid_wind_direction_deg(value) -> Optional[float]:
    direction = finite_float(value)

    if direction is None:
        return None

    if 0.0 <= direction <= 360.0:
        return direction

    return None


def metric_confidence(
    source: Optional[str],
    value,
    *,
    age_minutes: Optional[float] = None,
    station_distance_km: Optional[float] = None,
) -> str:
    if value is None:
        return "low"

    return confidence_from_source(source, age_minutes, station_distance_km)


def tz_for_latlon(lat: float, lon: float) -> str:
    return (
        tf.timezone_at_land(lat=lat, lng=lon)
        or tf.timezone_at(lat=lat, lng=lon)
        or TIMEZONE
        or "UTC"
    )


def iso_to_dt(s: str) -> datetime:
    if not s:
        raise ValueError("missing datetime string")

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

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
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def value_unit(qv: Optional[dict]):
    if not qv or qv.get("value") is None:
        return None, ""

    try:
        val = float(qv["value"])
    except (ValueError, TypeError):
        return None, ""

    return val, (qv.get("unitCode") or qv.get("uom") or "").lower()


def is_celsius_unit(unit_code: str) -> bool:
    u = (unit_code or "").lower()
    return "degc" in u or u.endswith(":c") or u.endswith("deg_c")


def extract_value_f(qv: Optional[dict]) -> Optional[float]:
    val, unit_code = value_unit(qv)

    if val is None:
        return None

    if is_celsius_unit(unit_code):
        return valid_temp_f(c_to_f(val))

    return valid_temp_f(val)


def wind_to_mph(qv: Optional[dict]) -> Optional[float]:
    val, unit = value_unit(qv)

    if val is None:
        return None

    if "m_s-1" in unit or "m/s" in unit:
        return valid_wind_speed_mph(val * 2.2369362920544)

    if "km_h-1" in unit or "km/h" in unit:
        return valid_wind_speed_mph(val * 0.621371)

    if unit.endswith(":kn") or unit.endswith(":kt") or "knot" in unit:
        return valid_wind_speed_mph(val * 1.150779)

    if "mi_h-1" in unit or "mph" in unit:
        return valid_wind_speed_mph(val)

    return valid_wind_speed_mph(val)


def parse_wind_speed_mph(wind_speed_value) -> Optional[float]:
    if wind_speed_value is None:
        return None

    if isinstance(wind_speed_value, dict):
        return wind_to_mph(wind_speed_value)

    s = str(wind_speed_value).strip().lower()

    if not s:
        return None

    if "calm" in s:
        return 0.0

    nums = re.findall(r"\d+(?:\.\d+)?", s)

    if not nums:
        return None

    try:
        val = max(float(x) for x in nums)
    except (ValueError, TypeError):
        return None

    if "km/h" in s or "kmh" in s or "kph" in s:
        return valid_wind_speed_mph(val * 0.621371)

    if "kt" in s or "knot" in s:
        return valid_wind_speed_mph(val * 1.150779)

    if "m/s" in s:
        return valid_wind_speed_mph(val * 2.2369362920544)

    return valid_wind_speed_mph(val)


def period_temp_f(period: dict) -> Optional[float]:
    qv_temp = period.get("temperature")

    if isinstance(qv_temp, dict):
        return extract_value_f(qv_temp)

    temp = qv_temp
    unit = (period.get("temperatureUnit") or "F").upper()

    if temp is None:
        return None

    try:
        temp_float = float(temp)
    except (ValueError, TypeError):
        return None

    if unit == "C":
        return valid_temp_f(c_to_f(temp_float))

    return valid_temp_f(temp_float)


def period_dewpoint_f(period: dict, temp_f: Optional[float] = None) -> Optional[float]:
    return valid_dewpoint_f(extract_value_f(period.get("dewpoint")), temp_f)


def period_wind_speed_mph(period: dict) -> Optional[float]:
    return valid_wind_speed_mph(parse_wind_speed_mph(period.get("windSpeed")))


def period_times(period: dict):
    st_raw = period.get("startTime")
    et_raw = period.get("endTime")

    if not st_raw:
        return None, None

    st = iso_to_dt(st_raw)
    et = iso_to_dt(et_raw) if et_raw else st + timedelta(hours=1)
    return st, et


def format_local_label(dt: datetime) -> str:
    return f"{dt.strftime('%b')} {dt.day}, {dt.strftime('%I:%M %p').lstrip('0')}"


def max_from_forecast_periods(forecast_json, start_utc: datetime, end_utc: datetime):
    max_temp = None
    periods = forecast_json.get("properties", {}).get("periods", []) if forecast_json else []

    for p in periods:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if et <= start_utc or st >= end_utc:
            continue

        temp_f = period_temp_f(p)

        if temp_f is None:
            continue

        max_temp = temp_f if max_temp is None else max(max_temp, temp_f)

    return max_temp


def max_from_hourly(hourly_json, start_utc: datetime, end_utc: datetime):
    max_temp = None
    periods = hourly_json.get("properties", {}).get("periods", []) if hourly_json else []

    for p in periods:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if et <= start_utc or st >= end_utc:
            continue

        temp_f = period_temp_f(p)

        if temp_f is None:
            continue

        max_temp = temp_f if max_temp is None else max(max_temp, temp_f)

    return max_temp


def min_from_hourly(hourly_json, start_utc: datetime, end_utc: datetime):
    min_temp = None
    periods = hourly_json.get("properties", {}).get("periods", []) if hourly_json else []

    for p in periods:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if et <= start_utc or st >= end_utc:
            continue

        temp_f = period_temp_f(p)

        if temp_f is None:
            continue

        min_temp = temp_f if min_temp is None else min(min_temp, temp_f)

    return min_temp


def station_observation_temp_f(props: dict) -> Optional[float]:
    return valid_temp_f(extract_value_f(props.get("temperature")))


def station_observation_dewpoint_f(props: dict, temp_f: Optional[float] = None) -> Optional[float]:
    return valid_dewpoint_f(extract_value_f(props.get("dewpoint")), temp_f)


def station_observation_wind_speed_mph(props: dict) -> Optional[float]:
    return valid_wind_speed_mph(wind_to_mph(props.get("windSpeed")))


def station_observation_wind_direction_deg(props: dict) -> Optional[float]:
    raw = (props.get("windDirection") or {}).get("value")
    return valid_wind_direction_deg(raw)


def paged_station_observations(station_id: str, start_utc: datetime, end_utc: datetime, limit: int = 500):
    base = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": iso_z(start_utc),
        "end": iso_z(end_utc),
        "limit": limit,
    }
    url = f"{base}?{urlencode(params)}"
    seen_urls = set()

    while url and url not in seen_urls:
        seen_urls.add(url)
        j = cached_get_json(url)

        for f in j.get("features", []):
            props = (f or {}).get("properties", {}) or {}
            ts = props.get("timestamp")

            if not ts:
                continue

            try:
                t = iso_to_dt(ts)
            except Exception:
                continue

            if start_utc <= t < end_utc:
                yield t, props

        url = (j.get("pagination") or {}).get("next")


def max_from_station_observations(station_id: str, start_utc: datetime, end_utc: datetime):
    max_temp = None

    try:
        rows = paged_station_observations(station_id, start_utc, end_utc)

        for _, props in rows:
            candidates = []

            temp_f = station_observation_temp_f(props)
            if temp_f is not None:
                candidates.append(temp_f)

            raw_message = props.get("rawMessage") or props.get("textDescription") or ""
            remark_max_f = metar_remark_6h_max_f(raw_message)
            if remark_max_f is not None:
                candidates.append(remark_max_f)

            for candidate in candidates:
                max_temp = candidate if max_temp is None else max(max_temp, candidate)

    except Exception:
        return None

    return max_temp


def min_from_station_observations(station_id: str, start_utc: datetime, end_utc: datetime):
    min_temp = None

    try:
        rows = paged_station_observations(station_id, start_utc, end_utc)

        for _, props in rows:
            candidates = []

            temp_f = station_observation_temp_f(props)
            if temp_f is not None:
                candidates.append(temp_f)

            raw_message = props.get("rawMessage") or props.get("textDescription") or ""
            remark_min_f = metar_remark_6h_min_f(raw_message)
            if remark_min_f is not None:
                candidates.append(remark_min_f)

            for candidate in candidates:
                min_temp = candidate if min_temp is None else min(min_temp, candidate)

    except Exception:
        return None

    return min_temp


def station_max_last_hours(station_id: str, now_utc: datetime, hours: int):
    if hours <= 0:
        return None

    return max_from_station_observations(station_id, now_utc - timedelta(hours=hours), now_utc)


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
            "obs_validation": {
                "temp": "missing",
                "dewpoint": "missing",
                "wind_speed": "missing",
                "wind_direction": "missing",
            },
        }

    props = (j or {}).get("properties", {}) or {}
    ts = props.get("timestamp")

    try:
        obs_time = iso_to_dt(ts) if ts else None
    except Exception:
        obs_time = None

    temp_f = station_observation_temp_f(props)
    dew_f = station_observation_dewpoint_f(props, temp_f)
    wind_speed_mph = station_observation_wind_speed_mph(props)
    wind_dir = station_observation_wind_direction_deg(props)

    return {
        "obs_time_utc": obs_time.replace(microsecond=0).isoformat() if obs_time else None,
        "obs_temp_f": temp_f,
        "obs_dewpoint_f": dew_f,
        "obs_wind_dir_deg": wind_dir,
        "obs_wind_speed_mph": wind_speed_mph,
        "obs_validation": {
            "temp": "valid" if temp_f is not None else "invalid_or_missing",
            "dewpoint": "valid" if dew_f is not None else "invalid_or_missing",
            "wind_speed": "valid" if wind_speed_mph is not None else "invalid_or_missing",
            "wind_direction": "valid" if wind_dir is not None else "invalid_or_missing",
        },
    }


def get_recent_station_observations(station_id: str, now_utc: datetime, lookback_minutes: int = 60, limit: int = 200):
    if lookback_minutes <= 0:
        return []

    start_utc = now_utc - timedelta(minutes=lookback_minutes)
    rows = []

    try:
        obs_rows = paged_station_observations(station_id, start_utc, now_utc + timedelta(seconds=1), limit=limit)

        for t, props in obs_rows:
            temp_f = station_observation_temp_f(props)

            if temp_f is None:
                continue

            dew_f = station_observation_dewpoint_f(props, temp_f)
            wind_speed_mph = station_observation_wind_speed_mph(props)
            wind_dir_deg = station_observation_wind_direction_deg(props)

            rows.append({
                "t": t.replace(microsecond=0),
                "temp_f": temp_f,
                "dewpoint_f": dew_f,
                "wind_speed_mph": wind_speed_mph,
                "wind_dir_deg": wind_dir_deg,
            })
    except Exception:
        return []

    rows.sort(key=lambda r: r["t"])
    return rows


def pick_closest_station_id(point_json, lat: float, lon: float, max_candidates: int = 200):
    stations_url = point_json.get("properties", {}).get("observationStations") if point_json else None

    if not stations_url:
        return None, None

    try:
        stations_json = cached_get_json(stations_url)
    except Exception:
        return None, None

    best = None
    best_d = None

    for f in stations_json.get("features", [])[:max_candidates]:
        props = f.get("properties") or {}
        sid = props.get("stationIdentifier")
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates")

        if not sid or not coords or len(coords) < 2:
            continue

        try:
            slon, slat = float(coords[0]), float(coords[1])
        except (ValueError, TypeError):
            continue

        d = haversine_km(lat, lon, slat, slon)

        if best_d is None or d < best_d:
            best_d = d
            best = sid

    return best, best_d


def future_hourly_periods(hourly_json, now_utc: datetime, hours: int):
    periods = hourly_json.get("properties", {}).get("periods", []) if hourly_json else []
    end_utc = now_utc + timedelta(hours=hours)
    window = []

    for p in periods:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if et <= now_utc or st >= end_utc:
            continue

        window.append(p)

    return window


def window_max_from_hourly(hourly_json, now_utc: datetime, hours: int):
    if hours <= 0:
        return {
            "start_utc": None,
            "end_utc": None,
            "max_temp_f": None,
            "max_dewpoint_f": None,
            "max_wind_speed_mph": None,
            "period_count": 0,
        }

    window = future_hourly_periods(hourly_json, now_utc, hours)
    max_temp_f = None
    max_dew_f = None
    max_wind_mph = None
    start_utc = None
    end_utc = None

    for p in window:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if start_utc is None or st < start_utc:
            start_utc = st

        if end_utc is None or et > end_utc:
            end_utc = et

        temp_f = period_temp_f(p)

        if temp_f is not None:
            max_temp_f = temp_f if max_temp_f is None else max(max_temp_f, temp_f)

        dp_f = period_dewpoint_f(p, temp_f)

        if dp_f is not None:
            max_dew_f = dp_f if max_dew_f is None else max(max_dew_f, dp_f)

        wind_mph = period_wind_speed_mph(p)

        if wind_mph is not None:
            max_wind_mph = wind_mph if max_wind_mph is None else max(max_wind_mph, wind_mph)

    return {
        "start_utc": start_utc.replace(microsecond=0).isoformat() if start_utc else None,
        "end_utc": end_utc.replace(microsecond=0).isoformat() if end_utc else None,
        "max_temp_f": max_temp_f,
        "max_dewpoint_f": max_dew_f,
        "max_wind_speed_mph": max_wind_mph,
        "period_count": len(window),
    }


def current_or_next_hour_from_hourly(hourly_json, now_utc: datetime):
    periods = hourly_json.get("properties", {}).get("periods", []) if hourly_json else []
    first_future = None

    for p in periods:
        try:
            st, et = period_times(p)
        except Exception:
            continue

        if st is None or et is None:
            continue

        if st <= now_utc < et:
            return p, st

        if st >= now_utc and first_future is None:
            first_future = (p, st)

    return first_future if first_future else (None, None)


def next_hour_from_hourly(hourly_json, now_utc: datetime):
    p, st = current_or_next_hour_from_hourly(hourly_json, now_utc)

    if not p or not st:
        return {
            "next_hour_start_utc": None,
            "next_hour_temp_f": None,
            "next_hour_dewpoint_f": None,
            "next_hour_wind_speed_mph": None,
            "next_hour_wind_direction": None,
            "next_hour_source": None,
        }

    temp_f = period_temp_f(p)
    dew_f = period_dewpoint_f(p, temp_f)
    wind_mph = period_wind_speed_mph(p)

    return {
        "next_hour_start_utc": st.replace(microsecond=0).isoformat(),
        "next_hour_temp_f": temp_f,
        "next_hour_dewpoint_f": dew_f,
        "next_hour_wind_speed_mph": wind_mph,
        "next_hour_wind_direction": p.get("windDirection"),
        "next_hour_source": "forecastHourly",
    }


def temp_at_horizon_from_hourly(hourly_json, now_utc: datetime, hours_ahead: int, tz_name: str | None = None):
    if hours_ahead < 0:
        return None

    target = now_utc + timedelta(hours=hours_ahead)
    periods = hourly_json.get("properties", {}).get("periods", [])
    best_p = None
    best_delta = None

    for p in periods:
        try:
            st = iso_to_dt(p["startTime"])
        except Exception:
            continue

        if st < now_utc:
            continue

        delta = abs((st - target).total_seconds())

        if best_p is None or delta < best_delta:
            best_p = p
            best_delta = delta

    if not best_p:
        return None

    temp_f = period_temp_f(best_p)
    dew_f = period_dewpoint_f(best_p, temp_f)
    wind_mph = period_wind_speed_mph(best_p)

    st_utc = iso_to_dt(best_p["startTime"]).astimezone(timezone.utc).replace(microsecond=0)

    item = {
        "hours_ahead": hours_ahead,
        "start_utc": st_utc.isoformat(),
        "temp_f": temp_f,
        "dewpoint_f": dew_f,
        "wind_speed_mph": wind_mph,
        "source": "forecastHourly",
        "dewpoint_source": "forecastHourly.dewpoint" if dew_f is not None else None,
        "wind_speed_source": "forecastHourly.windSpeed" if wind_mph is not None else None,
    }

    if tz_name:
        local_dt = st_utc.astimezone(ZoneInfo(tz_name)).replace(microsecond=0)
        item["start_local"] = local_dt.isoformat()
        item["start_local_label"] = format_local_label(local_dt)

    return item


def parse_iso_duration(duration: str) -> timedelta:
    if not duration:
        raise ValueError("missing ISO duration")

    m = re.fullmatch(
        r"P(?:(?P<days>\d+(?:\.\d+)?)D)?(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?",
        duration,
    )

    if not m:
        raise ValueError(f"unsupported ISO duration: {duration}")

    return timedelta(
        days=float(m.group("days") or 0),
        hours=float(m.group("hours") or 0),
        minutes=float(m.group("minutes") or 0),
        seconds=float(m.group("seconds") or 0),
    )


def parse_valid_time(valid_time: str) -> tuple[datetime, datetime]:
    if not valid_time or "/" not in valid_time:
        raise ValueError(f"invalid validTime: {valid_time}")

    start_raw, end_or_duration = valid_time.split("/", 1)
    start = iso_to_dt(start_raw)

    if end_or_duration.startswith("P"):
        end = start + parse_iso_duration(end_or_duration)
    else:
        end = iso_to_dt(end_or_duration)

    return start, end


def interval_overlap_seconds(st: datetime, et: datetime, start_utc: datetime, end_utc: datetime) -> float:
    overlap_start = max(st, start_utc)
    overlap_end = min(et, end_utc)
    return max((overlap_end - overlap_start).total_seconds(), 0.0)


def grid_layer(grid_json: Optional[dict], layer_name: str) -> dict:
    return ((grid_json or {}).get("properties") or {}).get(layer_name) or {}


def normalize_grid_value(layer_name: str, value, uom: str):
    if value is None:
        return None

    if isinstance(value, (list, dict)):
        return value

    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    u = (uom or "").lower()
    name = layer_name.lower()

    if name in {
        "temperature",
        "maxtemperature",
        "mintemperature",
        "apparenttemperature",
        "heatindex",
        "windchill",
    }:
        if is_celsius_unit(u):
            return valid_temp_f(c_to_f(v))
        return valid_temp_f(v)

    if name == "dewpoint":
        if is_celsius_unit(u):
            return valid_dewpoint_f(c_to_f(v))
        return valid_dewpoint_f(v)

    if name == "windspeed":
        if "m_s-1" in u or "m/s" in u:
            return valid_wind_speed_mph(v * 2.2369362920544)
        if "km_h-1" in u or "km/h" in u:
            return valid_wind_speed_mph(v * 0.621371)
        if ":kn" in u or ":kt" in u or "knot" in u:
            return valid_wind_speed_mph(v * 1.150779)
        return valid_wind_speed_mph(v)

    if name == "windgust":
        if "m_s-1" in u or "m/s" in u:
            return valid_wind_gust_mph(v * 2.2369362920544)
        if "km_h-1" in u or "km/h" in u:
            return valid_wind_gust_mph(v * 0.621371)
        if ":kn" in u or ":kt" in u or "knot" in u:
            return valid_wind_gust_mph(v * 1.150779)
        return valid_wind_gust_mph(v)

    if name in {"quantitativeprecipitation", "snowfallamount", "iceaccumulation"}:
        if "mm" in u or "milli" in u:
            return mm_to_inches(v)
        if u.endswith(":m") or "unit:m" in u:
            return v * 39.3700787
        return v

    if name == "visibility":
        if u.endswith(":m") or "unit:m" in u:
            return meters_to_miles(v)
        return v

    if name == "ceilingheight":
        if u.endswith(":m") or "unit:m" in u:
            return meters_to_feet(v)
        return v

    if name == "pressure":
        if "pa" in u:
            return pa_to_hpa(v)
        return v

    return v


def grid_values_in_window(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime):
    layer = grid_layer(grid_json, layer_name)
    uom = (layer.get("uom") or "").lower()
    rows = []

    for item in layer.get("values", []) or []:
        try:
            st, et = parse_valid_time(item.get("validTime"))
        except Exception:
            continue

        overlap = interval_overlap_seconds(st, et, start_utc, end_utc)

        if overlap <= 0:
            continue

        raw_value = item.get("value")
        value = normalize_grid_value(layer_name, raw_value, uom)

        if value is None:
            continue

        rows.append({
            "start_utc": st,
            "end_utc": et,
            "overlap_seconds": overlap,
            "value": value,
            "raw_value": raw_value,
            "uom": uom,
        })

    return rows


def grid_numeric_values(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime):
    rows = grid_values_in_window(grid_json, layer_name, start_utc, end_utc)
    return [r for r in rows if isinstance(r.get("value"), (int, float))]


def grid_max(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime) -> Optional[float]:
    rows = grid_numeric_values(grid_json, layer_name, start_utc, end_utc)

    if not rows:
        return None

    return max(float(r["value"]) for r in rows)


def grid_min(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime) -> Optional[float]:
    rows = grid_numeric_values(grid_json, layer_name, start_utc, end_utc)

    if not rows:
        return None

    return min(float(r["value"]) for r in rows)


def grid_weighted_avg(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime) -> Optional[float]:
    rows = grid_numeric_values(grid_json, layer_name, start_utc, end_utc)
    total_weight = sum(float(r["overlap_seconds"]) for r in rows)

    if total_weight <= 0:
        return None

    return sum(float(r["value"]) * float(r["overlap_seconds"]) for r in rows) / total_weight


def grid_interval_sum_prorated(grid_json: Optional[dict], layer_name: str, start_utc: datetime, end_utc: datetime) -> Optional[float]:
    rows = grid_numeric_values(grid_json, layer_name, start_utc, end_utc)

    if not rows:
        return None

    total = 0.0

    for r in rows:
        full_seconds = max((r["end_utc"] - r["start_utc"]).total_seconds(), 1.0)
        fraction = clamp(float(r["overlap_seconds"]) / full_seconds, 0.0, 1.0)
        total += float(r["value"]) * fraction

    return total


def grid_dominant_weather(grid_json: Optional[dict], start_utc: datetime, end_utc: datetime, limit: int = 5):
    rows = grid_values_in_window(grid_json, "weather", start_utc, end_utc)
    counts: dict[str, float] = {}

    for r in rows:
        value = r.get("value")

        if not isinstance(value, list):
            continue

        for item in value:
            if not isinstance(item, dict):
                continue

            weather = item.get("weather") or item.get("coverage") or item.get("intensity")

            if not weather:
                continue

            counts[str(weather)] = counts.get(str(weather), 0.0) + float(r.get("overlap_seconds") or 0)

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:limit]]


def grid_hazards(grid_json: Optional[dict], start_utc: datetime, end_utc: datetime, limit: int = 10):
    rows = grid_values_in_window(grid_json, "hazards", start_utc, end_utc)
    hazards = []
    seen = set()

    for r in rows:
        value = r.get("value")

        if not isinstance(value, list):
            continue

        for item in value:
            if not isinstance(item, dict):
                continue

            phenomenon = item.get("phenomenon")
            significance = item.get("significance")
            event_number = item.get("event_number") or item.get("eventNumber")
            key = (phenomenon, significance, event_number)

            if key in seen:
                continue

            seen.add(key)
            hazards.append({
                "phenomenon": phenomenon,
                "significance": significance,
                "event_number": event_number,
            })

            if len(hazards) >= limit:
                return hazards

    return hazards


def grid_window_summary(grid_json: Optional[dict], start_utc: datetime, end_utc: datetime) -> dict:
    if not grid_json:
        return {
            "grid_available": False,
            "grid_period_count": 0,
        }

    temp_rows = grid_numeric_values(grid_json, "temperature", start_utc, end_utc)

    return {
        "grid_available": True,
        "grid_update_time": ((grid_json.get("properties") or {}).get("updateTime")),
        "grid_period_count": len(temp_rows),
        "max_temp_f": grid_max(grid_json, "temperature", start_utc, end_utc),
        "min_temp_f": grid_min(grid_json, "temperature", start_utc, end_utc),
        "max_dewpoint_f": grid_max(grid_json, "dewpoint", start_utc, end_utc),
        "min_dewpoint_f": grid_min(grid_json, "dewpoint", start_utc, end_utc),
        "max_apparent_temp_f": grid_max(grid_json, "apparentTemperature", start_utc, end_utc),
        "max_heat_index_f": grid_max(grid_json, "heatIndex", start_utc, end_utc),
        "min_wind_chill_f": grid_min(grid_json, "windChill", start_utc, end_utc),
        "max_relative_humidity_pct": grid_max(grid_json, "relativeHumidity", start_utc, end_utc),
        "min_relative_humidity_pct": grid_min(grid_json, "relativeHumidity", start_utc, end_utc),
        "avg_sky_cover_pct": grid_weighted_avg(grid_json, "skyCover", start_utc, end_utc),
        "max_wind_speed_mph": grid_max(grid_json, "windSpeed", start_utc, end_utc),
        "max_wind_gust_mph": grid_max(grid_json, "windGust", start_utc, end_utc),
        "avg_wind_direction_deg": grid_weighted_avg(grid_json, "windDirection", start_utc, end_utc),
        "max_pop_pct": grid_max(grid_json, "probabilityOfPrecipitation", start_utc, end_utc),
        "total_qpf_in": grid_interval_sum_prorated(grid_json, "quantitativePrecipitation", start_utc, end_utc),
        "total_snow_in": grid_interval_sum_prorated(grid_json, "snowfallAmount", start_utc, end_utc),
        "total_ice_in": grid_interval_sum_prorated(grid_json, "iceAccumulation", start_utc, end_utc),
        "min_visibility_mi": grid_min(grid_json, "visibility", start_utc, end_utc),
        "min_ceiling_ft": grid_min(grid_json, "ceilingHeight", start_utc, end_utc),
        "avg_pressure_hpa": grid_weighted_avg(grid_json, "pressure", start_utc, end_utc),
        "dominant_weather": grid_dominant_weather(grid_json, start_utc, end_utc),
        "grid_hazards": grid_hazards(grid_json, start_utc, end_utc),
        "precip_amount_method": "interval_prorated",
        "source": "forecastGridData",
    }


def grid_day_summary(grid_json: Optional[dict], start_utc: datetime, end_utc: datetime) -> dict:
    summary = grid_window_summary(grid_json, start_utc, end_utc)

    if not grid_json:
        return summary

    official_max = grid_max(grid_json, "maxTemperature", start_utc, end_utc)
    official_min = grid_min(grid_json, "minTemperature", start_utc, end_utc)

    summary.update({
        "official_max_temp_f": official_max,
        "official_min_temp_f": official_min,
        "best_max_temp_f": official_max if official_max is not None else summary.get("max_temp_f"),
        "best_min_temp_f": official_min if official_min is not None else summary.get("min_temp_f"),
        "best_max_temp_source": "forecastGridData.maxTemperature" if official_max is not None else "forecastGridData.temperature",
        "best_min_temp_source": "forecastGridData.minTemperature" if official_min is not None else "forecastGridData.temperature",
    })

    return summary


def current_or_next_grid_value(grid_json: Optional[dict], layer_name: str, now_utc: datetime):
    if not grid_json:
        return None, None, None

    rows = grid_values_in_window(grid_json, layer_name, now_utc, now_utc + timedelta(days=7))

    if not rows:
        return None, None, None

    rows.sort(key=lambda r: r["start_utc"])

    for r in rows:
        if r["start_utc"] <= now_utc < r["end_utc"]:
            return r.get("value"), r["start_utc"], r["end_utc"]

    r = rows[0]
    return r.get("value"), r["start_utc"], r["end_utc"]


def grid_next_hour(grid_json: Optional[dict], now_utc: datetime) -> dict:
    temp, st, et = current_or_next_grid_value(grid_json, "temperature", now_utc)
    dew, _, _ = current_or_next_grid_value(grid_json, "dewpoint", now_utc)
    wind, _, _ = current_or_next_grid_value(grid_json, "windSpeed", now_utc)
    gust, _, _ = current_or_next_grid_value(grid_json, "windGust", now_utc)
    app, _, _ = current_or_next_grid_value(grid_json, "apparentTemperature", now_utc)
    heat, _, _ = current_or_next_grid_value(grid_json, "heatIndex", now_utc)
    chill, _, _ = current_or_next_grid_value(grid_json, "windChill", now_utc)
    pop, _, _ = current_or_next_grid_value(grid_json, "probabilityOfPrecipitation", now_utc)
    sky, _, _ = current_or_next_grid_value(grid_json, "skyCover", now_utc)
    visibility, _, _ = current_or_next_grid_value(grid_json, "visibility", now_utc)

    temp_f = valid_temp_f(temp)
    dew_f = valid_dewpoint_f(dew, temp_f)
    wind_mph = valid_wind_speed_mph(wind)
    gust_mph = valid_wind_gust_mph(gust, wind_mph)

    return {
        "grid_next_hour_start_utc": st.replace(microsecond=0).isoformat() if st else None,
        "grid_next_hour_end_utc": et.replace(microsecond=0).isoformat() if et else None,
        "grid_next_hour_temp_f": temp_f,
        "grid_next_hour_dewpoint_f": dew_f,
        "grid_next_hour_wind_speed_mph": wind_mph,
        "grid_next_hour_wind_gust_mph": gust_mph,
        "grid_next_hour_apparent_temp_f": valid_temp_f(app),
        "grid_next_hour_heat_index_f": valid_temp_f(heat),
        "grid_next_hour_wind_chill_f": valid_temp_f(chill),
        "grid_next_hour_pop_pct": pop if isinstance(pop, (int, float)) and 0 <= pop <= 100 else None,
        "grid_next_hour_sky_cover_pct": sky if isinstance(sky, (int, float)) and 0 <= sky <= 100 else None,
        "grid_next_hour_visibility_mi": visibility if isinstance(visibility, (int, float)) and visibility >= 0 else None,
        "grid_next_hour_source": "forecastGridData",
    }


def temp_at_horizon_from_grid(grid_json: Optional[dict], now_utc: datetime, hours_ahead: int, tz_name: str | None = None):
    if not grid_json or hours_ahead < 0:
        return None

    target = now_utc + timedelta(hours=hours_ahead)

    temp_rows = grid_numeric_values(grid_json, "temperature", now_utc, now_utc + timedelta(days=7))
    dew_rows = grid_numeric_values(grid_json, "dewpoint", now_utc, now_utc + timedelta(days=7))
    wind_rows = grid_numeric_values(grid_json, "windSpeed", now_utc, now_utc + timedelta(days=7))
    gust_rows = grid_numeric_values(grid_json, "windGust", now_utc, now_utc + timedelta(days=7))

    if not temp_rows:
        return None

    def pick_row(rows):
        best = None
        best_delta = None

        for r in rows:
            st = r["start_utc"]
            et = r["end_utc"]

            if st <= target < et:
                return r

            midpoint = st + (et - st) / 2
            delta = abs((midpoint - target).total_seconds())

            if best is None or delta < best_delta:
                best = r
                best_delta = delta

        return best

    temp_row = pick_row(temp_rows)

    if not temp_row:
        return None

    dew_row = pick_row(dew_rows)
    wind_row = pick_row(wind_rows)
    gust_row = pick_row(gust_rows)

    st_utc = temp_row["start_utc"].astimezone(timezone.utc).replace(microsecond=0)

    temp_f = valid_temp_f(temp_row.get("value"))
    dew_f = valid_dewpoint_f(dew_row.get("value") if dew_row else None, temp_f)
    wind_mph = valid_wind_speed_mph(wind_row.get("value") if wind_row else None)
    gust_mph = valid_wind_gust_mph(gust_row.get("value") if gust_row else None, wind_mph)

    item = {
        "hours_ahead": hours_ahead,
        "start_utc": st_utc.isoformat(),
        "temp_f": temp_f,
        "dewpoint_f": dew_f,
        "wind_speed_mph": wind_mph,
        "wind_gust_mph": gust_mph,
        "source": "forecastGridData.temperature",
        "dewpoint_source": "forecastGridData.dewpoint" if dew_f is not None else None,
        "wind_speed_source": "forecastGridData.windSpeed" if wind_mph is not None else None,
    }

    if tz_name:
        local_dt = st_utc.astimezone(ZoneInfo(tz_name)).replace(microsecond=0)
        item["start_local"] = local_dt.isoformat()
        item["start_local_label"] = format_local_label(local_dt)

    return item


def fetch_active_alerts_for_point(lat: float, lon: float):
    url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"

    try:
        j = cached_get_json(url, force_refresh=True)
    except Exception:
        return {
            "active_alerts": [],
            "active_alerts_error": "alerts_fetch_failed",
        }

    alerts = []

    for feature in j.get("features", []) or []:
        props = feature.get("properties") or {}
        alerts.append({
            "event": props.get("event"),
            "headline": props.get("headline"),
            "severity": props.get("severity"),
            "urgency": props.get("urgency"),
            "certainty": props.get("certainty"),
            "effective": props.get("effective"),
            "expires": props.get("expires"),
            "ends": props.get("ends"),
            "status": props.get("status"),
            "message_type": props.get("messageType"),
            "area_desc": props.get("areaDesc"),
        })

    return {
        "active_alerts": alerts,
        "active_alert_count": len(alerts),
    }


def confidence_from_source(source: Optional[str], age_minutes: Optional[float] = None, station_distance_km: Optional[float] = None):
    if source in {
        "forecastGridData",
        "forecastGridData.temperature",
        "forecastGridData.maxTemperature",
        "forecastGridData.minTemperature",
        "forecastGridData.dewpoint",
        "forecastGridData.windSpeed",
        "forecastGridData.windGust",
    }:
        return "high"

    if source in {"station_obs", "station_observation"}:
        if age_minutes is not None and station_distance_km is not None and age_minutes <= 20 and station_distance_km <= 10:
            return "high"
        if age_minutes is not None and age_minutes <= 35:
            return "medium"
        return "low"

    if source in {
        "forecastHourly",
        "hourly_forecast",
        "forecastHourly.dewpoint",
        "forecastHourly.windSpeed",
        "forecastHourly.windDirection",
    }:
        return "medium"

    if source in {"nowcast", "nowcast_15m", "nowcast_30m", "best_next_hour"}:
        return "medium"

    return "low"