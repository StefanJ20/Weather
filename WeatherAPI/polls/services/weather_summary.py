from datetime import datetime, date, timedelta, timezone
import math
from typing import Optional
from zoneinfo import ZoneInfo

from .calibration import (
    get_station_bias,
    get_station_calibration,
    record_bias_sample,
)
from .cache import cached_get_json
from .weather_math import (
    TIMEZONE,
    c_to_f,
    clamp,
    confidence_from_source,
    day_range_local,
    fetch_active_alerts_for_point,
    get_latest_station_observation,
    get_recent_station_observations,
    grid_day_summary,
    grid_next_hour,
    grid_window_summary,
    iso_to_dt,
    max_from_forecast_periods,
    max_from_hourly,
    max_from_station_observations,
    min_from_hourly,
    min_from_station_observations,
    next_hour_from_hourly,
    period_temp_f,
    pick_closest_station_id,
    station_max_last_hours,
    temp_at_horizon_from_grid,
    temp_at_horizon_from_hourly,
    tz_for_latlon,
    window_max_from_hourly,
    metric_confidence,
    valid_dewpoint_f,
    valid_wind_direction_deg,
    valid_wind_speed_mph,
)

def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def safe_float(value) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def hourly_temp_f(period: dict) -> Optional[float]:
    return period_temp_f(period)


def regression_from_recent_obs(recent_obs, now_utc: datetime, max_points: int = 12):
    if not recent_obs or len(recent_obs) < 3:
        return None

    recent = recent_obs[-max_points:]
    xs = []
    ys = []

    for r in recent:
        t = r.get("t")
        temp_f = safe_float(r.get("temp_f"))

        if t is None or temp_f is None:
            continue

        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        else:
            t = t.astimezone(timezone.utc)

        xs.append((t - now_utc).total_seconds())
        ys.append(temp_f)

    if len(xs) < 3:
        return None

    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)

    if denom == 0:
        return None

    slope = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n)) / denom
    intercept = y_mean - slope * x_mean

    return intercept, slope, n


def linear_trend_nowcast_temps(recent_obs, now_utc: datetime, horizons_minutes=(15, 30, 45, 60)):
    model = regression_from_recent_obs(recent_obs, now_utc)

    if not model:
        return None

    intercept, slope, _ = model
    out = []

    for minutes in horizons_minutes:
        seconds_ahead = minutes * 60.0
        out.append({
            "minutes_ahead": minutes,
            "temp_f_trend": intercept + slope * seconds_ahead,
        })

    return out


def pick_nowcast_temp(nowcast_points, minutes_ahead: int) -> Optional[float]:
    if not isinstance(nowcast_points, list):
        return None

    best = None
    best_delta = None

    for p in nowcast_points:
        m = p.get("minutes_ahead")
        temp_f = safe_float(p.get("temp_f"))

        if m is None or temp_f is None:
            continue

        delta = abs(float(m) - float(minutes_ahead))

        if best is None or delta < best_delta:
            best = temp_f
            best_delta = delta

    return best


def build_nowcast_layer(
    station_id: str,
    now_utc: datetime,
    forecast_next_hour_temp_f: Optional[float],
    lookback_minutes: int = 90,
    freshness_minutes: int = 40,
):
    if not station_id:
        return {"nowcast_error": "no_station_id"}

    recent = get_recent_station_observations(
        station_id,
        now_utc,
        lookback_minutes=lookback_minutes,
    )

    if not recent:
        return {"nowcast_error": "no_recent_obs"}

    latest_t = recent[-1]["t"]

    if latest_t.tzinfo is None:
        latest_t = latest_t.replace(tzinfo=timezone.utc)
    else:
        latest_t = latest_t.astimezone(timezone.utc)

    age_sec = max((now_utc - latest_t).total_seconds(), 0.0)
    forecast_next_hour_temp_f = safe_float(forecast_next_hour_temp_f)

    if age_sec > freshness_minutes * 60:
        if forecast_next_hour_temp_f is None:
            return {"nowcast_error": f"stale_obs ({round(age_sec / 60, 1)} min)"}

        blended = []

        for minutes in (15, 30, 45, 60):
            blended.append({
                "minutes_ahead": minutes,
                "temp_f": round(forecast_next_hour_temp_f, 2),
                "temp_f_trend": None,
                "weight_trend": 0.0,
            })

        return {
            "nowcast_method": "forecast_fallback_stale_obs",
            "nowcast_generated_utc": now_utc.isoformat(),
            "nowcast_obs_count": len(recent),
            "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
            "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
            "nowcast_next_0_60m": blended,
        }

    trend_preds = linear_trend_nowcast_temps(recent, now_utc)

    if not trend_preds:
        fallback_temp = forecast_next_hour_temp_f
        method = "forecast_fallback_no_trend"
        trend_value = None
        weight_trend = 0.0

        if fallback_temp is None:
            fallback_temp = safe_float(recent[-1].get("temp_f"))
            method = "repeat_latest_obs"
            trend_value = fallback_temp
            weight_trend = 1.0

        if fallback_temp is None:
            return {"nowcast_error": "no_usable_temp"}

        blended = []

        for minutes in (15, 30, 45, 60):
            blended.append({
                "minutes_ahead": minutes,
                "temp_f": round(fallback_temp, 2),
                "temp_f_trend": round(trend_value, 2) if trend_value is not None else None,
                "weight_trend": weight_trend,
            })

        return {
            "nowcast_method": method,
            "nowcast_generated_utc": now_utc.isoformat(),
            "nowcast_obs_count": len(recent),
            "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
            "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
            "nowcast_next_0_60m": blended,
        }

    blended = []
    method = "trend_only"

    for p in trend_preds:
        minutes = p["minutes_ahead"]
        temp_trend = p["temp_f_trend"]
        w_trend = clamp(0.95 - (minutes / 120.0), 0.45, 0.90)

        if forecast_next_hour_temp_f is None:
            temp_best = temp_trend
            method = "trend_only"
        else:
            temp_mix = w_trend * temp_trend + (1.0 - w_trend) * forecast_next_hour_temp_f
            cap = 3.0 if minutes <= 30 else 2.0
            temp_best = clamp(
                temp_mix,
                forecast_next_hour_temp_f - cap,
                forecast_next_hour_temp_f + cap,
            )
            method = "trend_blend_capped"

        blended.append({
            "minutes_ahead": minutes,
            "temp_f": round(float(temp_best), 2),
            "temp_f_trend": round(float(temp_trend), 2),
            "weight_trend": round(float(w_trend), 2),
        })

    return {
        "nowcast_method": method,
        "nowcast_generated_utc": now_utc.isoformat(),
        "nowcast_obs_count": len(recent),
        "nowcast_latest_obs_utc": latest_t.replace(microsecond=0).isoformat(),
        "nowcast_obs_age_minutes": round(age_sec / 60.0, 1),
        "nowcast_next_0_60m": blended,
    }


def estimate_remaining_day_low_model(
    station_id: Optional[str],
    hourly_json,
    now_utc: datetime,
    end_utc_full: datetime,
    lookback_minutes: int = 180,
):
    if end_utc_full <= now_utc:
        return {"pred_low_remaining_f": None, "pred_points": []}

    recent_obs = []

    if station_id:
        try:
            recent_obs = get_recent_station_observations(
                station_id,
                now_utc,
                lookback_minutes=lookback_minutes,
            )
        except Exception:
            recent_obs = []

    periods = hourly_json.get("properties", {}).get("periods", []) if hourly_json else []
    future_periods = []

    for p in periods:
        try:
            st = iso_to_dt(p["startTime"])
        except Exception:
            continue

        if now_utc <= st < end_utc_full:
            future_periods.append(p)

    if not future_periods:
        return {"pred_low_remaining_f": None, "pred_points": []}

    model = regression_from_recent_obs(recent_obs, now_utc, max_points=12)
    intercept = slope = None

    if model:
        intercept, slope, _ = model

    pred_points = []
    pred_low = None

    for p in future_periods:
        try:
            st = iso_to_dt(p["startTime"])
        except Exception:
            continue

        secs_ahead = (st - now_utc).total_seconds()
        hours_ahead = secs_ahead / 3600.0
        fc_f = hourly_temp_f(p)
        trend_f = intercept + slope * secs_ahead if intercept is not None and slope is not None else None

        if trend_f is None and fc_f is None:
            continue

        if trend_f is None:
            best = fc_f
            w_trend = 0.0
            method = "forecast_only"
        elif fc_f is None:
            best = trend_f
            w_trend = 1.0
            method = "trend_only"
        else:
            w_trend = clamp(0.85 - (hours_ahead / 12.0) * 0.45, 0.25, 0.85)
            mix = w_trend * trend_f + (1.0 - w_trend) * fc_f
            cap = 4.0 if hours_ahead <= 3 else 3.0
            best = clamp(mix, fc_f - cap, fc_f + cap)
            method = "blend_capped"

        pred_points.append({
            "start_utc": st.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            "hours_ahead": round(hours_ahead, 2),
            "temp_f": round(float(best), 2) if best is not None else None,
            "temp_f_trend": round(float(trend_f), 2) if trend_f is not None else None,
            "temp_f_forecast": round(float(fc_f), 2) if fc_f is not None else None,
            "weight_trend": round(float(w_trend), 2),
            "method": method,
        })

        if best is not None:
            pred_low = best if pred_low is None else min(pred_low, best)

    return {
        "pred_low_remaining_f": pred_low,
        "pred_points": pred_points,
    }


def apply_forecast_calibration(results: dict, station_id: str | None) -> None:
    raw_fc = results.get("forecasted_max_f")

    if not station_id or raw_fc is None:
        return

    cal = get_station_calibration(station_id, lookback_days=365)
    bias = get_station_bias(station_id, lookback_days=120)
    corrected_fc = None

    if cal and int(cal.get("sample_count", 0)) >= 60:
        a = float(cal["a"])
        b = float(cal["b"])
        corrected_fc = a + b * float(raw_fc)
        results["calibration_a"] = a
        results["calibration_b"] = b
        results["calibration_sample_count"] = int(cal["sample_count"])
        results["calibration_r2"] = float(cal.get("r2", 0.0))
        results["calibration_rmse_f"] = float(cal.get("rmse_f", 0.0))
        results["forecasted_max_calibrated_f"] = corrected_fc
        results["calibration_method"] = "linear"
    elif bias and int(bias.get("sample_count", 0)) >= 30:
        bias_mean = float(bias["mean_error_f"])
        corrected_fc = float(raw_fc) + bias_mean
        results["bias_mean_f"] = bias_mean
        results["bias_sample_count"] = int(bias["sample_count"])
        results["forecasted_max_bias_corrected_f"] = corrected_fc
        results["calibration_method"] = "mean_bias"

    if corrected_fc is not None:
        cal_candidates = [results.get("station_max_f"), corrected_fc]
        cal_candidates = [x for x in cal_candidates if x is not None]
        results["overall_max_calibrated_f"] = max(cal_candidates) if cal_candidates else None


def set_empty_hourly_results(results: dict) -> None:
    results["hourly_max_f"] = None
    results["hourly_min_f"] = None
    results["forecasted_max_f"] = None
    results["forecast_source"] = None
    results["best_temp_next_1_24h"] = []
    results["next_hour_start_utc"] = None
    results["next_hour_temp_f"] = None
    results["next_hour_dewpoint_f"] = None
    results["next_hour_wind_speed_mph"] = None
    results["next_hour_wind_direction"] = None
    results["station_running_min_f"] = None
    results["station_min_error"] = results.get("station_min_error")
    results["pred_low_remaining_f"] = None
    results["pred_low_remaining_points"] = []
    results["best_estimate_low_f"] = None
    results["overall_min_f"] = None


def set_empty_window_results(results: dict) -> None:
    for hours in (3, 6):
        prefix = f"next_{hours}h"
        results[f"{prefix}_start_utc"] = None
        results[f"{prefix}_end_utc"] = None
        results[f"{prefix}_max_temp_f"] = None
        results[f"{prefix}_max_dewpoint_f"] = None
        results[f"{prefix}_max_wind_speed_mph"] = None
        results[f"{prefix}_max_wind_gust_mph"] = None
        results[f"{prefix}_max_apparent_temp_f"] = None
        results[f"{prefix}_max_heat_index_f"] = None
        results[f"{prefix}_min_wind_chill_f"] = None
        results[f"{prefix}_max_pop_pct"] = None
        results[f"{prefix}_total_qpf_in"] = None
        results[f"{prefix}_total_snow_in"] = None
        results[f"{prefix}_total_ice_in"] = None
        results[f"{prefix}_avg_sky_cover_pct"] = None
        results[f"{prefix}_min_visibility_mi"] = None
        results[f"{prefix}_period_count"] = 0

def lower_confidence(confidence: str) -> str:
    if confidence == "high":
        return "medium"
    if confidence == "medium":
        return "low"
    return "low"


def confidence_label_from_score(score: float | None) -> str:
    if score is None:
        return "low"

    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def confidence_score_value(label: str | None) -> float:
    label = str(label or "").lower()

    if label == "high":
        return 1.0
    if label == "medium":
        return 0.65
    if label == "low":
        return 0.30
    return 0.45


def score_observation_freshness(age_minutes) -> tuple[float, str]:
    age = safe_float(age_minutes)

    if age is None:
        return 0.45, "no observation age available"
    if age <= 7:
        return 1.0, f"fresh observation age {round(age, 1)} min"
    if age <= 15:
        return 0.85, f"usable observation age {round(age, 1)} min"
    if age <= 25:
        return 0.65, f"aging observation age {round(age, 1)} min"
    if age <= 40:
        return 0.45, f"stale-leaning observation age {round(age, 1)} min"
    return 0.20, f"stale observation age {round(age, 1)} min"


def score_station_distance(distance_km) -> tuple[float, str]:
    d = safe_float(distance_km)

    if d is None:
        return 0.60, "station distance unavailable"
    if d <= 10:
        return 1.0, f"near station distance {round(d, 1)} km"
    if d <= 25:
        return 0.75, f"moderate station distance {round(d, 1)} km"
    if d <= 50:
        return 0.50, f"distant station distance {round(d, 1)} km"
    return 0.25, f"far station distance {round(d, 1)} km"


def score_temperature_source_agreement(results: dict) -> tuple[float, str, dict]:
    sources = []

    def add(name, value):
        x = safe_float(value)
        if x is not None:
            sources.append((name, x))

    add("current", results.get("current_temp_f"))
    add("station_obs", results.get("obs_temp_f"))
    add("grid_next_hour", results.get("grid_next_hour_temp_f"))
    add("hourly_next_hour", results.get("hourly_next_hour_temp_f") or results.get("next_hour_temp_f"))
    add("nowcast_60m", pick_nowcast_temp(results.get("nowcast_next_0_60m"), 60))

    unique = []
    seen = set()
    for name, value in sources:
        key = (name, round(value, 2))
        if key not in seen:
            seen.add(key)
            unique.append((name, value))

    if len(unique) < 2:
        return 0.55, "not enough independent temperature sources to compare", {"source_count": len(unique)}

    values = [v for _, v in unique]
    spread = max(values) - min(values)

    if spread <= 1.0:
        score = 1.0
        note = f"temperature sources agree within {round(spread, 2)}F"
    elif spread <= 1.5:
        score = 0.85
        note = f"temperature sources mostly agree within {round(spread, 2)}F"
    elif spread <= 2.5:
        score = 0.65
        note = f"temperature sources have moderate spread {round(spread, 2)}F"
    elif spread <= 4.0:
        score = 0.45
        note = f"temperature sources disagree by {round(spread, 2)}F"
    else:
        score = 0.20
        note = f"temperature sources strongly disagree by {round(spread, 2)}F"

    return score, note, {
        "source_count": len(unique),
        "spread_f": round(spread, 2),
        "sources": [{"name": name, "temp_f": round(value, 2)} for name, value in unique],
    }


def score_nowcast_grid_handoff(results: dict) -> tuple[float, str, dict]:
    nowcast_60 = pick_nowcast_temp(results.get("nowcast_next_0_60m"), 60)
    grid_next = safe_float(results.get("grid_next_hour_temp_f"))

    if nowcast_60 is None or grid_next is None:
        return 0.60, "handoff gap unavailable", {"handoff_gap_f": None}

    gap = abs(nowcast_60 - grid_next)

    if gap <= 1.0:
        score = 1.0
        note = f"nowcast/grid handoff is smooth: {round(gap, 2)}F gap"
    elif gap <= 2.5:
        score = 0.65
        note = f"nowcast/grid handoff has moderate gap: {round(gap, 2)}F"
    else:
        score = 0.30
        note = f"nowcast/grid handoff has large gap: {round(gap, 2)}F"

    return score, note, {"handoff_gap_f": round(gap, 2)}


def score_bias_correction_quality(results: dict) -> tuple[float, str, dict]:
    horizons = results.get("best_temp_next_1_24h") or []
    corrections = []

    for item in horizons:
        h = safe_float(item.get("hours_ahead"))
        if h is None or h > 6:
            continue

        applied = safe_float(item.get("applied_bias_correction_f"))
        capped = safe_float(item.get("display_max_cap_correction_f"))
        total = abs(applied or 0.0) + abs(capped or 0.0)
        corrections.append(total)

    max_correction = max(corrections) if corrections else 0.0

    if max_correction <= 0.75:
        score = 1.0
        note = f"small correction magnitude: {round(max_correction, 2)}F"
    elif max_correction <= 1.5:
        score = 0.80
        note = f"reasonable correction magnitude: {round(max_correction, 2)}F"
    elif max_correction <= 3.0:
        score = 0.55
        note = f"large correction magnitude: {round(max_correction, 2)}F"
    else:
        score = 0.25
        note = f"very large correction magnitude: {round(max_correction, 2)}F"

    return score, note, {"max_short_horizon_correction_f": round(max_correction, 2)}


def score_historical_calibration(results: dict) -> tuple[float, str, dict]:
    rmse = safe_float(results.get("calibration_rmse_f"))
    sample_count = results.get("calibration_sample_count") or results.get("bias_sample_count")

    try:
        sample_count = int(sample_count or 0)
    except (TypeError, ValueError):
        sample_count = 0

    if rmse is None:
        if sample_count >= 30:
            return 0.65, f"mean bias history available with {sample_count} samples", {"sample_count": sample_count, "rmse_f": None}
        return 0.55, "no recent forecast-error validation available", {"sample_count": sample_count, "rmse_f": None}

    if rmse <= 1.25:
        score = 1.0
        note = f"historical calibration RMSE is strong: {round(rmse, 2)}F"
    elif rmse <= 2.0:
        score = 0.75
        note = f"historical calibration RMSE is usable: {round(rmse, 2)}F"
    elif rmse <= 3.0:
        score = 0.50
        note = f"historical calibration RMSE is weak: {round(rmse, 2)}F"
    else:
        score = 0.25
        note = f"historical calibration RMSE is poor: {round(rmse, 2)}F"

    return score, note, {"sample_count": sample_count, "rmse_f": round(rmse, 2)}


def apply_prediction_confidence_layer(results: dict) -> None:
    obs_score, obs_note = score_observation_freshness(
        results.get("current_obs_age_minutes") or results.get("obs_age_minutes")
    )
    distance_score, distance_note = score_station_distance(results.get("station_distance_km"))
    agreement_score, agreement_note, agreement_meta = score_temperature_source_agreement(results)
    handoff_score, handoff_note, handoff_meta = score_nowcast_grid_handoff(results)
    correction_score, correction_note, correction_meta = score_bias_correction_quality(results)
    history_score, history_note, history_meta = score_historical_calibration(results)

    components = {
        "observation_freshness": {"score": round(obs_score, 3), "label": confidence_label_from_score(obs_score), "note": obs_note},
        "station_distance": {"score": round(distance_score, 3), "label": confidence_label_from_score(distance_score), "note": distance_note},
        "source_agreement": {"score": round(agreement_score, 3), "label": confidence_label_from_score(agreement_score), "note": agreement_note, **agreement_meta},
        "nowcast_grid_handoff": {"score": round(handoff_score, 3), "label": confidence_label_from_score(handoff_score), "note": handoff_note, **handoff_meta},
        "bias_correction_quality": {"score": round(correction_score, 3), "label": confidence_label_from_score(correction_score), "note": correction_note, **correction_meta},
        "historical_validation": {"score": round(history_score, 3), "label": confidence_label_from_score(history_score), "note": history_note, **history_meta},
    }

    # Weighted instead of strict minimum: one missing/neutral component should not
    # permanently block HIGH, but bad source disagreement or stale observations still matter.
    weights = {
        "observation_freshness": 0.24,
        "source_agreement": 0.24,
        "nowcast_grid_handoff": 0.18,
        "bias_correction_quality": 0.16,
        "historical_validation": 0.10,
        "station_distance": 0.08,
    }
    final_score = sum(components[k]["score"] * weights[k] for k in weights)
    final_label = confidence_label_from_score(final_score)

    hard_limits = []
    if obs_score < 0.45:
        hard_limits.append("stale observations")
    if agreement_score < 0.45:
        hard_limits.append("source disagreement")
    if handoff_score < 0.45:
        hard_limits.append("nowcast/grid handoff gap")
    if correction_score < 0.45:
        hard_limits.append("large correction")

    if hard_limits and final_label == "high":
        final_label = "medium"
    if len(hard_limits) >= 2:
        final_label = "low"

    reason_parts = [
        components["observation_freshness"]["note"],
        components["source_agreement"]["note"],
        components["nowcast_grid_handoff"]["note"],
        components["bias_correction_quality"]["note"],
        components["historical_validation"]["note"],
    ]

    results["prediction_confidence_score"] = round(final_score, 3)
    results["prediction_confidence"] = final_label
    results["prediction_confidence_components"] = components
    results["prediction_confidence_hard_limits"] = hard_limits
    results["prediction_confidence_reason"] = "; ".join(reason_parts)

    # Backward-compatible aliases for UIs that already display these card fields.
    for hours in (3, 6):
        results[f"best_next_{hours}h_max_temp_confidence"] = final_label

    results["best_next_hour_temp_confidence"] = final_label
    results["overall_max_confidence"] = final_label


def horizon_temp_confidence(
    method,
    raw_temp,
    best_temp,
    dewpoint_f=None,
    wind_speed_mph=None,
):
    method = str(method or "").lower()

    # This is prediction confidence, not raw-source confidence.
    # Raw grid can be high, but rejected/capped/smoothed adjustments lower trust.
    if "bias_rejected" in method:
        confidence = "low"
    elif method.startswith("raw_grid"):
        confidence = "high"
    elif "nowcast" in method:
        confidence = "medium"
    elif "grid_bias_corrected" in method:
        confidence = "medium"
    else:
        confidence = "low"

    if "capped" in method:
        confidence = lower_confidence(confidence)

    if "smoothed" in method:
        confidence = lower_confidence(confidence)

    temp = safe_float(best_temp)
    raw = safe_float(raw_temp)
    dew = safe_float(dewpoint_f)
    wind = safe_float(wind_speed_mph)

    if temp is not None and raw is not None:
        correction = abs(temp - raw)

        if correction >= 3.0:
            confidence = "low"
        elif correction >= 1.5:
            confidence = lower_confidence(confidence)

    if temp is not None and dew is not None:
        spread = temp - dew

        if dew > temp + 2.0:
            confidence = "low"
        elif spread <= 3.0:
            confidence = lower_confidence(confidence)

    if wind is not None:
        if wind >= 25.0:
            confidence = "low"
        elif wind >= 15.0:
            confidence = lower_confidence(confidence)

    return confidence

def corrected_horizon_temps(
    raw_horizons,
    nowcast_points,
    current_temp_f,
    raw_current_grid_temp_f,
    display_max_f=None,
    max_correction_hours: int = 6,
    max_bias_correction_f: float = 1.5,
    reject_bias_at_f: float = 5.0,
    max_above_display_max_f: float = 0.25,
):
    if not isinstance(raw_horizons, list):
        return []

    current_temp_f = safe_float(current_temp_f)
    raw_current_grid_temp_f = safe_float(raw_current_grid_temp_f)
    display_max_f = safe_float(display_max_f)

    bias_now = None

    if current_temp_f is not None and raw_current_grid_temp_f is not None:
        bias_now = current_temp_f - raw_current_grid_temp_f

    corrected = []
    previous_best = None

    for item in raw_horizons:
        h = item.get("hours_ahead")
        raw_temp = safe_float(item.get("temp_f"))

        if h is None or raw_temp is None:
            corrected.append(item)
            continue

        h_float = float(h)
        nowcast_temp = pick_nowcast_temp(nowcast_points, int(round(h_float * 60)))

        requested_bias_correction = 0.0
        applied_bias_correction = 0.0
        display_max_cap_correction = 0.0
        drop_warning = False
        bias_rejected = False
        correction_limited = False

        if h_float <= 1 and nowcast_temp is not None:
            best = nowcast_temp
            method = "nowcast"

        elif bias_now is not None and h_float <= max_correction_hours:
            weight = clamp(1.0 - (h_float / max_correction_hours), 0.0, 1.0)

            # Keep the handoff conservative.
            if h_float <= 2:
                weight *= 0.25
            elif h_float <= 3:
                weight *= 0.40
            elif h_float <= 4:
                weight *= 0.55

            requested_bias_correction = bias_now * weight

            if abs(bias_now) >= reject_bias_at_f:
                applied_bias_correction = 0.0
                best = raw_temp
                method = "raw_grid_bias_rejected"
                bias_rejected = True
            else:
                applied_bias_correction = clamp(
                    requested_bias_correction,
                    -max_bias_correction_f,
                    max_bias_correction_f,
                )
                correction_limited = abs(applied_bias_correction - requested_bias_correction) >= 0.01
                best = raw_temp + applied_bias_correction
                method = "grid_bias_corrected"

        else:
            best = raw_temp
            method = "raw_grid"

        # Never let adjusted horizon values contradict the official/display max by much.
        if display_max_f is not None and best > display_max_f + max_above_display_max_f:
            before_cap = best
            best = display_max_f + max_above_display_max_f
            display_max_cap_correction = best - before_cap
            method = f"{method}_max_capped"

        # Detect sharp drops, but do not rewrite them aggressively.
        if previous_best is not None:
            drop = previous_best - best

            if drop >= 4.0 and h_float >= 5:
                drop_warning = True
                method = f"{method}_sharp_drop_flagged"

        total_correction = best - raw_temp

        confidence = horizon_temp_confidence(
            method=method,
            raw_temp=raw_temp,
            best_temp=best,
            dewpoint_f=item.get("dewpoint_f"),
            wind_speed_mph=item.get("wind_speed_mph"),
        )

        if drop_warning and confidence == "high":
            confidence = "medium"
        elif drop_warning and confidence == "medium":
            confidence = "low"

        if display_max_cap_correction != 0.0 and confidence == "high":
            confidence = "medium"

        fixed = dict(item)
        fixed["raw_temp_f"] = round(float(raw_temp), 2)
        fixed["temp_f"] = round(float(best), 2)
        fixed["correction_f"] = round(float(total_correction), 2)
        fixed["requested_bias_correction_f"] = round(float(requested_bias_correction), 2)
        fixed["applied_bias_correction_f"] = round(float(applied_bias_correction), 2)
        fixed["display_max_cap_correction_f"] = round(float(display_max_cap_correction), 2)
        fixed["bias_now_f"] = round(float(bias_now), 2) if bias_now is not None else None
        fixed["bias_rejected"] = bias_rejected
        fixed["correction_limited"] = correction_limited
        fixed["sharp_drop_flagged"] = drop_warning
        fixed["method"] = method
        fixed["confidence"] = confidence
        fixed["confidence_inputs"] = {
            "method": method,
            "raw_temp_f": round(float(raw_temp), 2),
            "best_temp_f": round(float(best), 2),
            "total_correction_f": round(float(total_correction), 2),
            "requested_bias_correction_f": round(float(requested_bias_correction), 2),
            "applied_bias_correction_f": round(float(applied_bias_correction), 2),
            "display_max_f": display_max_f,
            "display_max_cap_correction_f": round(float(display_max_cap_correction), 2),
            "dewpoint_f": item.get("dewpoint_f"),
            "wind_speed_mph": item.get("wind_speed_mph"),
            "sharp_drop_flagged": drop_warning,
        }

        corrected.append(fixed)
        previous_best = best

    return corrected
    
def apply_grid_window_to_results(results: dict, prefix: str, summary: dict) -> None:
    results[f"{prefix}_max_temp_f"] = summary.get("max_temp_f")
    results[f"{prefix}_max_dewpoint_f"] = summary.get("max_dewpoint_f")
    results[f"{prefix}_max_wind_speed_mph"] = summary.get("max_wind_speed_mph")
    results[f"{prefix}_max_wind_gust_mph"] = summary.get("max_wind_gust_mph")
    results[f"{prefix}_max_apparent_temp_f"] = summary.get("max_apparent_temp_f")
    results[f"{prefix}_max_heat_index_f"] = summary.get("max_heat_index_f")
    results[f"{prefix}_min_wind_chill_f"] = summary.get("min_wind_chill_f")
    results[f"{prefix}_max_pop_pct"] = summary.get("max_pop_pct")
    results[f"{prefix}_total_qpf_in"] = summary.get("total_qpf_in")
    results[f"{prefix}_total_snow_in"] = summary.get("total_snow_in")
    results[f"{prefix}_total_ice_in"] = summary.get("total_ice_in")
    results[f"{prefix}_avg_sky_cover_pct"] = summary.get("avg_sky_cover_pct")
    results[f"{prefix}_min_visibility_mi"] = summary.get("min_visibility_mi")
    results[f"{prefix}_period_count"] = summary.get("grid_period_count", 0)
    results[f"{prefix}_source"] = "forecastGridData" if summary.get("grid_available") else None

def apply_best_atmospheric_layer(results: dict) -> None:
    age_minutes = safe_float(results.get("current_obs_age_minutes") or results.get("obs_age_minutes"))
    station_distance_km = safe_float(results.get("station_distance_km"))
    current_temp_f = safe_float(results.get("current_temp_f"))

    obs_is_fresh = age_minutes is not None and 0 <= age_minutes <= 35

    obs_dew = valid_dewpoint_f(results.get("obs_dewpoint_f"), current_temp_f)
    grid_dew = valid_dewpoint_f(results.get("grid_next_hour_dewpoint_f"), results.get("grid_next_hour_temp_f"))
    hourly_dew = valid_dewpoint_f(results.get("next_hour_dewpoint_f"), results.get("next_hour_temp_f"))

    if obs_is_fresh and obs_dew is not None:
        dew = obs_dew
        dew_source = "station_obs"
    elif grid_dew is not None:
        dew = grid_dew
        dew_source = "forecastGridData.dewpoint"
    elif hourly_dew is not None:
        dew = hourly_dew
        dew_source = "forecastHourly.dewpoint"
    else:
        dew = None
        dew_source = None

    obs_wind = valid_wind_speed_mph(results.get("obs_wind_speed_mph"))
    grid_wind = valid_wind_speed_mph(results.get("grid_next_hour_wind_speed_mph"))
    hourly_wind = valid_wind_speed_mph(results.get("next_hour_wind_speed_mph"))

    if obs_is_fresh and obs_wind is not None:
        wind = obs_wind
        wind_source = "station_obs"
    elif grid_wind is not None:
        wind = grid_wind
        wind_source = "forecastGridData.windSpeed"
    elif hourly_wind is not None:
        wind = hourly_wind
        wind_source = "forecastHourly.windSpeed"
    else:
        wind = None
        wind_source = None

    obs_dir = valid_wind_direction_deg(results.get("obs_wind_dir_deg"))
    hourly_dir = results.get("next_hour_wind_direction")

    if obs_is_fresh and obs_dir is not None:
        wind_dir = obs_dir
        wind_dir_source = "station_obs"
    elif hourly_dir is not None:
        wind_dir = hourly_dir
        wind_dir_source = "forecastHourly.windDirection"
    else:
        wind_dir = None
        wind_dir_source = None

    results["best_next_hour_dewpoint_f"] = round(dew, 2) if dew is not None else None
    results["best_next_hour_dewpoint_source"] = dew_source
    results["best_next_hour_dewpoint_confidence"] = metric_confidence(
        dew_source,
        dew,
        age_minutes=age_minutes,
        station_distance_km=station_distance_km,
    )

    results["best_next_hour_wind_speed_mph"] = round(wind, 2) if wind is not None else None
    results["best_next_hour_wind_speed_source"] = wind_source
    results["best_next_hour_wind_speed_confidence"] = metric_confidence(
        wind_source,
        wind,
        age_minutes=age_minutes,
        station_distance_km=station_distance_km,
    )

    results["best_next_hour_wind_direction"] = wind_dir
    results["best_next_hour_wind_direction_source"] = wind_dir_source
    results["best_next_hour_wind_direction_confidence"] = metric_confidence(
        wind_dir_source,
        wind_dir,
        age_minutes=age_minutes,
        station_distance_km=station_distance_km,
    )

    # These are summary confidence labels for the cards.
    for hours in (3, 6):
        source = results.get(f"next_{hours}h_source") or (
            "forecastGridData" if results.get("grid_available") else "forecastHourly"
        )

        results[f"best_next_{hours}h_max_temp_confidence"] = metric_confidence(
            source,
            results.get(f"best_next_{hours}h_max_temp_f"),
        )
        results[f"best_next_{hours}h_max_dewpoint_confidence"] = metric_confidence(
            source,
            results.get(f"best_next_{hours}h_max_dewpoint_f"),
        )
        results[f"best_next_{hours}h_max_wind_speed_confidence"] = metric_confidence(
            source,
            results.get(f"best_next_{hours}h_max_wind_speed_mph"),
        )

def apply_latest_observation_layer(
    results: dict,
    station_id: str | None,
    future_date: bool,
    tz_name: str,
):
    if not station_id or future_date:
        return

    try:
        obs = get_latest_station_observation(station_id)
        results.update(obs)

        if not obs.get("obs_time_utc"):
            return

        obs_time = datetime.fromisoformat(obs["obs_time_utc"])

        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=timezone.utc)
        else:
            obs_time = obs_time.astimezone(timezone.utc)

        now = utc_now()
        age_seconds = (now - obs_time).total_seconds()
        age_minutes = round(age_seconds / 60.0, 1)
        results["obs_age_minutes"] = age_minutes
        results["obs_is_stale"] = age_minutes > 35
        results["best_temp_now_obs_f"] = obs.get("obs_temp_f")
        results["best_temp_now_obs_time_utc"] = obs.get("obs_time_utc")

        fresh_minutes = 35
        obs_temp = safe_float(obs.get("obs_temp_f"))
        current_temp = None
        current_source = None

        nowcast_15 = pick_nowcast_temp(results.get("nowcast_next_0_60m"), 15)
        nowcast_30 = pick_nowcast_temp(results.get("nowcast_next_0_60m"), 30)
        
        if obs_temp is not None and 0 <= age_seconds <= 35 * 60:
            current_temp = obs_temp
            current_source = "station_obs"
            results["current_obs_age_minutes"] = age_minutes
        elif nowcast_15 is not None:
            current_temp = nowcast_15
            current_source = "nowcast_15m"
        elif nowcast_30 is not None:
            current_temp = nowcast_30
            current_source = "nowcast_30m"
        elif results.get("best_next_hour_temp_f") is not None:
            current_temp = safe_float(results.get("best_next_hour_temp_f"))
            current_source = "best_next_hour"
        elif results.get("next_hour_temp_f") is not None:
            current_temp = safe_float(results.get("next_hour_temp_f"))
            current_source = "hourly_forecast"
        elif results.get("grid_next_hour_temp_f") is not None:
            current_temp = safe_float(results.get("grid_next_hour_temp_f"))
            current_source = "forecastGridData.temperature"

        results["current_temp_f"] = round(current_temp, 2) if current_temp is not None else None
        results["current_temp_source"] = current_source
        results["current_temp_confidence"] = confidence_from_source(
            current_source,
            age_minutes,
            safe_float(results.get("station_distance_km")),
        )

        if obs_temp is not None and 0 <= age_seconds <= fresh_minutes * 60:
            for hours in (3, 6):
                key = f"best_next_{hours}h_max_temp_f"
                existing = safe_float(results.get(key))
                results[key] = max(existing, obs_temp) if existing is not None else obs_temp

            obs_dewpoint = safe_float(obs.get("obs_dewpoint_f"))

            if obs_dewpoint is not None:
                for hours in (3, 6):
                    key = f"best_next_{hours}h_max_dewpoint_f"
                    existing = safe_float(results.get(key))
                    results[key] = max(existing, obs_dewpoint) if existing is not None else obs_dewpoint

            obs_wind_speed = safe_float(obs.get("obs_wind_speed_mph"))

            if obs_wind_speed is not None:
                for hours in (3, 6):
                    key = f"best_next_{hours}h_max_wind_speed_mph"
                    existing = safe_float(results.get(key))
                    results[key] = max(existing, obs_wind_speed) if existing is not None else obs_wind_speed

    except Exception as e:
        results["latest_obs_error"] = str(e)
        results["obs_age_minutes"] = None
        results["obs_is_stale"] = None
        results["obs_time_utc"] = None
        results["obs_temp_f"] = None
        results["obs_dewpoint_f"] = None
        results["obs_wind_speed_mph"] = None
        results["obs_wind_dir_deg"] = None


def highest_temp_for_day(
    lat: float,
    lon: float,
    target_date: date = None,
    tz_name: str = TIMEZONE,
):
    if tz_name is None:
        tz_name = tz_for_latlon(lat, lon)

    tz = ZoneInfo(tz_name)

    if target_date is None:
        target_date = datetime.now(tz).date()

    now = utc_now()
    today_local = datetime.now(tz).date()
    future_date = target_date > today_local
    past_date = target_date < today_local
    start_utc, observed_end_utc = day_range_local(target_date, tz_name)
    _, full_end_utc = day_range_local(target_date, tz_name)

    if target_date == today_local and now < observed_end_utc:
        observed_end_utc = now

    results = {
        "lat": lat,
        "lon": lon,
        "date": target_date.isoformat(),
        "timezone": tz_name,
        "window_start_utc": start_utc.isoformat(),
        "window_end_utc": observed_end_utc.isoformat(),
        "window_end_utc_full": full_end_utc.isoformat(),
        "is_future_date": future_date,
        "is_past_date": past_date,
    }

    try:
        pts_url = f"https://api.weather.gov/points/{lat},{lon}"
        pts = cached_get_json(pts_url)
    except Exception as e:
        results["points_error"] = str(e)
        results["station_id"] = None
        results["station_distance_km"] = None
        results["forecast_max_f"] = None
        results["grid_error"] = None
        results["grid_available"] = False
        set_empty_hourly_results(results)
        set_empty_window_results(results)
        results["station_running_max_f"] = None
        results["observed_max_f"] = None
        results["station_max_f"] = None
        results["display_forecasted_max_f"] = None
        results["best_estimate_max_f"] = None
        results["overall_max_f"] = None
        return results

    props = pts.get("properties", {}) or {}
    forecast_url = props.get("forecast")
    hourly_url = props.get("forecastHourly")
    grid_url = props.get("forecastGridData")
    station_id, station_distance_km = pick_closest_station_id(pts, lat, lon)
    results["station_id"] = station_id
    results["station_distance_km"] = station_distance_km

    gj = None

    if grid_url and not past_date:
        try:
            gj = cached_get_json(grid_url)
            day_grid = grid_day_summary(gj, start_utc, full_end_utc)
            results["grid_available"] = day_grid.get("grid_available")
            results["grid_update_time"] = day_grid.get("grid_update_time")
            results["grid_day_period_count"] = day_grid.get("grid_period_count")
            results["grid_official_max_temp_f"] = day_grid.get("official_max_temp_f")
            results["grid_official_min_temp_f"] = day_grid.get("official_min_temp_f")
            results["grid_max_temp_f"] = day_grid.get("max_temp_f")
            results["grid_min_temp_f"] = day_grid.get("min_temp_f")
            results["grid_best_max_temp_f"] = day_grid.get("best_max_temp_f")
            results["grid_best_min_temp_f"] = day_grid.get("best_min_temp_f")
            results["grid_best_max_temp_source"] = day_grid.get("best_max_temp_source")
            results["grid_best_min_temp_source"] = day_grid.get("best_min_temp_source")
            results["grid_max_dewpoint_f"] = day_grid.get("max_dewpoint_f")
            results["grid_min_dewpoint_f"] = day_grid.get("min_dewpoint_f")
            results["grid_max_apparent_temp_f"] = day_grid.get("max_apparent_temp_f")
            results["grid_max_heat_index_f"] = day_grid.get("max_heat_index_f")
            results["grid_min_wind_chill_f"] = day_grid.get("min_wind_chill_f")
            results["grid_max_relative_humidity_pct"] = day_grid.get("max_relative_humidity_pct")
            results["grid_min_relative_humidity_pct"] = day_grid.get("min_relative_humidity_pct")
            results["grid_avg_sky_cover_pct"] = day_grid.get("avg_sky_cover_pct")
            results["grid_max_wind_speed_mph"] = day_grid.get("max_wind_speed_mph")
            results["grid_max_wind_gust_mph"] = day_grid.get("max_wind_gust_mph")
            results["grid_avg_wind_direction_deg"] = day_grid.get("avg_wind_direction_deg")
            results["grid_max_pop_pct"] = day_grid.get("max_pop_pct")
            results["grid_total_qpf_in"] = day_grid.get("total_qpf_in")
            results["grid_total_snow_in"] = day_grid.get("total_snow_in")
            results["grid_total_ice_in"] = day_grid.get("total_ice_in")
            results["grid_min_visibility_mi"] = day_grid.get("min_visibility_mi")
            results["grid_min_ceiling_ft"] = day_grid.get("min_ceiling_ft")
            results["grid_avg_pressure_hpa"] = day_grid.get("avg_pressure_hpa")
            results["grid_dominant_weather"] = day_grid.get("dominant_weather")
            results["grid_hazards"] = day_grid.get("grid_hazards")
            results["forecasted_max_f"] = day_grid.get("best_max_temp_f")
            results["forecast_source"] = day_grid.get("best_max_temp_source")

            nxt_grid = grid_next_hour(gj, now)
            results.update(nxt_grid)

            horizons = []
            for h in range(1, 13):
                item = temp_at_horizon_from_grid(gj, now, h, tz_name)
                if item:
                    horizons.append(item)
            results["best_temp_next_1_24h"] = horizons

            for hours in (3, 6):
                window_start = now
                window_end = now + timedelta(hours=hours)
                summary = grid_window_summary(gj, window_start, window_end)
                prefix = f"next_{hours}h"
                results[f"{prefix}_start_utc"] = window_start.isoformat()
                results[f"{prefix}_end_utc"] = window_end.isoformat()
                apply_grid_window_to_results(results, prefix, summary)

        except Exception as e:
            results["grid_error"] = str(e)
            results["grid_available"] = False
    else:
        results["grid_available"] = False

    try:
        alert_data = fetch_active_alerts_for_point(lat, lon)
        results.update(alert_data)
    except Exception as e:
        results["active_alerts"] = []
        results["active_alerts_error"] = str(e)

    if forecast_url and not past_date:
        try:
            fjson = cached_get_json(forecast_url)
            results["forecast_max_f"] = max_from_forecast_periods(
                fjson,
                start_utc,
                full_end_utc,
            )
        except Exception as e:
            results["forecast_error"] = str(e)
            results["forecast_max_f"] = None
    else:
        results["forecast_max_f"] = None

    hj = None

    if hourly_url and not past_date:
        try:
            hj = cached_get_json(hourly_url)
            results["hourly_min_f"] = min_from_hourly(hj, start_utc, full_end_utc)
            results["hourly_max_f"] = max_from_hourly(hj, start_utc, full_end_utc)

            if results.get("forecasted_max_f") is None:
                results["forecasted_max_f"] = results["hourly_max_f"]
                results["forecast_source"] = "forecastHourly" if results["hourly_max_f"] is not None else None

            if not results.get("best_temp_next_1_24h"):
                horizons = []
                for h in range(1, 25):
                    item = temp_at_horizon_from_hourly(hj, now, h, tz_name)
                    if item:
                        horizons.append(item)
                results["best_temp_next_1_24h"] = horizons

            nxt = next_hour_from_hourly(hj, now)

            if results.get("next_hour_temp_f") is None:
                results.update(nxt)
            else:
                results["hourly_next_hour_start_utc"] = nxt.get("next_hour_start_utc")
                results["hourly_next_hour_temp_f"] = nxt.get("next_hour_temp_f")
                results["hourly_next_hour_dewpoint_f"] = nxt.get("next_hour_dewpoint_f")
                results["hourly_next_hour_wind_speed_mph"] = nxt.get("next_hour_wind_speed_mph")
                results["hourly_next_hour_wind_direction"] = nxt.get("next_hour_wind_direction")

            if target_date == today_local:
                if station_id:
                    try:
                        results["station_running_min_f"] = min_from_station_observations(
                            station_id,
                            start_utc,
                            observed_end_utc,
                        )
                    except Exception as e:
                        results["station_min_error"] = str(e)
                        results["station_running_min_f"] = None
                else:
                    results["station_running_min_f"] = None

                low_model = estimate_remaining_day_low_model(
                    station_id=station_id,
                    hourly_json=hj,
                    now_utc=now,
                    end_utc_full=full_end_utc,
                    lookback_minutes=180,
                )
                results["pred_low_remaining_f"] = low_model["pred_low_remaining_f"]
                results["pred_low_remaining_points"] = low_model["pred_points"]
            else:
                results["station_running_min_f"] = None
                results["pred_low_remaining_f"] = None
                results["pred_low_remaining_points"] = []

            low_candidates = [
                results.get("station_running_min_f"),
                results.get("pred_low_remaining_f"),
                results.get("grid_best_min_temp_f"),
            ]
            low_candidates = [x for x in low_candidates if x is not None]
            results["best_estimate_low_f"] = min(low_candidates) if low_candidates else results.get("hourly_min_f")
            results["overall_min_f"] = results.get("best_estimate_low_f")

            try:
                forecast_for_nowcast = (
                    results.get("grid_next_hour_temp_f")
                    or results.get("next_hour_temp_f")
                    or results.get("hourly_next_hour_temp_f")
                )
                nowcast = build_nowcast_layer(
                    station_id=station_id,
                    now_utc=now,
                    forecast_next_hour_temp_f=forecast_for_nowcast,
                    lookback_minutes=90,
                    freshness_minutes=40,
                )
                results.update(nowcast)
            except Exception as e:
                results["nowcast_error"] = str(e)

            if not results.get("grid_available"):
                w3 = window_max_from_hourly(hj, now, 3)
                w6 = window_max_from_hourly(hj, now, 6)

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

            results["station_max_last_6h_f"] = station_max_last_hours(station_id, now, 6) if station_id else None
            results["station_max_last_24h_f"] = station_max_last_hours(station_id, now, 24) if station_id else None
            results["debug_now_utc"] = now.isoformat()
            results["debug_now_local"] = now.astimezone(tz).isoformat()
            results["debug_midnight_local"] = datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                tzinfo=tz,
            ).isoformat()
            results["debug_window_start_utc"] = start_utc.isoformat()
            results["debug_window_end_utc"] = observed_end_utc.isoformat()

        except Exception as e:
            results["hourly_error"] = str(e)

            if not results.get("grid_available"):
                set_empty_hourly_results(results)
                set_empty_window_results(results)
    else:
        if not results.get("grid_available"):
            set_empty_hourly_results(results)
            set_empty_window_results(results)

    if station_id and not future_date:
        try:
            obs_max = max_from_station_observations(station_id, start_utc, observed_end_utc)

            if obs_max is None and target_date == today_local:
                latest = get_latest_station_observation(station_id)
                obs_max = latest.get("obs_temp_f")

            results["station_running_max_f"] = obs_max
        except Exception as e:
            results["station_error"] = str(e)
            results["station_running_max_f"] = None
    else:
        results["station_running_max_f"] = None

    results["observed_max_f"] = results.get("station_running_max_f")
    results["station_max_f"] = results.get("observed_max_f")

    if results.get("grid_best_max_temp_f") is not None:
        results["display_forecasted_max_f"] = results["grid_best_max_temp_f"]
        results["display_forecasted_max_source"] = results.get("grid_best_max_temp_source")
    elif results.get("hourly_max_f") is not None:
        results["display_forecasted_max_f"] = results["hourly_max_f"]
        results["display_forecasted_max_source"] = "forecastHourly"
    else:
        results["display_forecasted_max_f"] = results.get("forecast_max_f")
        results["display_forecasted_max_source"] = "forecastPeriods" if results.get("forecast_max_f") is not None else None

    best_candidates = [
        results.get("observed_max_f"),
        results.get("display_forecasted_max_f"),
    ]
    best_candidates = [x for x in best_candidates if x is not None]
    results["best_estimate_max_f"] = max(best_candidates) if best_candidates else None
    results["overall_max_f"] = results.get("best_estimate_max_f")

    if (
        station_id
        and results.get("station_max_f") is not None
        and results.get("forecasted_max_f") is not None
        and target_date < today_local
    ):
        try:
            record_bias_sample(
                station_id,
                target_date,
                float(results["forecasted_max_f"]),
                float(results["station_max_f"]),
            )
        except Exception as e:
            results["bias_logging_error"] = str(e)

    apply_forecast_calibration(results, station_id)

    nowcast_60 = pick_nowcast_temp(results.get("nowcast_next_0_60m"), 60)
    results["best_next_hour_temp_f"] = (
        nowcast_60
        if nowcast_60 is not None
        else results.get("grid_next_hour_temp_f")
        if results.get("grid_next_hour_temp_f") is not None
        else results.get("next_hour_temp_f")
    )
    results["best_next_hour_dewpoint_f"] = None
    results["best_next_hour_dewpoint_source"] = None
    results["best_next_hour_dewpoint_confidence"] = "low"

    results["best_next_hour_wind_speed_mph"] = None
    results["best_next_hour_wind_speed_source"] = None
    results["best_next_hour_wind_speed_confidence"] = "low"

    results["best_next_hour_wind_gust_mph"] = results.get("grid_next_hour_wind_gust_mph")
    results["best_next_hour_apparent_temp_f"] = results.get("grid_next_hour_apparent_temp_f")
    results["best_next_hour_heat_index_f"] = results.get("grid_next_hour_heat_index_f")
    results["best_next_hour_wind_chill_f"] = results.get("grid_next_hour_wind_chill_f")
    results["best_next_hour_pop_pct"] = results.get("grid_next_hour_pop_pct")
    results["best_next_hour_sky_cover_pct"] = results.get("grid_next_hour_sky_cover_pct")
    results["best_next_hour_visibility_mi"] = results.get("grid_next_hour_visibility_mi")
    results["best_next_hour_wind_direction"] = results.get("next_hour_wind_direction")
    results["best_next_3h_max_temp_f"] = results.get("next_3h_max_temp_f")
    results["best_next_3h_max_dewpoint_f"] = results.get("next_3h_max_dewpoint_f")
    results["best_next_3h_max_wind_speed_mph"] = results.get("next_3h_max_wind_speed_mph")
    results["best_next_3h_max_wind_gust_mph"] = results.get("next_3h_max_wind_gust_mph")
    results["best_next_3h_max_apparent_temp_f"] = results.get("next_3h_max_apparent_temp_f")
    results["best_next_3h_max_heat_index_f"] = results.get("next_3h_max_heat_index_f")
    results["best_next_3h_min_wind_chill_f"] = results.get("next_3h_min_wind_chill_f")
    results["best_next_3h_max_pop_pct"] = results.get("next_3h_max_pop_pct")
    results["best_next_3h_total_qpf_in"] = results.get("next_3h_total_qpf_in")
    results["best_next_3h_total_snow_in"] = results.get("next_3h_total_snow_in")
    results["best_next_3h_total_ice_in"] = results.get("next_3h_total_ice_in")
    results["best_next_3h_avg_sky_cover_pct"] = results.get("next_3h_avg_sky_cover_pct")
    results["best_next_3h_min_visibility_mi"] = results.get("next_3h_min_visibility_mi")
    results["best_next_6h_max_temp_f"] = results.get("next_6h_max_temp_f")
    results["best_next_6h_max_dewpoint_f"] = results.get("next_6h_max_dewpoint_f")
    results["best_next_6h_max_wind_speed_mph"] = results.get("next_6h_max_wind_speed_mph")
    results["best_next_6h_max_wind_gust_mph"] = results.get("next_6h_max_wind_gust_mph")
    results["best_next_6h_max_apparent_temp_f"] = results.get("next_6h_max_apparent_temp_f")
    results["best_next_6h_max_heat_index_f"] = results.get("next_6h_max_heat_index_f")
    results["best_next_6h_min_wind_chill_f"] = results.get("next_6h_min_wind_chill_f")
    results["best_next_6h_max_pop_pct"] = results.get("next_6h_max_pop_pct")
    results["best_next_6h_total_qpf_in"] = results.get("next_6h_total_qpf_in")
    results["best_next_6h_total_snow_in"] = results.get("next_6h_total_snow_in")
    results["best_next_6h_total_ice_in"] = results.get("next_6h_total_ice_in")
    results["best_next_6h_avg_sky_cover_pct"] = results.get("next_6h_avg_sky_cover_pct")
    results["best_next_6h_min_visibility_mi"] = results.get("next_6h_min_visibility_mi")

    apply_latest_observation_layer(results, station_id, future_date, tz_name)
    apply_best_atmospheric_layer(results)
    nowcast_60 = pick_nowcast_temp(results.get("nowcast_next_0_60m"), 60)
    results["best_next_hour_temp_f"] = (nowcast_60 if nowcast_60 is not None else results.get("grid_next_hour_temp_f") if results.get("grid_next_hour_temp_f") is not None else results.get("next_hour_temp_f"))

    results["raw_temp_next_1_24h"] = results.get("best_temp_next_1_24h", [])
    results["best_temp_next_1_24h"] = corrected_horizon_temps(
        raw_horizons=results.get("raw_temp_next_1_24h"),
        nowcast_points=results.get("nowcast_next_0_60m"),
        current_temp_f=results.get("current_temp_f"),
        raw_current_grid_temp_f=results.get("grid_next_hour_temp_f"),
        display_max_f=results.get("display_forecasted_max_f"),
        max_correction_hours=6,
        max_bias_correction_f=3.0,
        reject_bias_at_f=6.0,
    )

    results["overall_max_source"] = (
        "station_observation"
        if results.get("observed_max_f") is not None and results.get("observed_max_f") == results.get("overall_max_f")
        else results.get("display_forecasted_max_source")
    )
    results["overall_max_confidence"] = confidence_from_source(results.get("overall_max_source"))
    results["overall_min_source"] = (
        "forecastGridData.minTemperature"
        if results.get("grid_best_min_temp_f") is not None and results.get("overall_min_f") == results.get("grid_best_min_temp_f")
        else "station_or_nowcast"
        if results.get("overall_min_f") is not None
        else None
    )
    results["overall_min_confidence"] = confidence_from_source(results.get("overall_min_source"))
    results["display_forecasted_max_confidence"] = confidence_from_source(
        results.get("display_forecasted_max_source")
    )

    results["forecasted_max_confidence"] = confidence_from_source(
        results.get("forecast_source")
    )

    results["station_max_confidence"] = confidence_from_source(
        "station_observation" if results.get("station_max_f") is not None else None,
        results.get("obs_age_minutes"),
        results.get("station_distance_km"),
    )

    results["best_next_hour_temp_confidence"] = confidence_from_source(
        "nowcast" if nowcast_60 is not None else (
            "forecastGridData.temperature"
            if results.get("grid_next_hour_temp_f") is not None
            else "forecastHourly"
        )
    )

    apply_prediction_confidence_layer(results)

    return results