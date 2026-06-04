from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional, TypedDict

from django.db import IntegrityError, transaction
from django.db.models import Avg, Count, Max, Min
from django.utils import timezone as dj_timezone

from ..models import StationBiasSample


class StationBiasResult(TypedDict):
    sample_count: int
    mean_error_f: float
    lookback_days: int
    min_date: date | None
    max_date: date | None


class StationCalibrationResult(TypedDict):
    sample_count: int
    a: float
    b: float
    r2: float
    rmse_f: float
    lookback_days: int
    min_date: date | None
    max_date: date | None


def _is_valid_temp_f(value: object) -> bool:
    if value is None:
        return False

    try:
        x = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(x) and -100.0 <= x <= 150.0


def _normalized_station_id(station_id: str) -> str:
    return str(station_id or "").strip().upper()


def record_bias_sample(
    station_id: str,
    day: date,
    forecast_high_f: float,
    observed_high_f: float,
) -> None:
    sid = _normalized_station_id(station_id)

    if not sid:
        return

    if not isinstance(day, date):
        return

    if not _is_valid_temp_f(forecast_high_f) or not _is_valid_temp_f(observed_high_f):
        return

    forecast = float(forecast_high_f)
    observed = float(observed_high_f)
    error = observed - forecast

    defaults = {
        "forecast_high_f": forecast,
        "observed_high_f": observed,
        "error_f": error,
    }

    try:
        with transaction.atomic():
            StationBiasSample.objects.update_or_create(
                station_id=sid,
                date=day,
                defaults=defaults,
            )
    except IntegrityError:
        StationBiasSample.objects.filter(station_id=sid, date=day).update(**defaults)


def get_station_bias(station_id: str, lookback_days: int = 120) -> Optional[StationBiasResult]:
    sid = _normalized_station_id(station_id)

    if not sid or lookback_days <= 0:
        return None

    cutoff = dj_timezone.localdate() - timedelta(days=lookback_days)

    qs = StationBiasSample.objects.filter(
        station_id=sid,
        date__gte=cutoff,
        forecast_high_f__isnull=False,
        observed_high_f__isnull=False,
        error_f__isnull=False,
    )

    agg = qs.aggregate(
        sample_count=Count("id"),
        mean_error_f=Avg("error_f"),
        min_date=Min("date"),
        max_date=Max("date"),
    )

    n = int(agg["sample_count"] or 0)
    mean_error = agg["mean_error_f"]

    if n <= 0 or mean_error is None:
        return None

    return {
        "sample_count": n,
        "mean_error_f": float(mean_error),
        "lookback_days": int(lookback_days),
        "min_date": agg["min_date"],
        "max_date": agg["max_date"],
    }


def get_station_calibration(
    station_id: str,
    lookback_days: int = 365,
    min_samples: int = 60,
) -> Optional[StationCalibrationResult]:
    sid = _normalized_station_id(station_id)

    if not sid or lookback_days <= 0 or min_samples <= 1:
        return None

    cutoff = dj_timezone.localdate() - timedelta(days=lookback_days)

    rows = list(
        StationBiasSample.objects.filter(
            station_id=sid,
            date__gte=cutoff,
            forecast_high_f__isnull=False,
            observed_high_f__isnull=False,
        )
        .order_by("date")
        .values("date", "forecast_high_f", "observed_high_f")
    )

    clean_rows = []

    for row in rows:
        forecast = row.get("forecast_high_f")
        observed = row.get("observed_high_f")

        if not _is_valid_temp_f(forecast) or not _is_valid_temp_f(observed):
            continue

        clean_rows.append(
            {
                "date": row.get("date"),
                "forecast_high_f": float(forecast),
                "observed_high_f": float(observed),
            }
        )

    n = len(clean_rows)

    if n < min_samples:
        return None

    xs = [row["forecast_high_f"] for row in clean_rows]
    ys = [row["observed_high_f"] for row in clean_rows]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    ss_xx = sum((x - x_mean) ** 2 for x in xs)

    if ss_xx <= 1e-10:
        return None

    ss_xy = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    b = ss_xy / ss_xx
    a = y_mean - b * x_mean

    preds = [a + b * x for x in xs]
    residuals = [ys[i] - preds[i] for i in range(n)]

    sse = sum(r * r for r in residuals)
    rmse = math.sqrt(sse / n)

    ss_yy = sum((y - y_mean) ** 2 for y in ys)
    r2 = 0.0 if ss_yy <= 1e-10 else max(0.0, min(1.0, 1.0 - (sse / ss_yy)))

    dates = [row["date"] for row in clean_rows if row.get("date") is not None]

    return {
        "sample_count": n,
        "a": float(a),
        "b": float(b),
        "r2": float(r2),
        "rmse_f": float(rmse),
        "lookback_days": int(lookback_days),
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
    }
