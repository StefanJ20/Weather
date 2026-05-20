from datetime import timedelta

from django.db.models import Avg, Count
from django.utils import timezone as dj_timezone

from ..models import StationBiasSample


def record_bias_sample(station_id: str, day, forecast_high_f: float, observed_high_f: float) -> None:
    StationBiasSample.objects.update_or_create(
        station_id=station_id,
        date=day,
        defaults={
            "forecast_high_f": float(forecast_high_f),
            "observed_high_f": float(observed_high_f),
            "error_f": float(observed_high_f) - float(forecast_high_f),
        },
    )


def get_station_bias(station_id: str, lookback_days: int = 120) -> dict | None:
    cutoff = dj_timezone.now().date() - timedelta(days=lookback_days)

    qs = StationBiasSample.objects.filter(
        station_id=station_id,
        date__gte=cutoff,
    )

    agg = qs.aggregate(
        sample_count=Count("id"),
        mean_error_f=Avg("error_f"),
    )

    n = int(agg["sample_count"] or 0)

    if n <= 0:
        return None

    return {
        "sample_count": n,
        "mean_error_f": float(agg["mean_error_f"] or 0.0),
        "lookback_days": lookback_days,
    }


def get_station_calibration(station_id: str, lookback_days: int = 365) -> dict | None:
    cutoff = dj_timezone.now().date() - timedelta(days=lookback_days)

    qs = StationBiasSample.objects.filter(
        station_id=station_id,
        date__gte=cutoff,
    ).values(
        "forecast_high_f",
        "observed_high_f",
    )

    rows = list(qs)
    n = len(rows)

    if n < 30:
        return None

    xs = [float(r["forecast_high_f"]) for r in rows]
    ys = [float(r["observed_high_f"]) for r in rows]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    denom = sum((x - x_mean) ** 2 for x in xs)

    if denom == 0:
        return None

    b = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n)) / denom
    a = y_mean - b * x_mean

    return {
        "sample_count": n,
        "a": float(a),
        "b": float(b),
        "lookback_days": lookback_days,
    }