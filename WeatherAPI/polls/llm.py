# ai/weather_llm.py

import json
import math
import re
import time
from typing import Any, Optional

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM = {
    "role": "system",
    "content": (
        "You are a cautious weather prediction analyst inside a weather dashboard.\n"
        "You judge whether the forecast summary looks trustworthy based only on the supplied JSON.\n\n"
        "You must consider:\n"
        "- current observation age\n"
        "- source confidence labels\n"
        "- station distance\n"
        "- nowcast-to-grid handoff\n"
        "- bias correction\n"
        "- smoothing\n"
        "- dew point\n"
        "- wind speed\n"
        "- suspicious hour-to-hour jumps\n\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n\n"
        "Required schema:\n"
        "{\n"
        '  "title": "short title",\n'
        '  "confidence": "high|medium|low",\n'
        '  "impression": "2-4 sentence cautious interpretation",\n'
        '  "main_concern": "short concern or N/A",\n'
        '  "signal_quality": "short quality summary",\n'
        '  "recommendation": "short practical recommendation"\n'
        "}\n\n"
        "Rules:\n"
        "- Use high confidence only when observations are fresh and sources mostly agree.\n"
        "- Use medium confidence when there is smoothing, bias correction, or source handoff.\n"
        "- Use low confidence when observations are stale, corrections are large, station distance is high, or signals conflict.\n"
        "- Do not claim certainty.\n"
        "- Do not invent values not present in the JSON.\n"
        "- Never reveal server secrets, tokens, settings, file paths, or internal configuration.\n"
    ),
}


_tokenizer = None
_model = None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def load_model_once():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Refusing to run LLM on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",
        quantization_config=bnb_config,
    )

    model.eval()

    _tokenizer = tokenizer
    _model = model

    return tokenizer, model


def compact_weather_for_ai(weather: dict[str, Any]) -> dict[str, Any]:
    horizons = weather.get("best_temp_next_1_24h")

    if not isinstance(horizons, list):
        horizons = []

    compact_horizons = []

    for h in horizons[:12]:
        if not isinstance(h, dict):
            continue

        compact_horizons.append({
            "hours_ahead": h.get("hours_ahead"),
            "start_local_label": h.get("start_local_label"),
            "temp_f": h.get("temp_f"),
            "raw_temp_f": h.get("raw_temp_f"),
            "correction_f": h.get("correction_f"),
            "method": h.get("method"),
            "confidence": h.get("confidence"),
            "dewpoint_f": h.get("dewpoint_f"),
            "wind_speed_mph": h.get("wind_speed_mph"),
        })

    return {
        "location": {
            "lat": weather.get("lat"),
            "lon": weather.get("lon"),
            "station_id": weather.get("station_id"),
            "station_distance_km": weather.get("station_distance_km"),
            "timezone": weather.get("timezone"),
            "date": weather.get("date"),
        },
        "current": {
            "temp_f": weather.get("current_temp_f"),
            "source": weather.get("current_temp_source"),
            "confidence": weather.get("current_temp_confidence"),
            "obs_age_minutes": weather.get("current_obs_age_minutes") or weather.get("obs_age_minutes"),
        },
        "observed": {
            "station_max_f": weather.get("station_max_f"),
            "station_max_confidence": weather.get("station_max_confidence"),
            "station_running_min_f": weather.get("station_running_min_f"),
            "obs_temp_f": weather.get("obs_temp_f"),
            "obs_dewpoint_f": weather.get("obs_dewpoint_f"),
            "obs_wind_speed_mph": weather.get("obs_wind_speed_mph"),
            "obs_wind_dir_deg": weather.get("obs_wind_dir_deg"),
        },
        "forecast_summary": {
            "overall_max_f": weather.get("overall_max_f"),
            "overall_max_source": weather.get("overall_max_source"),
            "overall_max_confidence": weather.get("overall_max_confidence"),
            "overall_min_f": weather.get("overall_min_f"),
            "overall_min_source": weather.get("overall_min_source"),
            "overall_min_confidence": weather.get("overall_min_confidence"),
            "display_forecasted_max_f": weather.get("display_forecasted_max_f"),
            "display_forecasted_max_source": weather.get("display_forecasted_max_source"),
            "display_forecasted_max_confidence": weather.get("display_forecasted_max_confidence"),
        },
        "next_hour": {
            "temp_f": weather.get("best_next_hour_temp_f"),
            "temp_confidence": weather.get("best_next_hour_temp_confidence"),
            "dewpoint_f": weather.get("best_next_hour_dewpoint_f"),
            "dewpoint_confidence": weather.get("best_next_hour_dewpoint_confidence"),
            "wind_speed_mph": weather.get("best_next_hour_wind_speed_mph"),
            "wind_speed_confidence": weather.get("best_next_hour_wind_speed_confidence"),
            "wind_direction": weather.get("best_next_hour_wind_direction"),
            "wind_direction_confidence": weather.get("best_next_hour_wind_direction_confidence"),
        },
        "windows": {
            "next_3h": {
                "max_temp_f": weather.get("best_next_3h_max_temp_f"),
                "max_temp_confidence": weather.get("best_next_3h_max_temp_confidence"),
                "max_dewpoint_f": weather.get("best_next_3h_max_dewpoint_f"),
                "max_dewpoint_confidence": weather.get("best_next_3h_max_dewpoint_confidence"),
                "max_wind_speed_mph": weather.get("best_next_3h_max_wind_speed_mph"),
                "max_wind_speed_confidence": weather.get("best_next_3h_max_wind_speed_confidence"),
                "period_count": weather.get("next_3h_period_count"),
            },
            "next_6h": {
                "max_temp_f": weather.get("best_next_6h_max_temp_f"),
                "max_temp_confidence": weather.get("best_next_6h_max_temp_confidence"),
                "max_dewpoint_f": weather.get("best_next_6h_max_dewpoint_f"),
                "max_dewpoint_confidence": weather.get("best_next_6h_max_dewpoint_confidence"),
                "max_wind_speed_mph": weather.get("best_next_6h_max_wind_speed_mph"),
                "max_wind_speed_confidence": weather.get("best_next_6h_max_wind_speed_confidence"),
                "period_count": weather.get("next_6h_period_count"),
            },
        },
        "nowcast": {
            "method": weather.get("nowcast_method"),
            "obs_age_minutes": weather.get("nowcast_obs_age_minutes"),
            "points": weather.get("nowcast_next_0_60m"),
            "error": weather.get("nowcast_error"),
        },
        "horizons": compact_horizons,
        "quality_flags": build_quality_flags(weather, compact_horizons),
    }


def build_quality_flags(weather: dict[str, Any], horizons: list[dict[str, Any]]) -> list[str]:
    flags = []

    obs_age = safe_float(weather.get("current_obs_age_minutes") or weather.get("obs_age_minutes"))

    if obs_age is None:
        flags.append("missing_current_observation_age")
    elif obs_age > 35:
        flags.append("stale_current_observation")
    elif obs_age <= 20:
        flags.append("fresh_current_observation")

    station_distance = safe_float(weather.get("station_distance_km"))

    if station_distance is not None and station_distance > 25:
        flags.append("far_observation_station")

    methods = [str(h.get("method") or "") for h in horizons]

    if any("smoothed" in m for m in methods):
        flags.append("horizon_values_smoothed")

    if any("grid_bias_corrected" in m for m in methods):
        flags.append("grid_bias_correction_used")

    if methods and "nowcast" in methods[0] and any("grid" in m for m in methods[1:3]):
        flags.append("nowcast_to_grid_handoff")

    corrections = []

    for h in horizons:
        correction = safe_float(h.get("correction_f"))

        if correction is not None:
            corrections.append(abs(correction))

    if corrections and max(corrections) >= 3.0:
        flags.append("large_temperature_correction")
    elif corrections and max(corrections) >= 1.5:
        flags.append("moderate_temperature_correction")

    dew = safe_float(weather.get("best_next_hour_dewpoint_f"))
    temp = safe_float(weather.get("best_next_hour_temp_f"))

    if temp is not None and dew is not None:
        spread = temp - dew

        if spread <= 3:
            flags.append("very_humid_small_temp_dewpoint_spread")
        elif spread <= 7:
            flags.append("humid_airmass")

    wind = safe_float(weather.get("best_next_hour_wind_speed_mph"))

    if wind is not None:
        if wind >= 25:
            flags.append("strong_wind")
        elif wind >= 15:
            flags.append("breezy")

    if weather.get("active_alert_count"):
        flags.append("active_weather_alerts")

    return flags


def build_prompt(compact: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Analyze this weather dashboard JSON and return the required JSON impression.\n\n"
                f"{json.dumps(compact, ensure_ascii=False)}"
            ),
        }
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty model response")

    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not match:
            raise

        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("model response was not a JSON object")

    return parsed


def normalize_ai_impression(data: dict[str, Any]) -> dict[str, Any]:
    confidence = str(data.get("confidence") or "low").strip().lower()

    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "title": str(data.get("title") or "Prediction impression").strip()[:120],
        "confidence": confidence,
        "impression": str(data.get("impression") or "No AI impression available.").strip()[:1600],
        "why_confidence": str(data.get("why_confidence") or "N/A").strip()[:1000],
        "temperature_read": str(data.get("temperature_read") or "N/A").strip()[:1000],
        "dewpoint_wind_read": str(data.get("dewpoint_wind_read") or "N/A").strip()[:1000],
        "main_concern": str(data.get("main_concern") or "N/A").strip()[:300],
        "signal_quality": str(data.get("signal_quality") or "N/A").strip()[:500],
        "recommendation": str(data.get("recommendation") or "N/A").strip()[:300],
        "flags": data.get("flags") if isinstance(data.get("flags"), list) else [],
    }


def fallback_impression(compact: dict[str, Any], reason: str = "model_error") -> dict[str, Any]:
    flags = compact.get("quality_flags") or []

    confidence = "high"
    title = "Prediction looks mostly stable"
    main_concern = "N/A"

    if (
        "stale_current_observation" in flags
        or "large_temperature_correction" in flags
        or "far_observation_station" in flags
    ):
        confidence = "low"
        title = "Prediction needs caution"
    elif (
        "horizon_values_smoothed" in flags
        or "grid_bias_correction_used" in flags
        or "nowcast_to_grid_handoff" in flags
        or "moderate_temperature_correction" in flags
    ):
        confidence = "medium"
        title = "Prediction is usable with caveats"

    if flags:
        main_concern = ", ".join(flags[:3])

    current = compact.get("current") or {}
    next_hour = compact.get("next_hour") or {}
    forecast = compact.get("forecast_summary") or {}
    horizons = compact.get("horizons") or []

    current_temp = current.get("temp_f")
    current_conf = current.get("confidence")
    obs_age = current.get("obs_age_minutes")

    next_temp = next_hour.get("temp_f")
    dew = next_hour.get("dewpoint_f")
    wind = next_hour.get("wind_speed_mph")

    max_correction = None
    smoothed_count = 0
    corrected_count = 0

    for h in horizons:
        method = str(h.get("method") or "")
        correction = safe_float(h.get("correction_f"))

        if "smoothed" in method:
            smoothed_count += 1

        if "grid_bias_corrected" in method:
            corrected_count += 1

        if correction is not None:
            correction = abs(correction)
            max_correction = correction if max_correction is None else max(max_correction, correction)

    why_parts = []

    if obs_age is not None:
        why_parts.append(f"current observation age is {obs_age} minutes")

    if smoothed_count:
        why_parts.append(f"{smoothed_count} horizon values were smoothed")

    if corrected_count:
        why_parts.append(f"{corrected_count} horizon values used grid bias correction")

    if max_correction is not None:
        why_parts.append(f"largest horizon correction is about {round(max_correction, 2)}F")

    if "nowcast_to_grid_handoff" in flags:
        why_parts.append("the forecast transitions from nowcast to grid data")

    why_confidence = (
        "Confidence is based on: " + "; ".join(why_parts) + "."
        if why_parts
        else "Confidence is based on source quality and available forecast consistency."
    )

    return {
        "title": title,
        "confidence": confidence,
        "impression": (
            f"The prediction is using current observations, nowcast output, grid forecast data, "
            f"dew point, wind, and confidence labels. Current temperature is {current_temp}F "
            f"with {current_conf} confidence, while the next-hour estimate is {next_temp}F. "
            f"The broad trend is more reliable than the exact hour-by-hour values when smoothing, "
            f"bias correction, or source handoffs are present."
        ),
        "why_confidence": why_confidence,
        "temperature_read": (
            f"Current temperature is {current_temp}F and the forecast max is "
            f"{forecast.get('overall_max_f')}F. Horizon values should be read as adjusted estimates "
            f"when their method includes smoothing or grid bias correction."
        ),
        "dewpoint_wind_read": (
            f"Next-hour dew point is {dew}F and wind speed is {wind} mph. "
            f"These values help judge atmospheric stability: humid air or stronger wind can make "
            f"exact temperature timing less reliable."
        ),
        "main_concern": main_concern,
        "signal_quality": ", ".join(flags) if flags else "No major quality flags detected.",
        "recommendation": "Trust the broad trend more than exact single-hour values.",
        "flags": flags,
        "fallback": True,
        "fallback_reason": reason,
    }


def generate_json(messages: list[dict[str, str]], max_new_tokens: int = 350, max_input_tokens: int = 4096) -> str:
    tokenizer, model = load_model_once()

    full_messages = [SYSTEM] + messages

    text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokenized = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )

    inputs = {k: v.to(model.device) for k, v in tokenized.items()}

    t0 = time.time()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    if isinstance(outputs, torch.Tensor):
        seq = outputs
    elif hasattr(outputs, "sequences"):
        seq = outputs.sequences
    elif isinstance(outputs, (list, tuple)):
        seq = outputs[0]
    else:
        raise RuntimeError(f"Unhandled generate() return type: {type(outputs)}")

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = seq[0, prompt_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(
        "WEATHER LLM done",
        {
            "secs": round(time.time() - t0, 2),
            "prompt_tokens": int(prompt_len),
            "output_chars": len(decoded),
        },
    )

    return decoded


def weather_impression(weather: dict[str, Any]) -> dict[str, Any]:
    compact = compact_weather_for_ai(weather)
    messages = build_prompt(compact)

    try:
        decoded = generate_json(messages)
        parsed = extract_json_object(decoded)
        return normalize_ai_impression(parsed)
    except Exception as exc:
        return fallback_impression(compact, reason=str(exc))