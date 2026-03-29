from __future__ import annotations

import json
import urllib.request

WMO_DESCRIPTIONS: dict[int, str] = {
    0: "Clear", 1: "Mostly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Showers", 81: "Rain Showers", 82: "Heavy Showers",
    85: "Snow Showers", 86: "Heavy Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ Hail", 99: "Thunderstorm",
}


def get_weather_summary(lat: float, lon: float, *, timeout: int = 5) -> str:
    """Return e.g. '72°F, Partly Cloudy', or '' on any failure."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weather_code"
        f"&temperature_unit=fahrenheit&forecast_days=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        temp = data["current"]["temperature_2m"]
        code = int(data["current"]["weather_code"])
        cond = WMO_DESCRIPTIONS.get(code, f"Code {code}")
        return f"{round(temp)}°F, {cond}"
    except Exception:
        return ""
