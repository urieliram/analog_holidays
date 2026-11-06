"""Descarga el PRONÓSTICO archivado de temperatura para las weather zones de ERCOT.

A diferencia de `fetch_weather_ercot.py`, que baja lo observado (ERA5), este
script baja lo que el modelo meteorológico predijo *en su momento*: es la única
variante admisible dentro del método, porque el pronóstico de demanda se emite
a las 10:00 del D-1 y alimenta precios, planeación y despacho.

Se traen dos horizontes:
  previous_day1 -> corrida de ~1 día antes. Es el caso operativo: a las 10:00
                   del D-1 ya está disponible.
  previous_day2 -> corrida de ~2 días antes. Cota conservadora, por si alguien
                   objeta que la corrida del D-1 llegó demasiado tarde.

Uso:  MPLBACKEND=Agg python3 analog_holidays/experiments/fetch_weather_forecast_ercot.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from fetch_weather_ercot import ZONE_POINTS  # noqa: E402  (mismo paquete de puntos)

HOL = Path(__file__).resolve().parent.parent / "holidays"
OUT = HOL / "weather_forecast_ercot.csv"

# Sólo el periodo del panel: el archivo de pronósticos no cubre 2016.
START, END = "2024-12-01", "2026-04-01"
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
TZ = "America/Chicago"
LEADS = ("previous_day1", "previous_day2")


def fetch_point(lat: float, lon: float, retries: int = 4) -> pd.DataFrame:
    variables = ",".join(f"temperature_2m_{lead}" for lead in LEADS)
    url = (f"{API}?latitude={lat}&longitude={lon}"
           f"&start_date={START}&end_date={END}"
           f"&hourly={variables}&timezone={TZ}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                payload = json.load(resp)
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"    reintento {attempt + 1} tras {type(exc).__name__} ({wait}s)", flush=True)
            time.sleep(wait)
    hourly = payload["hourly"]
    return pd.DataFrame(
        {lead: hourly[f"temperature_2m_{lead}"] for lead in LEADS},
        index=pd.to_datetime(hourly["time"]),
    ).astype("float64")


def main() -> None:
    per_lead: dict[str, dict[str, pd.Series]] = {lead: {} for lead in LEADS}

    for zone, points in ZONE_POINTS.items():
        print(f"{zone}: {len(points)} punto(s)", flush=True)
        acc = {lead: None for lead in LEADS}
        total_w = 0.0
        for lat, lon, weight in points:
            frame = fetch_point(lat, lon)
            for lead in LEADS:
                contrib = frame[lead] * weight
                acc[lead] = contrib if acc[lead] is None else acc[lead].add(contrib, fill_value=0.0)
            total_w += weight
            cov = {lead: int(frame[lead].notna().sum()) for lead in LEADS}
            print(f"    ({lat},{lon}) w={weight} -> cobertura {cov}", flush=True)
            time.sleep(1.5)
        for lead in LEADS:
            per_lead[lead][f"ERCOT_demand_{zone}"] = acc[lead] / total_w

    demand = pd.read_csv(HOL / "holiday_demand_ercot.csv", parse_dates=["ds"])
    frames = []
    for lead in LEADS:
        df = pd.DataFrame(per_lead[lead]).sort_index()
        loads = {c: demand[c].mean() for c in df.columns if c in demand.columns}
        total_load = sum(loads.values())
        df["ERCOT_demand_ERCOT"] = sum(df[c] * (w / total_load) for c, w in loads.items())
        df = df.stack(dropna=False).rename("temp").reset_index()
        df.columns = ["ds", "unique_id", "temp"]
        df["lead"] = lead
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.round(3).to_csv(OUT, index=False)
    print(f"\n-> {OUT}  ({len(out)} filas)", flush=True)
    print(out.groupby("lead")["temp"].agg(["count", "mean", "min", "max"]).round(2), flush=True)


if __name__ == "__main__":
    main()
