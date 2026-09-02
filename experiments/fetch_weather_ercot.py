"""Descarga temperatura horaria ERA5 para las ocho weather zones de ERCOT.

Las zonas de ERCOT son regiones climáticas, no comerciales, así que cada una
admite una temperatura representativa. La carga sigue a la población, no a la
geometría, de modo que cada zona se resume con varios puntos urbanos ponderados
por población en lugar de con su centroide.

Fuente: reanálisis ERA5 vía Open-Meteo (sin API key). Es la temperatura
OBSERVADA, no la pronosticada: sirve para diagnosticar si el clima explica el
sesgo, pero NO es admisible dentro de un método operativo, porque no está
disponible a la hora de emisión. Para eso existe `historical-forecast-api`.

Uso:  MPLBACKEND=Agg python3 analog_holidays/experiments/fetch_weather_ercot.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HOL = Path(__file__).resolve().parent.parent / "holidays"
OUT = HOL / "weather_ercot.csv"

START, END = "2016-01-01", "2026-04-01"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/Chicago"

# (lat, lon, peso). Los pesos aproximan la población de cada zona; sólo tienen
# que ser correctos en orden de magnitud para que el promedio siga a la carga.
ZONE_POINTS: dict[str, list[tuple[float, float, float]]] = {
    "COAST": [(29.76, -95.37, 0.85), (30.08, -94.10, 0.15)],          # Houston, Beaumont
    "EAST": [(32.35, -95.30, 0.60), (32.50, -94.74, 0.40)],           # Tyler, Longview
    "FWEST": [(31.997, -102.078, 0.60), (31.845, -102.368, 0.40)],    # Midland, Odessa
    "NORTH": [(33.91, -98.49, 1.00)],                                 # Wichita Falls
    "NCENT": [(32.78, -96.80, 0.55), (32.755, -97.33, 0.35),
              (31.55, -97.15, 0.10)],                                 # Dallas, Fort Worth, Waco
    "SOUTH": [(26.20, -98.23, 0.45), (27.80, -97.40, 0.35),
              (27.53, -99.49, 0.20)],                                 # McAllen, Corpus, Laredo
    "SCENT": [(29.42, -98.49, 0.50), (30.27, -97.74, 0.50)],          # San Antonio, Austin
    "WEST": [(32.45, -99.73, 0.55), (31.46, -100.44, 0.45)],          # Abilene, San Angelo
}


def fetch_point(lat: float, lon: float, retries: int = 4) -> pd.Series:
    """Temperatura horaria de un punto, indexada por hora local de Texas."""
    url = (f"{ARCHIVE}?latitude={lat}&longitude={lon}"
           f"&start_date={START}&end_date={END}"
           f"&hourly=temperature_2m&timezone={TZ}")
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
    return pd.Series(
        hourly["temperature_2m"],
        index=pd.to_datetime(hourly["time"]),
        name=f"{lat},{lon}",
        dtype="float64",
    )


def main() -> None:
    zone_series: dict[str, pd.Series] = {}
    for zone, points in ZONE_POINTS.items():
        print(f"{zone}: {len(points)} punto(s)", flush=True)
        acc, total_w = None, 0.0
        for lat, lon, weight in points:
            serie = fetch_point(lat, lon)
            contrib = serie * weight
            acc = contrib if acc is None else acc.add(contrib, fill_value=0.0)
            total_w += weight
            print(f"    ({lat},{lon}) w={weight} -> {len(serie)} h, "
                  f"media {serie.mean():.1f} C", flush=True)
            time.sleep(1.5)          # cortesía con la API pública
        zone_series[f"ERCOT_demand_{zone}"] = acc / total_w

    df = pd.DataFrame(zone_series).sort_index()
    df.index.name = "ds"

    # La zona agregada ERCOT se pondera por la demanda media de cada zona, que
    # es lo que realmente determina la temperatura que "siente" el sistema.
    demand = pd.read_csv(HOL / "holiday_demand_ercot.csv", parse_dates=["ds"])
    zone_cols = [c for c in df.columns]
    loads = {c: demand[c].mean() for c in zone_cols if c in demand.columns}
    total_load = sum(loads.values())
    df["ERCOT_demand_ERCOT"] = sum(df[c] * (w / total_load) for c, w in loads.items())

    df.round(3).to_csv(OUT)
    print(f"\n-> {OUT}  ({len(df)} horas x {df.shape[1]} zonas)", flush=True)
    print(df.describe().T[["mean", "min", "max"]].round(1), flush=True)


if __name__ == "__main__":
    main()
