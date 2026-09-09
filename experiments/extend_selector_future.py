"""Extender el selector de festivos hacia adelante, sin fuga de información.

`holidays/holiday_selector_features.csv` sólo cubre los festivos ya ocurridos y
perfilados. Para pronosticar un festivo futuro hace falta su fila de selector,
pero no puede construirse igual: los campos de perfil (best_matching_weekday,
clusters de perfil diario y de evento) se derivan del propio día, que todavía no
ha pasado.

`build_future_holiday_selector_features` resuelve eso rellenando esos campos
desde los *priors* históricos del mismo festivo, así que la fila futura queda
libre de fuga. Este script la aplica por serie y añade sólo las fechas nuevas.

Uso:  /usr/bin/python3 analog_holidays/experiments/extend_selector_future.py [--end 2026-12-31]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/uriel/GIT")

from analog_holidays.shared.identify_holidays import (  # noqa: E402
    build_future_holiday_selector_features,
    load_holidays_catalog,
)

HOL = Path("/home/uriel/GIT/analog_holidays/holidays")
SELECTOR = HOL / "holiday_selector_features.csv"
PRIORS = HOL / "holiday_selector_priors.csv"
DEMAND = HOL / "holiday_demand_mx.csv"
# El catálogo MX vive en el proyecto pml; aquí está ignorado por .gitignore (*.json).
CATALOG = Path("/home/uriel/GIT/pml/analog_holidays/holidays_recognized.json")
GROUP_COLS = ("unique_id", "anchor_holiday_name", "holiday_day_type")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CATALOG.exists():
        raise SystemExit(f"falta el catálogo de festivos: {CATALOG}")

    selector = pd.read_csv(SELECTOR, parse_dates=["date"])
    priors = pd.read_csv(PRIORS)
    demand = pd.read_csv(DEMAND, parse_dates=["ds"])
    series = [c for c in demand.columns if c != "ds" and not c.endswith("_holiday")]

    last = selector["date"].max().normalize()
    start = last + pd.Timedelta(days=1)
    end = pd.Timestamp(args.end).normalize()
    print(f"selector actual: {len(selector)} filas, hasta {last.date()}")
    print(f"extendiendo {start.date()} → {end.date()}\n")

    catalog = load_holidays_catalog(CATALOG, 2020, int(end.year))

    frames = []
    for uid in series:
        # Todas las fechas presentes en la tabla horaria, no sólo las observadas:
        # el horizonte de pronóstico ya existe como calendario aunque su demanda
        # aún esté vacía, y es precisamente para esas filas que sirve esta función.
        available = demand["ds"].dt.normalize().unique()
        future = build_future_holiday_selector_features(
            df_holidays=catalog, df_priors=priors, available_dates=available,
            holidays_path=CATALOG, group_cols=GROUP_COLS,
            start_date=start, end_date=end, unique_id=uid,
        )
        print(f"  {uid:18s} filas futuras={len(future)}")
        frames.append(future)

    if not frames or all(f.empty for f in frames):
        raise SystemExit("no se generó ninguna fila futura")

    future_all = pd.concat(frames, ignore_index=True)
    merged = (pd.concat([selector, future_all], ignore_index=True)
                .drop_duplicates(subset=["unique_id", "date", "holiday_name"], keep="first")
                .sort_values(["unique_id", "date", "holiday_name"])
                .reset_index(drop=True))

    added = len(merged) - len(selector)
    print(f"\nfilas añadidas: {added}  (total {len(merged)})")
    nuevas = merged[merged["date"] > last]
    if not nuevas.empty:
        print(nuevas[["date", "holiday_name", "anchor_holiday_name", "holiday_day_type"]]
              .drop_duplicates().to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: no se escribe nada.")
        return

    backup = SELECTOR.with_suffix(f".csv.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(SELECTOR, backup)
    merged.to_csv(SELECTOR, index=False)
    print(f"\nrespaldo: {backup.name}\n-> {SELECTOR}")


if __name__ == "__main__":
    main()
