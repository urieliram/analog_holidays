"""Rellenar la demanda observada del CSV de festivos desde la base PML.

`holidays/holiday_demand_mx.csv` ya trae el calendario y las banderas de día
especial hasta 2027, pero la columna de demanda se quedó en 2026-05-17. El
observado vive en `pml/bd/bd_pml.db`, tabla RESULTS con `model = 0` ("actual"),
para los uid 17–24 (SIN + las siete gerencias), con los mismos `unique_id`.

**Sólo rellena, nunca reescribe.** El histórico ya validado no se toca, y las
exclusiones deliberadas —todo 2022, más 2020-04-06 y 2021-04-05, contaminados
por comportamiento de cuarentena— se preservan porque el script no escribe una
sola fila anterior al último dato existente. La base PML *sí* tiene 2022; si se
copiara completa se reintroduciría justo lo que se excluyó a propósito.

Uso:  /usr/bin/python3 analog_holidays/experiments/sync_demand_from_pml.py [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/home/uriel/GIT/analog_holidays")
CSV = ROOT / "holidays" / "holiday_demand_mx.csv"
PML_DB = Path("/home/uriel/GIT/pml/bd/bd_pml.db")
ACTUAL_MODEL = 0          # CAT_MODEL: 0 = "actual"
DEMAND_UIDS = range(17, 25)


def read_pml_actuals() -> pd.DataFrame:
    if not PML_DB.exists():
        raise SystemExit(f"no encuentro la base PML: {PML_DB}")
    uri = f"file:{PML_DB}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        q = f"""
            SELECT t.unique_id, r.ds, r.y
            FROM RESULTS r
            JOIN TIMESERIES t ON t.uid = r.uid
            WHERE r.model = {ACTUAL_MODEL}
              AND r.uid BETWEEN {DEMAND_UIDS.start} AND {DEMAND_UIDS.stop - 1}
        """
        df = pd.read_sql(q, con)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.pivot_table(index="ds", columns="unique_id", values="y", aggfunc="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv = pd.read_csv(CSV, parse_dates=["ds"]).set_index("ds").sort_index()
    demand_cols = [c for c in csv.columns if not c.endswith("_holiday")]

    observed = csv[demand_cols].dropna(how="all")
    cutoff = observed.index.max()
    print(f"CSV: último dato observado {cutoff}  ({len(observed)} horas)")

    pml = read_pml_actuals()
    print(f"PML: {pml.index.min()} → {pml.index.max()}  ({len(pml)} horas, "
          f"{pml.shape[1]} series)")

    missing = [c for c in demand_cols if c not in pml.columns]
    if missing:
        raise SystemExit(f"la base PML no trae estas series: {missing}")

    # Sólo lo posterior al corte: el histórico validado y sus exclusiones
    # deliberadas quedan intactos por construcción.
    new = pml.loc[pml.index > cutoff, demand_cols]
    new = new.reindex(columns=demand_cols)
    if new.empty:
        print("nada nuevo que agregar.")
        return

    target = csv.index.intersection(new.index)
    outside = new.index.difference(csv.index)
    print(f"\nfilas nuevas: {len(new)}  ({new.index.min()} → {new.index.max()})")
    print(f"  dentro del calendario del CSV: {len(target)}")
    if len(outside):
        print(f"  ⚠️ fuera del calendario del CSV (se ignoran): {len(outside)} "
              f"({outside.min()} → {outside.max()})")

    if args.dry_run:
        print("\n--dry-run: no se escribe nada.")
        return

    backup = CSV.with_suffix(f".csv.bak_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(CSV, backup)
    print(f"\nrespaldo: {backup.name}")

    before = csv[demand_cols].notna().sum().sum()
    csv.loc[target, demand_cols] = new.loc[target, demand_cols].to_numpy()
    after = csv[demand_cols].notna().sum().sum()

    csv.reset_index().to_csv(CSV, index=False)
    still = csv[demand_cols].dropna(how="all").index.max()
    print(f"celdas de demanda: {before:,} → {after:,}  (+{after - before:,})")
    print(f"nuevo último dato observado: {still}")


if __name__ == "__main__":
    main()
