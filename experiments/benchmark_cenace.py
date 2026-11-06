"""Comparar analog-holidays contra el pronóstico operativo propio de CENACE.

Hasta ahora el método sólo se había medido contra líneas base reconstruidas
—naive estacional y Similar-Days—. La base PML guarda el pronóstico que CENACE
publica junto con su demanda real (`RESULTS.model = 40`, alimentado del archivo
diario `demand_AAAAMMDD.csv`, columna `pronostico_mw`), lo que permite medir el
método contra el incumbente en el sistema real.

Dos mediciones:
  1. Pareada sobre los festivos que caen en el tramo con pronóstico CENACE.
  2. Contexto: el error de CENACE en festivos frente a días ordinarios, que es
     la motivación empírica de tratar los festivos aparte.

⚠️ Convención horaria: la base etiqueta inicio de intervalo y el CSV del repo
fin de intervalo; hay que sumar una hora al índice de la base (ver
`sync_demand_from_pml.py`). Sin ese ajuste el error medido sube ~3 pp por puro
desfase.

⚠️ El horizonte de emisión del pronóstico CENACE no está registrado en la base.
Es el que CENACE publica en su archivo diario de demanda, o sea su pronóstico
operativo del día; es la referencia más cercana disponible a nuestra emisión a
D-1, pero no está verificado que coincida hora por hora.

Uso:  /usr/bin/python3 analog_holidays/experiments/benchmark_cenace.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/home/uriel/GIT/analog_holidays")
PML_DB = Path("/home/uriel/GIT/pml/bd/bd_pml.db")
CHAMPION = ROOT / "experiments" / "experiment_2026_08_25_00_37_criterion_holiday_identity"
CENACE_MODEL = 40


def load_cenace() -> pd.DataFrame:
    with sqlite3.connect(f"file:{PML_DB}?mode=ro", uri=True) as con:
        q = f"""SELECT t.unique_id, r.ds, r.y
                FROM RESULTS r JOIN TIMESERIES t ON t.uid = r.uid
                WHERE r.model = {CENACE_MODEL} AND r.uid BETWEEN 17 AND 24"""
        df = pd.read_sql(q, con)
    df["ds"] = pd.to_datetime(df["ds"]) + pd.Timedelta(hours=1)   # inicio → fin de intervalo
    return df.pivot_table(index="ds", columns="unique_id", values="y", aggfunc="last")


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    ok = np.isfinite(actual) & np.isfinite(forecast) & (np.abs(actual) > 1e-9)
    if ok.sum() < 20:                      # exigir el día casi completo
        return np.nan
    return float(np.mean(np.abs(forecast[ok] - actual[ok]) / np.abs(actual[ok])) * 100)


def main() -> None:
    cen = load_cenace()
    demand = pd.read_csv(ROOT / "holidays" / "holiday_demand_mx.csv", parse_dates=["ds"])
    demand = demand.set_index("ds").sort_index()
    series = [c for c in cen.columns if c in demand.columns]
    flags = {s: f"{s}_holiday" for s in series}

    lo, hi = cen.index.min(), cen.index.max()
    print(f"pronóstico CENACE: {lo} → {hi}  ({len(cen)} horas, {len(series)} series)\n")

    # ── 1. Pareado en festivos ────────────────────────────────────────────────
    met = pd.read_csv(CHAMPION / "metrics.csv", parse_dates=["target_date"])
    met = met[(met.target_date >= lo.normalize()) & (met.target_date <= hi.normalize())]
    dates = sorted(met["target_date"].unique())
    print(f"festivos evaluables en el tramo: {len(dates)} "
          f"({', '.join(pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates)})\n")

    rows = []
    for _, r in met.iterrows():
        uid, day = r["unique_id"], r["target_date"].normalize()
        if uid not in series:
            continue
        hours = pd.date_range(day + pd.Timedelta(hours=1), day + pd.Timedelta(hours=24), freq="h")
        a = demand[uid].reindex(hours).to_numpy(dtype=float)
        c = cen[uid].reindex(hours).to_numpy(dtype=float)
        rows.append({"unique_id": uid, "target_date": day.date(),
                     "holiday": r["holiday_label"],
                     "analog": r["mape_24_pct"], "cenace": mape(a, c)})
    cells = pd.DataFrame(rows).dropna()
    print(f"=== 1. PAREADO EN FESTIVOS ({len(cells)} celdas) ===")
    print(f"  analog-holidays  mediana={cells.analog.median():6.3f} %  media={cells.analog.mean():6.3f} %")
    print(f"  CENACE           mediana={cells.cenace.median():6.3f} %  media={cells.cenace.mean():6.3f} %")
    if len(cells) >= 6:
        stat, p = wilcoxon(cells.analog, cells.cenace)
        skill = 1 - cells.analog.median() / cells.cenace.median()
        print(f"  analog gana {(cells.analog < cells.cenace).mean():.1%} de las celdas | "
              f"delta mediano={np.median(cells.analog - cells.cenace):+.3f} pp | "
              f"skill={skill:+.3f} | p={p:.4f}")
    print()
    print(cells.pivot_table(index="target_date", columns=None,
                            values=["analog", "cenace"], aggfunc="median").round(3).to_string())
    print()
    print(cells.groupby("unique_id")[["analog", "cenace"]].median().round(3).to_string())

    # ── 2. CENACE: festivos vs días ordinarios ────────────────────────────────
    print("\n=== 2. CONTEXTO: error de CENACE, festivo vs día ordinario ===")
    ctx = []
    for uid in series:
        sub = demand.loc[lo:hi, [uid, flags[uid]]].copy()
        sub["cen"] = cen[uid].reindex(sub.index)
        sub["day"] = sub.index.normalize()
        for day, g in sub.groupby("day"):
            m = mape(g[uid].to_numpy(float), g["cen"].to_numpy(float))
            if np.isfinite(m):
                ctx.append({"unique_id": uid, "day": day,
                            "holiday": bool(g[flags[uid]].max() == 1), "mape": m})
    ctx = pd.DataFrame(ctx)
    g = ctx.groupby("holiday")["mape"].agg(["size", "median", "mean"]).round(3)
    g.index = ["día ordinario", "festivo"]
    print(g.to_string())
    fest, ord_ = ctx[ctx.holiday]["mape"], ctx[~ctx.holiday]["mape"]
    if len(fest) > 5:
        print(f"\n  el error de CENACE es {fest.median()/ord_.median():.2f}× mayor en festivos "
              f"({fest.median():.3f} % vs {ord_.median():.3f} %)")

    out = ROOT / "docs" / "benchmark_cenace_cells.csv"
    cells.to_csv(out, index=False)
    ctx.to_csv(ROOT / "docs" / "benchmark_cenace_daily.csv", index=False)
    print(f"\n-> {out.name} y benchmark_cenace_daily.csv")


if __name__ == "__main__":
    main()
