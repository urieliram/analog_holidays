"""¿Cuánta señal climática sobrevive al usar pronóstico en vez de observación?

`analyze_weather_bias.py` mostró que la anomalía de grados-día explica ~18-25 %
de la varianza del sesgo, pero usando temperatura OBSERVADA, que no existe a la
hora de emisión. Aquí se repite exactamente la misma medición sustituyendo la
temperatura del día festivo por el pronóstico archivado de D-1 (y de D-2 como
cota conservadora). La climatología sigue viniendo del histórico observado: es
conocida de antemano, así que no contamina el ejercicio.

Uso:  MPLBACKEND=Agg python3 analog_holidays/experiments/analyze_weather_exante.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "experiments" / "experiment_2026_08_31_06_38_ercot_criterion_holiday_identity"
BALANCE_POINT = 18.3


def climatology(observed: pd.DataFrame) -> pd.DataFrame:
    """Grados-día normales por día del año, suavizados +/-7 días."""
    dd = (observed - BALANCE_POINT).abs().resample("D").mean()
    clim = dd.groupby(dd.index.dayofyear).mean()
    return (pd.concat([clim, clim, clim]).rolling(15, center=True).mean()
              .iloc[len(clim):2 * len(clim)])


def main() -> None:
    obs = pd.read_csv(ROOT / "holidays" / "weather_ercot.csv",
                      parse_dates=["ds"]).set_index("ds").sort_index()
    clim = climatology(obs)
    obs_dd = (obs - BALANCE_POINT).abs().resample("D").mean()

    fc = pd.read_csv(ROOT / "holidays" / "weather_forecast_ercot.csv", parse_dates=["ds"])
    fc["day"] = fc["ds"].dt.normalize()
    fc["dd"] = (fc["temp"] - BALANCE_POINT).abs()
    fc_dd = (fc.groupby(["lead", "unique_id", "day"])["dd"].mean().unstack("lead"))

    met = pd.read_csv(CHAMPION / "metrics.csv", parse_dates=["target_date"])
    met = met.dropna(subset=["mpe_24_pct"])

    rows = []
    for _, r in met.iterrows():
        zone, day = r["unique_id"], r["target_date"].normalize()
        if zone not in obs.columns or day not in obs_dd.index:
            continue
        normal = clim.at[day.dayofyear, zone]
        rec = {"unique_id": zone, "target_date": day, "holiday": r["holiday_label"],
               "mpe_24": r["mpe_24_pct"],
               "dd_anom_obs": obs_dd.at[day, zone] - normal,
               "dd_obs": obs_dd.at[day, zone]}
        for lead in ("previous_day1", "previous_day2"):
            key = (zone, day)
            val = fc_dd[lead].get(key, np.nan) if key in fc_dd.index else np.nan
            rec[f"dd_anom_{lead}"] = val - normal
            rec[f"dd_{lead}"] = val
        rows.append(rec)

    d = pd.DataFrame(rows)
    print(f"celdas: {len(d)}  |  con pronóstico D-1: {d['dd_anom_previous_day1'].notna().sum()}\n")

    # 1. ¿Qué tan bueno es el pronóstico meteorológico en sí?
    print("=== Calidad del pronóstico de temperatura en los días objetivo ===")
    for lead in ("previous_day1", "previous_day2"):
        sub = d[["dd_obs", f"dd_{lead}"]].dropna()
        err = sub[f"dd_{lead}"] - sub["dd_obs"]
        r = sub["dd_obs"].corr(sub[f"dd_{lead}"])
        print(f"  {lead:15s} MAE={err.abs().mean():.2f} grados-dia  sesgo={err.mean():+.2f}  "
              f"corr con observado r={r:.3f}")

    # 2. La medición que importa: señal sobre el sesgo, ex ante vs oráculo.
    print("\n=== Varianza del sesgo explicada (MPE_24) ===")
    print(f"{'predictor':28s} {'n':>4s} {'Pearson':>9s} {'R2':>7s} {'Spearman':>9s} {'R2':>7s} {'p':>10s}")
    variants = [("dd_anom_obs", "observado (ORÁCULO)"),
                ("dd_anom_previous_day1", "pronóstico D-1 (operativo)"),
                ("dd_anom_previous_day2", "pronóstico D-2 (conservador)")]

    def line(frame: pd.DataFrame, col: str, label: str) -> None:
        sub = frame[["mpe_24", col]].dropna()
        if len(sub) < 5:
            print(f"{label:28s} {len(sub):>4d}   (datos insuficientes)")
            return
        pr, _ = stats.pearsonr(sub[col], sub["mpe_24"])
        sr, sp = stats.spearmanr(sub[col], sub["mpe_24"])
        print(f"{label:28s} {len(sub):>4d} {pr:>+9.3f} {pr**2:>6.1%} {sr:>+9.3f} {sr**2:>6.1%} {sp:>10.2e}")

    print("-- por celda --")
    for col, label in variants:
        line(d, col, label)

    sysd = d.groupby("target_date").mean(numeric_only=True)
    print("-- choque común del sistema (media entre zonas por fecha) --")
    for col, label in variants:
        line(sysd, col, label)

    print("-- choque común, excluyendo la helada MLK-2025 (robustez) --")
    for col, label in variants:
        line(sysd.drop(pd.Timestamp("2025-01-20"), errors="ignore"), col, label)

    d.to_csv(ROOT / "docs" / "weather_exante_cells.csv", index=False)
    print(f"\n-> docs/weather_exante_cells.csv ({len(d)} celdas)")


if __name__ == "__main__":
    main()
