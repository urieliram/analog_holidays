"""¿La temperatura explica el sesgo de nivel que deja el método analog?

Contexto: el 89 % del error es sesgo de nivel, no de forma, y en ERCOT el 58 %
de la varianza de ese sesgo es un choque del mismo día común a todo el sistema
--- la firma de un evento climático. Este script pone a prueba esa hipótesis
con temperatura OBSERVADA, es decir, en el caso más favorable posible: si ni
con información de oráculo el clima explica el sesgo, con pronóstico tampoco.

Convención de signo: mpe = (real - pronóstico)/real, así que MPE > 0 significa
sub-pronóstico. Una ola de calor debería empujar el MPE hacia arriba.

Uso:  MPLBACKEND=Agg python3 analog_holidays/experiments/analyze_weather_bias.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "experiments" / "experiment_2026_08_31_06_38_ercot_criterion_holiday_identity"
BALANCE_POINT = 18.3          # °C, ~65 °F, punto de balance estándar de la industria
HEAD_HOURS = 14               # la ventana previa que el modelo sí observa


def load_weather() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "holidays" / "weather_ercot.csv", parse_dates=["ds"])
    return df.set_index("ds").sort_index()


def daily_frames(hourly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Media diaria, grados-día y anomalía frente a la climatología del día del año."""
    out = {}
    daily_mean = hourly.resample("D").mean()
    # Grados-día: la demanda sube tanto por frío como por calor, así que la
    # distancia al punto de balance predice carga mejor que la temperatura cruda.
    degree_days = (hourly - BALANCE_POINT).abs().resample("D").mean()

    for name, frame in (("temp", daily_mean), ("dd", degree_days)):
        doy = frame.index.dayofyear
        # Climatología suavizada: media por día del año con ventana circular de
        # +/-7 días, para que no la domine el ruido de un solo año.
        clim = frame.groupby(doy).mean()
        clim = pd.concat([clim, clim, clim]).rolling(15, center=True).mean().iloc[len(clim):2 * len(clim)]
        anom = frame - clim.reindex(doy).to_numpy()
        out[name] = frame
        out[f"{name}_anom"] = anom
    return out


def main() -> None:
    hourly = load_weather()
    frames = daily_frames(hourly)

    met = pd.read_csv(CHAMPION / "metrics.csv", parse_dates=["target_date", "forecast_start"])
    met = met[met["fail"].fillna(False).astype(str).str.lower().isin(["false", "nan", ""])]
    met = met.dropna(subset=["mpe_24_pct"])

    rows = []
    for _, r in met.iterrows():
        zone, day = r["unique_id"], r["target_date"].normalize()
        if zone not in hourly.columns:
            continue
        rec = {
            "unique_id": zone, "target_date": day, "holiday": r["holiday_label"],
            "mpe_24": r["mpe_24_pct"], "mape_24": r["mape_24_pct"],
        }
        for key in ("temp", "temp_anom", "dd", "dd_anom"):
            f = frames[key]
            rec[key] = f.at[day, zone] if day in f.index else np.nan

        # Salto térmico entre la ventana observada y el festivo: el modelo ancla
        # su nivel en las horas previas, así que un cambio de temperatura entre
        # ambas es información que no puede tener.
        start = r["forecast_start"]
        head = hourly.loc[start:start + pd.Timedelta(hours=HEAD_HOURS - 1), zone]
        hol = hourly.loc[day:day + pd.Timedelta(hours=23), zone]
        if len(head) and len(hol):
            rec["delta_head_to_holiday"] = hol.mean() - head.mean()
            rec["delta_dd"] = ((hol - BALANCE_POINT).abs().mean()
                               - (head - BALANCE_POINT).abs().mean())
        rows.append(rec)

    d = pd.DataFrame(rows).dropna(subset=["temp_anom"])
    print(f"celdas con clima emparejado: {len(d)} de {len(met)}\n")

    def report(title: str, frame: pd.DataFrame, target: str, preds: list[str]) -> None:
        print(f"=== {title}  (n={len(frame)}) ===")
        for p in preds:
            sub = frame[[target, p]].dropna()
            if len(sub) < 5:
                continue
            r = sub[target].corr(sub[p])
            print(f"  {p:24s} r={r:+.3f}   R2={r**2:6.1%}")
        print()

    preds = ["temp", "temp_anom", "dd", "dd_anom", "delta_head_to_holiday", "delta_dd"]
    report("Por celda: MPE_24 (sesgo con signo)", d, "mpe_24", preds)
    report("Por celda: |MPE_24| (magnitud del sesgo)",
           d.assign(abs_mpe=d["mpe_24"].abs()), "abs_mpe", preds)

    # El choque común: promedio entre zonas por fecha. Es el componente que
    # sabemos que domina (58 % de la varianza) y el candidato natural a ser clima.
    sysd = d.groupby("target_date").agg(
        mpe_24=("mpe_24", "mean"), temp_anom=("temp_anom", "mean"),
        dd_anom=("dd_anom", "mean"), delta_dd=("delta_dd", "mean"),
        delta_head_to_holiday=("delta_head_to_holiday", "mean"),
        temp=("temp", "mean"), dd=("dd", "mean"))
    report("Choque común del sistema: MPE medio entre zonas, por fecha",
           sysd, "mpe_24", preds)

    print("=== Detalle por fecha (choque del sistema) ===")
    view = sysd.join(d.groupby("target_date")["holiday"].first())
    view = view[["holiday", "mpe_24", "temp_anom", "dd_anom", "delta_dd"]]
    print(view.sort_values("mpe_24").round(2).to_string())

    d.to_csv(ROOT / "docs" / "weather_bias_cells.csv", index=False)
    print(f"\n-> docs/weather_bias_cells.csv ({len(d)} celdas)")


if __name__ == "__main__":
    main()
