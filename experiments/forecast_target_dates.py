"""Emitir el pronóstico horario de días festivos futuros como CSV entregable.

A diferencia de los runners de experimento, que evalúan el método contra días ya
ocurridos y guardan métricas, este script produce el artefacto operativo: el
perfil horario pronosticado, una columna por gerencia regional.

Cada corrida entrega la ventana de 38 h del método:
    14 h del día previo (desde las 10:00) + las 24 h del festivo.

Configuración: la campeona documentada en `docs/RESULTADO_TECHO.md`
(`holiday_identity`, sin tope de k, búsqueda PCR/PLS/Ridge/Lasso).

**Verificación previa.** El método ancla el pronóstico en la ventana observada
inmediatamente anterior a la hora de emisión (`serie[-38:]`). Si la serie de
demanda no llega hasta esa hora, el runner *no falla*: silenciosamente empareja
la última ventana disponible —de meses atrás— contra los análogos históricos y
devuelve números sin sentido. Por eso aquí se aborta antes de calcular nada.

Uso:
    MPLBACKEND=Agg python3 analog_holidays/experiments/forecast_target_dates.py \
        --targets 2026-09-16:"Independence Day" \
        --out docs/pronostico_2026_09.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, "/home/uriel/GIT")

import run_deep_regression_sweep as base  # noqa: E402
from analog_holidays.analog.analog_holidays import (  # noqa: E402
    run_analog_holidays,
    tune_analog_holidays_optuna,
)

SELECTORS = EXPERIMENTS / "selectors"
SEARCH = ["PCR", "PLS", "RidgeReg", "LassoReg"]
CRITERION = "holiday_identity"

# Etiquetas cortas pedidas en el entregable, en el orden de las gerencias.
REGION_LABEL = {
    "SEN_demand_CEL": "CEL", "SEN_demand_ORI": "ORI", "SEN_demand_OCC": "OCC",
    "SEN_demand_NOR": "NOR", "SEN_demand_NES": "NES", "SEN_demand_NTE": "NTE",
    "SEN_demand_PEN": "PEN", "SEN_demand_SIN": "SIN",
}
COLUMN_ORDER = ["CEL", "ORI", "OCC", "NOR", "NES", "NTE", "PEN", "SIN"]


def rebind_selector(criterion: str) -> Path:
    path = SELECTORS / f"selector_{criterion}.csv"
    if not path.exists():
        raise SystemExit(f"falta el selector {criterion!r}: {path}\n"
                         f"genéralo con: python3 {EXPERIMENTS}/gen_selectors.py")
    base.SELECTOR_FEATURES_PATH = path
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["unique_id"] = df["unique_id"].astype(str)
    base.selector_features_df = df

    col = base.CLUSTER_COLUMN
    base.selector_cluster_lookup_by_id = {
        uid: (df.loc[df.unique_id == uid].dropna(subset=[col])
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")[col].to_dict())
        for uid in base.series_unique_ids}
    base.selector_anchor_lookup_by_id = {
        uid: (df.loc[df.unique_id == uid].dropna(subset=["anchor_holiday_name"])
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")["anchor_holiday_name"].to_dict())
        for uid in base.series_unique_ids}
    return path


def preflight(targets: list[tuple[str, str]], selector_path: Path) -> None:
    """Abortar si falta el dato observado o la etiqueta de festivo del objetivo."""
    demand = pd.read_csv(base.SOURCE_PATH, parse_dates=["ds"])
    selector = pd.read_csv(selector_path, parse_dates=["date"])
    problems: list[str] = []

    for uid in base.series_unique_ids:
        if uid not in demand.columns:
            problems.append(f"la serie {uid} no existe en {base.SOURCE_PATH.name}")
            continue
        last = demand.loc[demand[uid].notna(), "ds"].max()
        for target, _label in targets:
            issue = pd.Timestamp(target) - pd.Timedelta(hours=base.FORECAST_START_OFFSET_HOURS)
            if pd.isna(last) or last < issue:
                problems.append(
                    f"{uid}: la demanda observada termina {last}, pero el pronóstico de "
                    f"{target} se emite {issue} y necesita las 38 h previas "
                    f"({issue - pd.Timedelta(hours=base.SEASON_LENGTH)} → {issue}). "
                    f"Faltan {(issue - last).days} días de dato.")
                break

    for target, label in targets:
        ts = pd.Timestamp(target).normalize()
        hit = selector[selector["date"] == ts]
        if hit.empty:
            problems.append(
                f"{target} ({label}) no está en el selector {selector_path.name}; "
                f"la última fecha que contiene es {selector['date'].max().date()}. "
                f"Sin etiqueta de cluster no hay pool de análogos.")

    if problems:
        print("\n".join(f"  ✗ {p}" for p in dict.fromkeys(problems)), file=sys.stderr)
        raise SystemExit(
            "\nNo se puede emitir el pronóstico: falta información de entrada.\n"
            "El método empareja la ventana observada previa a la emisión contra los\n"
            "análogos históricos; sin ese dato el resultado sería espurio, no aproximado.")


def forecast_one(uid: str, target: str, label: str) -> pd.Series:
    """Perfil horario de 38 h para una serie y un festivo, config campeona."""
    ts = pd.Timestamp(target).normalize()
    cluster = base.selector_cluster_lookup_by_id.get(uid, {}).get(ts, pd.NA)
    min_k = base.OPTUNA_MIN_K_BY_CLUSTER.get(str(cluster), base.OPTUNA_MIN_K)

    tuning = tune_analog_holidays_optuna(
        unique_id=uid, source_path=base.SOURCE_PATH, train_end=ts,
        season_length=base.SEASON_LENGTH,
        forecast_start_offset_hours=base.FORECAST_START_OFFSET_HOURS,
        initial_k=base.K, initial_typedist=base.TYPEDIST, initial_typereg=base.TYPEREG,
        typedist_choices=base.OPTUNA_TYPEDIST_CHOICES, typereg_choices=SEARCH,
        scale_method=base.SCALE_METHOD, scale_method_choices=base.OPTUNA_SCALE_METHOD_CHOICES,
        initial_n_components=base.N_COMPONENTS, initial_regressor_params=base.REGRESSOR_PARAMS,
        optuna_min_k=min_k, optuna_max_k=None,
        n_trials=base.OPTUNA_N_TRIALS, timeout_sec=base.OPTUNA_TIMEOUT_SEC,
        max_eval_dates=base.OPTUNA_MAX_EVAL_DATES, random_seed=base.OPTUNA_RANDOM_SEED,
        special_labels=base.SPECIAL_LABELS, min_special_points=base.MIN_SPECIAL_POINTS,
        min_event_gap=base.MIN_EVENT_GAP, max_events=base.MAX_EVENTS,
        selector_features_path=base.SELECTOR_FEATURES_PATH, cluster_column=base.CLUSTER_COLUMN,
        match_target_cluster=base.MATCH_TARGET_CLUSTER,
        recent_weekend_analogs=base.RECENT_WEEKEND_ANALOGS,
    )
    best = tuning.best_config
    run = run_analog_holidays(
        unique_id=uid, target_date=ts, source_path=base.SOURCE_PATH,
        season_length=base.SEASON_LENGTH,
        forecast_start_offset_hours=base.FORECAST_START_OFFSET_HOURS,
        k=int(best["k"]), typedist=str(best["typedist"]), typereg=str(best["typereg"]),
        scale_method=best.get("scale_method", base.SCALE_METHOD),
        n_components=int(best["n_components"]),
        regressor_params=dict(best.get("regressor_params", {})), levels=base.LEVELS,
        special_labels=base.SPECIAL_LABELS, min_special_points=base.MIN_SPECIAL_POINTS,
        min_event_gap=base.MIN_EVENT_GAP, max_events=base.MAX_EVENTS,
        expected_target_label=None,
        selector_features_path=base.SELECTOR_FEATURES_PATH, cluster_column=base.CLUSTER_COLUMN,
        match_target_cluster=base.MATCH_TARGET_CLUSTER,
        recent_weekend_analogs=base.RECENT_WEEKEND_ANALOGS,
    )
    start = pd.Timestamp(run.forecast_start)
    idx = pd.date_range(start, periods=len(run.forecast_profile), freq="h")
    print(f"    {REGION_LABEL.get(uid, uid):4s} k={best['k']} {best['typereg']:9s} "
          f"{best['typedist']:9s} análogos={len(run.positions)} fail={run.fail}", flush=True)
    return pd.Series(np.asarray(run.forecast_profile, dtype=float), index=idx, name=uid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=['2026-09-16:Independence Day'],
                    help='pares FECHA:ETIQUETA, p.ej. 2026-09-16:"Independence Day"')
    ap.add_argument("--out", default="docs/pronostico_festivos.csv")
    args = ap.parse_args()

    targets = []
    for item in args.targets:
        date, _, label = item.partition(":")
        targets.append((date.strip(), label.strip() or date.strip()))

    selector_path = rebind_selector(CRITERION)
    base.OPTUNA_MAX_K_BY_CLUSTER = {}
    preflight(targets, selector_path)

    frames = []
    for target, label in targets:
        print(f"{target} ({label}):", flush=True)
        cols = {}
        for uid in base.series_unique_ids:
            cols[uid] = forecast_one(uid, target, label)
        frames.append(pd.DataFrame(cols))

    wide = pd.concat(frames).sort_index()
    wide = wide.rename(columns=REGION_LABEL)
    out = pd.DataFrame({
        "date": wide.index.strftime("%Y-%m-%d"),
        "hour": wide.index.hour,
    })
    for c in COLUMN_ORDER:
        if c in wide.columns:
            out[c] = wide[c].round(1).to_numpy()

    path = Path("/home/uriel/GIT/analog_holidays") / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n-> {path}  ({len(out)} horas × {len(COLUMN_ORDER)} gerencias)", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
