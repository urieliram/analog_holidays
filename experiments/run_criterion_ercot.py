"""Correr el panel ERCOT (9 series x 19 fechas = 171 celdas) bajo un criterio de cluster.

`run_deep_regression_sweep` está cableado a MX en tiempo de importación (fuente, series,
selector). Este script reapunta esas globales a ERCOT y regenera los lookups antes de
llamar a `run_variant`, que es la misma maquinaria usada para MX — así los dos paneles
son directamente comparables.

Uso:  cd /home/uriel/GIT
      MPLBACKEND=Agg python3 analog_holidays/experiments/run_criterion_ercot.py holiday_identity
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, "/home/uriel/GIT")

import run_deep_regression_sweep as base  # noqa: E402
from analog_holidays.analog.analog_holidays import (  # noqa: E402
    get_available_unique_ids,
    prepare_audit_working_copy,
)
from analog_holidays.shared.identify_holidays import (  # noqa: E402
    assign_holiday_selector_analog_clusters,
)

HOL = Path("/home/uriel/GIT/analog_holidays/holidays")
SELECTORS = EXPERIMENTS / "selectors"
GROUP_COLS = ("unique_id", "anchor_holiday_name", "holiday_day_type")
CLUSTER_LABELS = ("F", "G", "H")
DROP = ["analog_cluster", "analog_cluster_criterion", "analog_criterion", "analog_criterion_value"]

# Mismas 19 fechas objetivo que las corridas ERCOT previas (n_metric_rows = 171).
FULL_TARGETS = [
    ("2025-01-01", "New Year's Day"),        ("2025-01-20", "Martin Luther King Jr. Day"),
    ("2025-02-17", "Presidents' Day"),       ("2025-03-02", "Texas Independence Day"),
    ("2025-04-21", "San Jacinto Day"),       ("2025-05-26", "Memorial Day"),
    ("2025-06-19", "Juneteenth National Independence Day"),
    ("2025-07-04", "Independence Day"),      ("2025-08-27", "Lyndon B. Johnson Day"),
    ("2025-09-01", "Labor Day"),             ("2025-11-11", "Veterans Day"),
    ("2025-11-27", "Thanksgiving Day"),      ("2025-11-28", "Day after Thanksgiving"),
    ("2025-12-24", "Christmas Eve"),         ("2025-12-25", "Christmas Day"),
    ("2026-01-01", "New Year's Day"),        ("2026-01-19", "Martin Luther King Jr. Day"),
    ("2026-02-16", "Presidents' Day"),       ("2026-03-02", "Texas Independence Day"),
]
SEARCH = ["PCR", "PLS", "RidgeReg", "LassoReg"]


def build_selector(criterion: str) -> Path:
    """Re-sella el selector ERCOT con el criterio pedido; un archivo por criterio."""
    SELECTORS.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(HOL / "holiday_selector_features_ercot.csv", parse_dates=["date"])
    pri = pd.read_csv(HOL / "holiday_selector_priors_ercot.csv")
    res = assign_holiday_selector_analog_clusters(
        df_selector=sel.drop(columns=DROP, errors="ignore"), df_priors=pri,
        criterion=criterion, group_cols=GROUP_COLS, cluster_labels=CLUSTER_LABELS)
    df = res["df_selector_clusters"]
    prior_col = res.get("analog_criterion_prior_col")
    drop = ["analog_criterion", "analog_criterion_value"] + ([prior_col] if prior_col else [])
    out = (df.drop(columns=drop, errors="ignore")
             .assign(analog_cluster_criterion=criterion)
             .sort_values(["unique_id", "date", "holiday_name"])
             .reset_index(drop=True))
    path = SELECTORS / f"selector_ercot_{criterion}.csv"
    out.to_csv(path, index=False)
    cl = out["analog_cluster"]
    per = out.dropna(subset=["analog_cluster"]).groupby(["unique_id", "analog_cluster"]).size()
    print(f"selector ERCOT {criterion}: clusters={cl.dropna().nunique()} "
          f"NaN={int(cl.isna().sum())}/{len(out)} "
          f"pool/serie min={per.min()} med={int(per.median())} max={per.max()} -> {path.name}",
          flush=True)
    return path


def rebind_to_ercot(selector_path: Path) -> None:
    """Reapunta la fuente, las series y el selector del runner hacia ERCOT."""
    original = HOL / "holiday_demand_ercot.csv"
    base.SOURCE_PATH = prepare_audit_working_copy(
        original, working_dir=HOL / "working", prefix="holiday_demand_ercot", reuse_today=True)
    base.UNIQUE_IDS = get_available_unique_ids(base.SOURCE_PATH)
    base.series_unique_ids = [str(u) for u in base.UNIQUE_IDS]

    base.SELECTOR_FEATURES_PATH = selector_path
    df = pd.read_csv(selector_path, parse_dates=["date"])
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


def main() -> None:
    criterion = sys.argv[1] if len(sys.argv) > 1 else "holiday_identity"
    path = build_selector(criterion)
    rebind_to_ercot(path)
    base.OPTUNA_MAX_K_BY_CLUSTER = {}          # el cap {'H':6} es específico de observance_tier

    print(f"ERCOT | criterio={criterion} | {len(base.series_unique_ids)} series "
          f"x {len(FULL_TARGETS)} fechas", flush=True)
    name, med, mean, picks = base.run_variant(
        SEARCH, f"ercot_criterion_{criterion}", target_items=FULL_TARGETS, min_event_gap=24,
        config_extra={
            "experiment_family": "ercot_replication",
            "dataset": "ercot",
            "analog_cluster_criterion": criterion,
            "selector_features_path": str(path),
            "note": "replicación ERCOT del panel MX, todos los arreglos activos, sin cap de k",
        },
    )
    print(f"\n>>> ERCOT {criterion}: {name} | median mape_24={med:.3f}% mean={mean:.3f}% "
          f"| picks={picks}", flush=True)


if __name__ == "__main__":
    main()
