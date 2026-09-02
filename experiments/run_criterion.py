"""Run the 152-cell MX production panel under one analog-cluster criterion.

Usage:  python3 run_criterion.py <criterion>

Rebinds the runner's selector-features globals to a per-criterion CSV so several
criteria can run concurrently in separate processes without racing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS = Path("/home/uriel/GIT/analog_holidays/experiments")
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, "/home/uriel/GIT")

import run_deep_regression_sweep as base  # noqa: E402

SELECTORS = Path(__file__).resolve().parent / "selectors"

FULL_TARGETS = [
    ("2025-01-01", "New Year's Day"), ("2025-02-03", "Constitution Day"),
    ("2025-03-17", "Benito Juarez's Birthday"), ("2025-04-17", "Maundy Thursday"),
    ("2025-04-18", "Good Friday"), ("2025-04-19", "Holy Saturday"),
    ("2025-05-01", "Labor Day"), ("2025-09-16", "Independence Day"),
    ("2025-11-17", "Mexican Revolution Day"), ("2025-12-24", "Christmas Eve"),
    ("2025-12-25", "Christmas Day"), ("2025-12-31", "New Year's Eve"),
    ("2026-01-01", "New Year's Day"), ("2026-02-02", "Constitution Day"),
    ("2026-03-16", "Benito Juarez's Birthday"), ("2026-04-02", "Maundy Thursday"),
    ("2026-04-03", "Good Friday"), ("2026-04-04", "Holy Saturday"),
    ("2026-05-01", "Labor Day"),
]
SEARCH = ["PCR", "PLS", "RidgeReg", "LassoReg"]


def rebind(criterion: str) -> None:
    """Point the runner's module-level selector state at this criterion's CSV."""
    path = SELECTORS / f"selector_{criterion}.csv"
    if not path.exists():
        raise SystemExit(f"missing selector CSV for {criterion!r}: {path}")

    base.SELECTOR_FEATURES_PATH = path
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["unique_id"] = df["unique_id"].astype(str)
    base.selector_features_df = df

    col = base.CLUSTER_COLUMN
    base.selector_cluster_lookup_by_id = {
        uid: (df.loc[df["unique_id"] == uid].dropna(subset=[col])
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")[col].to_dict())
        for uid in base.series_unique_ids
    }
    base.selector_anchor_lookup_by_id = {
        uid: (df.loc[df["unique_id"] == uid].dropna(subset=["anchor_holiday_name"])
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")["anchor_holiday_name"].to_dict())
        for uid in base.series_unique_ids
    }


def main() -> None:
    criterion = sys.argv[1]
    max_k = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rebind(criterion)
    # Per-cluster k caps were tuned for the 3-letter observance_tier labels and do
    # not transfer to criteria with a different label alphabet.
    base.OPTUNA_MAX_K_BY_CLUSTER = {}
    slug = f"criterion_{criterion}" if max_k is None else f"criterion_{criterion}_kcap{max_k}"
    if max_k is not None:
        # Same ceiling for every cluster letter, so the k axis is held fixed while
        # only the grouping semantics vary.
        base.OPTUNA_MAX_K_BY_CLUSTER = {
            str(c): max_k for c in base.selector_features_df[base.CLUSTER_COLUMN].dropna().unique()
        }
    print(f"criterio={criterion} max_k={max_k} | 8 series x {len(FULL_TARGETS)} fechas", flush=True)

    name, med, mean, picks = base.run_variant(
        SEARCH, slug, target_items=FULL_TARGETS, min_event_gap=24,
        config_extra={
            "experiment_family": "cluster_criterion_sweep_postfix",
            "target_dates_var": "TARGET_DATES_2025 (19, 2025-2026)",
            "analog_cluster_criterion": criterion,
            "selector_features_path": str(SELECTORS / f"selector_{criterion}.csv"),
            "note": "criterion sweep, all bug fixes active",
            "uniform_max_k": max_k,
        },
    )
    print(f"\n>>> {criterion}: {name} | median mape_24={med:.3f}% mean={mean:.3f}% | picks={picks}", flush=True)


if __name__ == "__main__":
    main()
