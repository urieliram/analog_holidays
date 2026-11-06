"""Re-stamp holiday_selector_features.csv with a chosen analog-cluster criterion.

Lightweight reproduction of M_identify_HOLIDAYS.ipynb section 11 (the analog-cluster
assignment): it loads the already-built selector features + priors, re-runs
`assign_holiday_selector_analog_clusters` with the requested criterion, and rewrites
the selector CSV with fresh `analog_cluster` / `analog_cluster_criterion` columns.

Use it to switch the active criterion (e.g. to `observance_tier`) without re-running
the heavy clustering sections of the notebook. The previous CSV is backed up first.

Run:  python -m analog_holidays.shared.regen_analog_clusters --criterion observance_tier
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from analog_holidays.shared.identify_holidays import (
    ANALOG_CLUSTER_CRITERIA_CATALOG,
    assign_holiday_selector_analog_clusters,
)

_HOLIDAYS_DIR = Path(__file__).resolve().parents[1] / "holidays"
SELECTOR_FEATURES_PATH = _HOLIDAYS_DIR / "holiday_selector_features.csv"
SELECTOR_PRIORS_PATH = _HOLIDAYS_DIR / "holiday_selector_priors.csv"
GROUP_COLS = ("unique_id", "anchor_holiday_name", "holiday_day_type")
CLUSTER_LABELS = ("F", "G", "H")
_DROP_COLS = ["analog_cluster", "analog_cluster_criterion", "analog_criterion", "analog_criterion_value"]


def regen(criterion: str, backup: bool = True) -> pd.DataFrame:
    if criterion not in ANALOG_CLUSTER_CRITERIA_CATALOG:
        raise SystemExit(
            f"Unknown criterion {criterion!r}. Options: {sorted(ANALOG_CLUSTER_CRITERIA_CATALOG)}"
        )
    sel = pd.read_csv(SELECTOR_FEATURES_PATH, parse_dates=["date"]).drop(columns=_DROP_COLS, errors="ignore")
    pri = pd.read_csv(SELECTOR_PRIORS_PATH)

    results = assign_holiday_selector_analog_clusters(
        df_selector=sel,
        df_priors=pri,
        criterion=criterion,
        group_cols=GROUP_COLS,
        cluster_labels=CLUSTER_LABELS,
    )
    df = results["df_selector_clusters"]
    catalog = results["analog_cluster_catalog"]
    prior_col = results.get("analog_criterion_prior_col")

    drop = ["analog_criterion", "analog_criterion_value"] + ([prior_col] if prior_col else [])
    out = (
        df.drop(columns=drop, errors="ignore")
        .assign(analog_cluster_criterion=criterion)
        .sort_values(["unique_id", "date", "holiday_name"])
        .reset_index(drop=True)
    )
    if out["analog_cluster"].isna().any():
        missing = sorted(out.loc[out["analog_cluster"].isna(), "anchor_holiday_name"].dropna().unique())
        raise SystemExit(f"Criterion {criterion!r} left rows unassigned. Unmapped anchors: {missing}")

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    if backup:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        bak = SELECTOR_FEATURES_PATH.with_suffix(f".bak_{stamp}.csv")
        shutil.copyfile(SELECTOR_FEATURES_PATH, bak)
        print(f"Backed up previous selector -> {bak.name}")
    out.to_csv(SELECTOR_FEATURES_PATH, index=False)

    print(f"Re-stamped {SELECTOR_FEATURES_PATH.name} with criterion={criterion!r}")
    print(f"  rows={len(out)} | series={out['unique_id'].nunique()}")
    print("  analog_cluster catalog:")
    for _, r in catalog.iterrows():
        print(f"    {r['analog_cluster']} <- {r['analog_criterion_value']}")
    print("  rows per cluster:", out.groupby("analog_cluster").size().to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--criterion", default="observance_tier")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    regen(args.criterion, backup=not args.no_backup)


if __name__ == "__main__":
    main()
