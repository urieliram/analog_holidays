"""Generate one MX selector-features CSV per analog-cluster criterion.

Each variant run needs its own file so the panel runs can go in parallel without
racing on the shared holiday_selector_features.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/uriel/GIT")

from analog_holidays.shared.identify_holidays import (  # noqa: E402
    ANALOG_CLUSTER_CRITERIA_CATALOG,
    assign_holiday_selector_analog_clusters,
)

HOL = Path("/home/uriel/GIT/analog_holidays/holidays")
OUT = Path(__file__).resolve().parent / "selectors"
GROUP_COLS = ("unique_id", "anchor_holiday_name", "holiday_day_type")
CLUSTER_LABELS = ("F", "G", "H")
DROP = ["analog_cluster", "analog_cluster_criterion", "analog_criterion", "analog_criterion_value"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sel0 = pd.read_csv(HOL / "holiday_selector_features.csv", parse_dates=["date"])
    pri = pd.read_csv(HOL / "holiday_selector_priors.csv")

    for criterion in ANALOG_CLUSTER_CRITERIA_CATALOG:
        sel = sel0.drop(columns=DROP, errors="ignore").copy()
        try:
            res = assign_holiday_selector_analog_clusters(
                df_selector=sel, df_priors=pri, criterion=criterion,
                group_cols=GROUP_COLS, cluster_labels=CLUSTER_LABELS,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{criterion:32s} FALLO: {type(exc).__name__}: {exc}")
            continue

        df = res["df_selector_clusters"]
        prior_col = res.get("analog_criterion_prior_col")
        drop = ["analog_criterion", "analog_criterion_value"] + ([prior_col] if prior_col else [])
        out = (
            df.drop(columns=drop, errors="ignore")
            .assign(analog_cluster_criterion=criterion)
            .sort_values(["unique_id", "date", "holiday_name"])
            .reset_index(drop=True)
        )
        path = OUT / f"selector_{criterion}.csv"
        out.to_csv(path, index=False)

        cl = out["analog_cluster"]
        n_groups = cl.dropna().nunique()
        n_na = int(cl.isna().sum())
        # pool size per cluster, per series: how many analog candidates a target can draw on
        per = (out.dropna(subset=["analog_cluster"])
                  .groupby(["unique_id", "analog_cluster"]).size())
        print(f"{criterion:32s} clusters={n_groups:3d}  NaN={n_na:4d}/{len(out)}  "
              f"pool/serie min={per.min():3d} med={int(per.median()):3d} max={per.max():3d}  -> {path.name}")


if __name__ == "__main__":
    main()
