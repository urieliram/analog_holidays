"""Analyze the deep-cluster regression sweep.

Reads every experiment_*_deep_reg_* folder plus the deep subset of the robust
winner baseline (experiment_2026_06_14_21_46) and prints a comparison:
  - deep median / mean mape_24 per variant,
  - per-date medians,
  - regressor picks per variant,
  - per-(series, target_date) win/loss of each variant vs the control run.

Usage:  python3 experiments/analyze_deep_regression_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
DEEP_DATES = ["2025-01-01", "2025-09-16", "2025-12-24", "2025-12-25", "2025-12-31", "2026-01-01"]
BASELINE = "experiment_2026_06_14_21_46"
ORDER = ["control_baselinesearch", "PCR", "PLS", "RidgeReg", "LassoReg", "RF_enriched"]


def _load(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / "metrics.csv")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
    return df


def _deep(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["target_date"].isin(DEEP_DATES)].copy()


def main():
    variants = {}
    for d in sorted(EXP.glob("experiment_*_deep_reg_*")):
        tag = d.name.split("_deep_reg_")[-1]
        variants[tag] = _deep(_load(d))

    base_deep = _deep(_load(EXP / BASELINE))
    print(f"\nBASELINE {BASELINE} (deep subset): "
          f"median={base_deep['mape_24_pct'].median():.3f}%  mean={base_deep['mape_24_pct'].mean():.3f}%  n={len(base_deep)}")

    print("\n================ VARIANT SUMMARY (deep subset, mape_24) ================")
    print(f"{'variant':28s} {'median':>8s} {'mean':>8s} {'n':>4s}  {'vs control (median)':>20s}")
    ctrl = variants.get("control_baselinesearch")
    ctrl_med = ctrl["mape_24_pct"].median() if ctrl is not None else np.nan
    for tag in ORDER:
        if tag not in variants:
            continue
        v = variants[tag]
        med, mean = v["mape_24_pct"].median(), v["mape_24_pct"].mean()
        delta = "" if tag == "control_baselinesearch" else f"{med - ctrl_med:+.3f} pp"
        print(f"{tag:28s} {med:8.3f} {mean:8.3f} {len(v):4d}  {delta:>20s}")

    print("\n================ PER-DATE MEDIAN mape_24 ================")
    rows = []
    for tag in ORDER:
        if tag not in variants:
            continue
        s = variants[tag].groupby("target_date")["mape_24_pct"].median()
        s.name = tag
        rows.append(s)
    if rows:
        table = pd.concat(rows, axis=1).reindex(DEEP_DATES)
        print(table.round(2).to_string())

    print("\n================ REGRESSOR PICKS per variant ================")
    for tag in ORDER:
        if tag in variants:
            print(f"{tag:28s} {variants[tag]['typereg'].value_counts().to_dict()}")

    if ctrl is not None:
        print("\n================ WIN/LOSS vs control (per series×date, mape_24) ================")
        ck = ctrl.set_index(["unique_id", "target_date"])["mape_24_pct"]
        for tag in ORDER:
            if tag == "control_baselinesearch" or tag not in variants:
                continue
            vk = variants[tag].set_index(["unique_id", "target_date"])["mape_24_pct"]
            common = ck.index.intersection(vk.index)
            diff = vk.loc[common] - ck.loc[common]
            wins = int((diff < -1e-9).sum())   # variant better
            losses = int((diff > 1e-9).sum())
            print(f"{tag:28s} better={wins:3d}  worse={losses:3d}  ties={len(common)-wins-losses:3d}  "
                  f"mean_delta={diff.mean():+.3f}pp")


if __name__ == "__main__":
    main()
