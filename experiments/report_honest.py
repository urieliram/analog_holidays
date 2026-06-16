"""Honest performance report: per-region + noise-floor-aware aggregate + skill vs naive.

The global median (3.74%) is inflated by 3 intrinsically-noisy regions (NES/NTE/PEN; see
diagnose_regional.py). This report reframes performance honestly:
  - per-region method MAPE next to a NAIVE persistence baseline (same holiday, prior year),
  - SKILL = 1 - method/persistence (how much the method beats naive), per region,
  - reframed headlines: all vs tractable-only vs hard-only, and overall skill.

If skill is high AND uniform across regions, the 3x raw-MAPE spread is region difficulty,
not method quality -> the method is better than the global number suggests.

Run:  python3 experiments/report_honest.py [path/to/metrics.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "holidays" / "holiday_demand_mx.csv"
SEL = ROOT / "holidays" / "holiday_selector_features.csv"
DEFAULT_METRICS = ROOT / "experiments" / "experiment_2026_06_16_11_35_production_cap_H6_plotted" / "metrics.csv"
HARD = ["SEN_demand_NES", "SEN_demand_NTE", "SEN_demand_PEN"]


def main(metrics_path):
    m = pd.read_csv(metrics_path)
    m["target_date"] = pd.to_datetime(m["target_date"]).dt.normalize()
    src = pd.read_csv(SRC, parse_dates=["ds"])
    src["d"] = src["ds"].dt.normalize(); src["h"] = src["ds"].dt.hour
    sel = pd.read_csv(SEL, parse_dates=["date"])
    sel["date"] = sel["date"].dt.normalize()

    def profile(region, date):
        sub = src[(src["d"] == date)].sort_values("h")[region]
        v = sub.to_numpy(dtype=float)
        return v if len(v) == 24 and np.isfinite(v).all() else None

    def prior_instance(region, label, day_type, target_date):
        cand = sel[(sel["unique_id"].astype(str) == region)
                   & (sel["anchor_holiday_name"] == label)
                   & (sel["holiday_day_type"] == day_type)
                   & (sel["date"] < target_date)]
        if cand.empty:
            cand = sel[(sel["unique_id"].astype(str) == region)
                       & (sel["anchor_holiday_name"] == label) & (sel["date"] < target_date)]
        return cand["date"].max() if not cand.empty else None

    rows = []
    for _, r in m.iterrows():
        reg, td = r["unique_id"], r["target_date"]
        actual = profile(reg, td)
        pd_ = prior_instance(reg, r.get("holiday_label"), r.get("holiday_day_type"), td)
        persist_mape = np.nan
        if actual is not None and pd_ is not None:
            pf = profile(reg, pd_)
            if pf is not None:
                denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
                persist_mape = float(np.nanmean(np.abs(actual - pf) / denom * 100))
        rows.append(dict(region=reg, method=r["mape_24_pct"], persist=persist_mape))
    d = pd.DataFrame(rows)
    d["skill"] = 1 - d["method"] / d["persist"]

    print("================ PER-REGION (method vs naive persistence) ================")
    print(f"{'region':16s} {'n':>3s} {'method_med':>10s} {'method_mean':>11s} {'persist_med':>11s} {'skill_med':>9s}")
    for reg, g in d.groupby("region"):
        gs = g.dropna(subset=["persist"])
        print(f"{reg:16s} {len(g):3d} {g['method'].median():10.2f} {g['method'].mean():11.2f} "
              f"{gs['persist'].median():11.2f} {gs['skill'].median():9.2f}")

    def blk(name, sub):
        s = sub.dropna(subset=["persist"])
        print(f"  {name:22s} n={len(sub):3d} | method med={sub['method'].median():.2f} mean={sub['method'].mean():.2f}"
              f" | persist med={s['persist'].median():.2f} | skill med={s['skill'].median():.2f}"
              f" | method<persist {100*(s['method']<s['persist']).mean():.0f}%")

    print("\n================ REFRAMED HEADLINES ================")
    blk("ALL (raw headline)", d)
    blk("TRACTABLE (5 regions)", d[~d["region"].isin(HARD)])
    blk("HARD (NES/NTE/PEN)", d[d["region"].isin(HARD)])
    print("\nReading: if skill is similar in HARD and TRACTABLE, the method is uniformly good and the")
    print("raw-MAPE spread is region difficulty (irreducible noise), not a method gap.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METRICS)
