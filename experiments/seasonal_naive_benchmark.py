"""Seasonal-naive benchmark: champion analog model vs same-holiday naive baselines.

For every (region, holiday target) the champion is compared against baselines built ONLY from
the same holiday's own history (e.g. Independence forecast from past Independences):
  persist1 = last year's same holiday (most recent prior instance)
  mean2/mean3/mean4 = mean profile of the last 2 / 3 / 4 instances
MAPE is over the 24 holiday hours, same window as the champion's mape_24_pct.

Champion metrics come from a logged run's metrics.csv (default: the cap-H6 plotted production run).
Writes experiments/seasonal_naive_results.csv and prints per-holiday / per-region / overall tables.

Run:  python3 experiments/seasonal_naive_benchmark.py [path/to/metrics.csv]
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
HARD = {"SEN_demand_NES", "SEN_demand_NTE", "SEN_demand_PEN"}


def main(metrics_path):
    m = pd.read_csv(metrics_path)
    m["target_date"] = pd.to_datetime(m["target_date"]).dt.normalize()
    src = pd.read_csv(SRC, parse_dates=["ds"])
    src["d"] = src["ds"].dt.normalize(); src["h"] = src["ds"].dt.hour
    sel = pd.read_csv(SEL, parse_dates=["date"]); sel["date"] = sel["date"].dt.normalize()
    sel["unique_id"] = sel["unique_id"].astype(str)

    def profile(region, date):
        v = src[src["d"] == date].sort_values("h")[region].to_numpy(dtype=float)
        return v if len(v) == 24 and np.isfinite(v).all() else None

    def prior_dates(region, anchor, day_type, target_date):
        c = sel[(sel["unique_id"] == region) & (sel["anchor_holiday_name"] == anchor)
                & (sel["holiday_day_type"] == day_type) & (sel["date"] < target_date)]
        if c.empty:
            c = sel[(sel["unique_id"] == region) & (sel["anchor_holiday_name"] == anchor)
                    & (sel["date"] < target_date)]
        return sorted(c["date"].unique(), reverse=True)

    def mape(actual, fc):
        denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
        return float(np.nanmean(np.abs(actual - fc) / denom * 100))

    rows = []
    for _, r in m.iterrows():
        reg, td = r["unique_id"], r["target_date"]
        actual = profile(reg, td)
        if actual is None:
            continue
        profs = [p for p in (profile(reg, d) for d in prior_dates(reg, r.get("holiday_label"),
                 r.get("holiday_day_type"), td)) if p is not None]
        rec = dict(region=reg, holiday=r.get("holiday_label"), target_date=td.strftime("%Y-%m-%d"),
                   cluster=r.get("analog_cluster"), champion=r["mape_24_pct"], n_prior=len(profs))
        if len(profs) >= 1: rec["persist1"] = mape(actual, profs[0])
        if len(profs) >= 2: rec["mean2"] = mape(actual, np.mean(profs[:2], axis=0))
        if len(profs) >= 3: rec["mean3"] = mape(actual, np.mean(profs[:3], axis=0))
        if len(profs) >= 4: rec["mean4"] = mape(actual, np.mean(profs[:4], axis=0))
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "experiments" / "seasonal_naive_results.csv", index=False)
    cols = ["champion", "persist1", "mean2", "mean3", "mean4"]

    def agg(sub, label):
        out = {"group": label, "n": len(sub)}
        for c in cols:
            out[c + "_med"] = round(sub[c].median(), 2)
        for c in cols[1:]:
            both = sub.dropna(subset=["champion", c])
            out["champ<" + c] = f"{100*(both['champion']<both[c]).mean():.0f}%"
        return out

    print("================ OVERALL & BY DIFFICULTY (median mape_24) ================")
    print(pd.DataFrame([agg(d, "ALL"),
                        agg(d[~d.region.isin(HARD)], "TRACTABLE-5"),
                        agg(d[d.region.isin(HARD)], "HARD-3")]).to_string(index=False))

    print("\n================ BY HOLIDAY (median across regions) ================")
    print(d.groupby("holiday")[cols].median().round(2).sort_values("champion", ascending=False).to_string())

    print("\n================ BY REGION (median across holidays) ================")
    print(d.groupby("region")[cols].median().round(2).to_string())

    print("\n================ champion mean vs best naive (per holiday) ================")
    g = d.groupby("holiday")[cols].median()
    g["best_naive"] = g[["persist1", "mean2", "mean3", "mean4"]].min(axis=1)
    g["champ_edge_pp"] = (g["best_naive"] - g["champion"]).round(2)
    print(g[["champion", "best_naive", "champ_edge_pp"]].round(2).sort_values("champ_edge_pp").to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METRICS)
