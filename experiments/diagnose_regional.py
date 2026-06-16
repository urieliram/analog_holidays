"""Regional diagnostic: why are NES/NTE/PEN ~3x worse than CEL/SIN/OCC?

Tests three hypotheses for the regional holiday-MAPE spread, using the source demand and
a plotted production run's metrics.csv:
  H1 poor/thin analog pool   -> selected_analogs vs k (starvation)
  H2 noisier demand          -> normal-day residual around (dow,hour) climatology (rRMSE)
  H3 inconsistent observance -> year-to-year std of the holiday demand drop

Run:  python3 experiments/diagnose_regional.py [path/to/metrics.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "holidays" / "holiday_demand_mx.csv"
DEFAULT_METRICS = ROOT / "experiments" / "experiment_2026_06_16_11_35_production_cap_H6_plotted" / "metrics.csv"
BAD = {"SEN_demand_NES", "SEN_demand_NTE", "SEN_demand_PEN"}
GOOD = {"SEN_demand_CEL", "SEN_demand_SIN", "SEN_demand_OCC"}


def grp(u):
    return "BAD " if u in BAD else ("good" if u in GOOD else "    ")


def main(metrics_path):
    df = pd.read_csv(SRC, parse_dates=["ds"])
    df["date"] = df["ds"].dt.normalize(); df["hour"] = df["ds"].dt.hour; df["dow"] = df["ds"].dt.dayofweek
    regions = [c for c in df.columns if c.startswith("SEN_demand_") and not c.endswith("_holiday")]
    m = pd.read_csv(metrics_path)
    mape = m.groupby("unique_id")["mape_24_pct"].median()

    rows = []
    for r in regions:
        hol = r + "_holiday"
        d = df[["date", "dow", "hour", r, hol]].dropna(subset=[r]).copy()
        mean_mw = d[r].mean()
        # H2: normal-day intrinsic noise = rRMSE around (dow,hour) climatology on non-holidays
        normal = d[d[hol] == 0]
        clim = normal.groupby(["dow", "hour"])[r].transform("mean")
        rrmse = float(np.sqrt(np.mean(((normal[r] - clim) / clim * 100) ** 2)))
        # H3: year-to-year variability of the daily holiday drop vs weekday-normal
        daily = d.groupby("date").agg(dm=(r, "mean"), ishol=(hol, "max"), dow=("dow", "first")).reset_index()
        cl = daily[daily["ishol"] == 0].groupby("dow")["dm"].mean()
        daily["drop"] = (1 - daily["dm"] / daily["dow"].map(cl)) * 100
        hd = daily[daily["ishol"] == 1]["drop"]
        # H1: starvation
        g = m[m["unique_id"] == r]
        starved = int((g["selected_analogs"] < g["k"]).sum())
        rows.append(dict(region=r, grp=grp(r), mean_MW=mean_mw, normday_rRMSE=rrmse,
                         drop_mean=hd.mean(), drop_yoy_std=hd.std(),
                         mean_k=g["k"].mean(), starved=starved, holiMAPE=float(mape.get(r, np.nan))))
    t = pd.DataFrame(rows).sort_values("holiMAPE")
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    ok = t.dropna(subset=["holiMAPE"])
    print("\ncorr(holiMAPE, mean_MW)        =", round(ok["holiMAPE"].corr(ok["mean_MW"]), 3), " (H1 scale)")
    print("corr(holiMAPE, normday_rRMSE)  =", round(ok["holiMAPE"].corr(ok["normday_rRMSE"]), 3), " (H2 noise)")
    print("corr(holiMAPE, drop_yoy_std)   =", round(ok["holiMAPE"].corr(ok["drop_yoy_std"]), 3), " (H3 observance)")
    print("total starved cells (sel<k)    =", int(t["starved"].sum()), " (H1 pool: 0 => no starvation)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METRICS)
