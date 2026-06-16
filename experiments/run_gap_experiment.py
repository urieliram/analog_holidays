"""MIN_EVENT_GAP axis experiment: 24 vs 12.

One-axis study on the full 19-date set (2025-2026), everything else at production
(observance_tier, MIN_K=2, full history, adaptive regressor search PCR/PLS/Ridge/Lasso).
Lowering MIN_EVENT_GAP 24 -> 12 lets candidate analog events overlap every 12h instead of
24h, ~doubling the analog candidate pool. Tests whether a larger pool helps the
analog-starved deep cells -- and resolves the uncommitted gap=12 config debt WITH evidence.

Reuses the parametrized machinery in run_deep_regression_sweep.run_variant.

Run headless:  MPLBACKEND=Agg python3 experiments/run_gap_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_deep_regression_sweep as base  # noqa: E402  (runs source/selector setup on import)

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
SEARCH = ["PCR", "PLS", "RidgeReg", "LassoReg"]  # production adaptive search
DEEP = {"2025-01-01", "2025-09-16", "2025-12-24", "2025-12-25", "2025-12-31", "2026-01-01"}

RUNS = [(24, "gap24_control"), (12, "gap12")]


def _config_extra(gap):
    return {
        "experiment_family": "min_event_gap_axis",
        "target_dates_var": "TARGET_DATES_2025 (19, 2025-2026)",
        "baseline_compare": "gap=24 control within this experiment (and 14_21_46 = 3.76%)",
        "min_event_gap_swept": gap,
    }


def main():
    print(f"MIN_EVENT_GAP axis | {len(base.series_unique_ids)} series x {len(FULL_TARGETS)} dates", flush=True)
    dirs = {}
    for gap, slug in RUNS:
        name, med, mean, picks = base.run_variant(
            SEARCH, f"event_gap_{slug}", target_items=FULL_TARGETS,
            min_event_gap=gap, config_extra=_config_extra(gap),
        )
        dirs[gap] = name
        print(f">>> gap={gap}: median mape_24={med:.3f}% mean={mean:.3f}% -> {name}", flush=True)

    # ---- comparison ----
    exp_root = Path(__file__).resolve().parent

    def load(gap):
        df = pd.read_csv(exp_root / dirs[gap] / "metrics.csv")
        df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
        df["seg"] = np.where(df["target_date"].isin(DEEP), "deep", "soft")
        return df

    g24, g12 = load(24), load(12)
    print("\n================ MIN_EVENT_GAP 24 vs 12 ================")
    for seg in ["all", "deep", "soft"]:
        a = g24 if seg == "all" else g24[g24["seg"] == seg]
        b = g12 if seg == "all" else g12[g12["seg"] == seg]
        print(f"  {seg:5s} n={len(a):3d} | gap24 median={a['mape_24_pct'].median():.3f}% mean={a['mape_24_pct'].mean():.3f}%"
              f" | gap12 median={b['mape_24_pct'].median():.3f}% mean={b['mape_24_pct'].mean():.3f}%"
              f" | dmedian={b['mape_24_pct'].median()-a['mape_24_pct'].median():+.3f}pp")

    k24 = g24.set_index(["unique_id", "target_date"])["mape_24_pct"]
    k12 = g12.set_index(["unique_id", "target_date"])["mape_24_pct"]
    common = k24.index.intersection(k12.index)
    diff = k12.loc[common] - k24.loc[common]
    print(f"\n  win/loss gap12 vs gap24: better={int((diff < -1e-9).sum())} "
          f"worse={int((diff > 1e-9).sum())} ties={int((diff.abs() <= 1e-9).sum())} | mean_delta={diff.mean():+.3f}pp")

    print("\n  per-holiday median mape_24 (gap24 -> gap12):")
    m = (pd.concat([g24.assign(g="g24"), g12.assign(g="g12")])
         .groupby(["holiday_label", "g"])["mape_24_pct"].median().unstack("g"))
    m["delta"] = m["g12"] - m["g24"]
    print(m.round(2).sort_values("delta").to_string())


if __name__ == "__main__":
    main()
