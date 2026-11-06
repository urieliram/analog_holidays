"""Full production run WITH plots, to inspect performance visually and measure errors.

Runs the current PRODUCTION config (observance_tier, MIN_K=2, MIN_EVENT_GAP=24, full history,
adaptive PCR/PLS/Ridge/Lasso search, OPTUNA_MAX_K_BY_CLUSTER={'H':6}) over the 19 target dates
and generates, per SEN region:
  - batch_inference_<region>.png      forecast vs actual, all 19 holidays in a grid
  - batch_pair_sequences_<region>.png  X/X' and Y/Y' analog pair sequences

plus the usual metrics.csv / summary.csv. Prints a per-region and per-holiday error read-out.

Run headless:  MPLBACKEND=Agg python3 experiments/run_full_plotted.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_deep_regression_sweep as base  # noqa: E402

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


def main():
    base.OPTUNA_MAX_K_BY_CLUSTER = {"H": 6}  # production
    print("Full plotted production run | cap H<=6 | 8 series x 19 dates", flush=True)
    try:
        name, med, mean, picks = base.run_variant(
            SEARCH, "production_cap_H6_plotted", target_items=FULL_TARGETS, min_event_gap=24,
            make_plots=True,
            config_extra={"experiment_family": "production_plotted",
                          "target_dates_var": "TARGET_DATES_2025 (19, 2025-2026)",
                          "optuna_max_k_by_cluster": {"H": 6},
                          "note": "production config; figures = inference grids + pair sequences per region"},
        )
    finally:
        # Reset even on failure; a leaked cap would silently apply to later runs.
        base.OPTUNA_MAX_K_BY_CLUSTER = {}
    exp_dir = Path(__file__).resolve().parent / name
    print(f"\n>>> registered {name} | median mape_24={med:.3f}% mean={mean:.3f}% | picks={picks}", flush=True)
    print(f">>> plots in: {exp_dir / 'plots'}", flush=True)

    # ---- error read-out ----
    df = pd.read_csv(exp_dir / "metrics.csv")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
    print("\n================ ERROR BY REGION (mape_24, raw vs bias-adjusted) ================")
    g = df.groupby("unique_id").agg(
        n=("mape_24_pct", "size"),
        median_mape24=("mape_24_pct", "median"), mean_mape24=("mape_24_pct", "mean"),
        median_adj=("mape_holiday24_bias_adjusted_pct", "median"),
        bias24_mean=("bias_24", "mean"),
    ).round(2)
    print(g.to_string())
    print(f"  ALL: median_mape24={df['mape_24_pct'].median():.2f}  mean={df['mape_24_pct'].mean():.2f}"
          f"  median_adj={df['mape_holiday24_bias_adjusted_pct'].median():.2f}")

    print("\n================ ERROR BY HOLIDAY (median mape_24 across 8 regions) ================")
    h = df.groupby("holiday_label").agg(
        median_mape24=("mape_24_pct", "median"), mean_mape24=("mape_24_pct", "mean"),
        worst=("mape_24_pct", "max"),
    ).round(2).sort_values("median_mape24", ascending=False)
    print(h.to_string())

    print("\n================ WORST 10 CELLS (region x holiday) ================")
    worst = df.nlargest(10, "mape_24_pct")[
        ["unique_id", "target_date", "holiday_label", "analog_cluster", "k", "typereg", "mape_24_pct"]
    ].round(2)
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
