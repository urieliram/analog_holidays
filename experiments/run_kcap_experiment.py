"""Per-cluster k-ceiling experiment (deep holidays).

At gap=24, deep/Mode-A cells degrade sharply when Optuna picks k>=7 (deep median by k bucket:
k<=3 -> 3.99%, k4-6 -> 3.31%, k7+ -> 12.22%). The fix is NOT "few analogs" but "don't let k
exceed ~6". This caps k on the full-observance cluster H (where the catastrophic k>=7 deep
cells live -- NYD/Independence/Christmas Day) via the new OPTUNA_MAX_K_BY_CLUSTER mirror of
OPTUNA_MIN_K_BY_CLUSTER.

Control (no cap) = experiment_2026_06_16_00_29_event_gap_gap24_control (ALL 3.810%, deep 3.605%).
Variants: cap H at 6 (sweet-spot edge) and at 4 (tighter probe). Full 19 dates, all else production.

Run headless:  MPLBACKEND=Agg python3 experiments/run_kcap_experiment.py
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
DEEP = {"2025-01-01", "2025-09-16", "2025-12-24", "2025-12-25", "2025-12-31", "2026-01-01"}
CONTROL_DIR = "experiment_2026_06_16_00_29_event_gap_gap24_control"

RUNS = [
    ({"H": 6}, "kcap_H6"),
    ({"H": 4}, "kcap_H4"),
]


def main():
    exp_root = Path(__file__).resolve().parent
    print(f"k-cap experiment | {len(base.series_unique_ids)} series x {len(FULL_TARGETS)} dates", flush=True)
    dirs = {"control": CONTROL_DIR}
    for cap, slug in RUNS:
        base.OPTUNA_MAX_K_BY_CLUSTER = cap
        try:
            name, med, mean, picks = base.run_variant(
                SEARCH, slug, target_items=FULL_TARGETS, min_event_gap=24,
                config_extra={"experiment_family": "kcap_by_cluster",
                              "target_dates_var": "TARGET_DATES_2025 (19, 2025-2026)",
                              "baseline_compare": f"{CONTROL_DIR} (no cap; ALL 3.810%, deep 3.605%)",
                              "optuna_max_k_by_cluster": dict(cap)},
            )
        finally:
            # Reset even on failure; a leaked cap would silently apply to later variants.
            base.OPTUNA_MAX_K_BY_CLUSTER = {}
        dirs[slug] = name
        print(f">>> {slug} cap={cap}: median mape_24={med:.3f}% mean={mean:.3f}% -> {name}", flush=True)

    def load(tag):
        df = pd.read_csv(exp_root / dirs[tag] / "metrics.csv")
        df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
        df["seg"] = np.where(df["target_date"].isin(DEEP), "deep", "soft")
        return df

    frames = {tag: load(tag) for tag in dirs}
    ctrl = frames["control"]
    print("\n================ k-CAP on cluster H vs no-cap control ================")
    for tag in dirs:
        f = frames[tag]
        line = f"  {tag:10s}"
        for seg in ["all", "deep", "soft"]:
            s = f if seg == "all" else f[f["seg"] == seg]
            line += f" | {seg} med={s['mape_24_pct'].median():.3f} mean={s['mape_24_pct'].mean():.3f}"
        print(line)

    # which H cells actually changed (k capped), and collateral on civic-H
    print("\n================ effect on H-cluster cells (k>6 at control) ================")
    ch = ctrl[ctrl["analog_cluster"] == "H"].set_index(["unique_id", "target_date"])
    for tag, cap in [("kcap_H6", 6), ("kcap_H4", 4)]:
        vh = frames[tag][frames[tag]["analog_cluster"] == "H"].set_index(["unique_id", "target_date"])
        common = ch.index.intersection(vh.index)
        changed = common[(ch.loc[common, "k"] > cap)]
        d_all = vh.loc[common, "mape_24_pct"] - ch.loc[common, "mape_24_pct"]
        print(f"\n  {tag}: H cells with control k>{cap}: {len(changed)} of {len(common)}")
        if len(changed):
            before = ch.loc[changed, "mape_24_pct"]
            after = vh.loc[changed, "mape_24_pct"]
            seg = np.where(changed.get_level_values("target_date").isin(DEEP), "deep", "civic")
            tab = pd.DataFrame({"seg": seg, "k_ctrl": ch.loc[changed, "k"].values,
                                "mape_before": before.values, "mape_after": after.values})
            tab["delta"] = tab["mape_after"] - tab["mape_before"]
            print(tab.round(2).to_string(index=False))
            print(f"    capped cells mean delta: {tab['delta'].mean():+.3f}pp "
                  f"(deep {tab[tab.seg=='deep']['delta'].mean():+.3f}, civic {tab[tab.seg=='civic']['delta'].mean():+.3f})")
        print(f"  all H cells mean delta vs control: {d_all.mean():+.3f}pp")


if __name__ == "__main__":
    main()
