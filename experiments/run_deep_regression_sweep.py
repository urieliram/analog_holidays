"""Deep-cluster regression sweep (Mode A).

Sweeps the regression axis ONLY on the deep / Mode-A holidays (the worst-MAPE
drivers: Christmas, New Year, Independence), holding everything else at the
production baseline (observance_tier, uniform MIN_K=2, full history). Each
variant changes only OPTUNA_TYPEREG_CHOICES and is registered as its own
experiment folder, directly comparable to the deep subset of the robust winner
`experiment_2026_06_14_21_46` (deep median mape_24 = 4.03%).

Variants:
  control      : baseline search  ['PCR','PLS','RidgeReg','LassoReg']  -> must reproduce ~4.03%
  PCR / PLS / RidgeReg / LassoReg : each linear regressor forced singly (Optuna still tunes k/scale/n_comp/typedist)
  RF_enriched  : adds the non-linear RandomForest to the search space

Reuses the notebook pipeline verbatim (P_analog_holidays_38h_ahead cluster.ipynb
cells In[7] rolling loop + In[8] panel-metric / bias-adjustment enrichment) so
metrics.csv carries the same schema as prior experiments.

Run headless:  MPLBACKEND=Agg python3 experiments/run_deep_regression_sweep.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/uriel/GIT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import analog_holidays.analog.analog_holidays as analog_holidays_module
from analog_holidays.analog.analog_holidays import (
    run_analog_holidays,
    tune_analog_holidays_optuna,
    prepare_audit_working_copy,
)
from analog_holidays.shared.experiment_logging import save_experiment_run

# ── Source + selector (mirror notebook In[1]/In[2]/In[6]) ─────────────────────
_ORIGINAL_SOURCE = PROJECT_ROOT / "analog_holidays" / "holidays" / "holiday_demand_mx.csv"
_WORKING_SOURCE_DIR = _ORIGINAL_SOURCE.parent / "working"
SOURCE_PATH = prepare_audit_working_copy(
    _ORIGINAL_SOURCE, working_dir=_WORKING_SOURCE_DIR, prefix="holiday_demand_mx", reuse_today=True
)
SELECTOR_FEATURES_PATH = PROJECT_ROOT / "analog_holidays" / "holidays" / "holiday_selector_features.csv"
CLUSTER_COLUMN = "analog_cluster"

UNIQUE_IDS = analog_holidays_module.get_available_unique_ids(SOURCE_PATH)
UNIQUE_ID = UNIQUE_IDS[0]

# ── Fixed production baseline knobs (notebook In[3]) ──────────────────────────
SPECIAL_LABELS = ("holiday",)
FORECAST_START_OFFSET_HOURS = 14
SEASON_LENGTH = 38
K = None
OPTUNA_MIN_K = 2
OPTUNA_MIN_K_BY_CLUSTER = {"F": 2, "G": 2, "H": 2}
# Per-cluster k ceiling (None = no cap beyond the dynamic pool/24 bound). Used to test
# whether capping k on the deep/full cluster avoids the k>=7 blow-ups on deep holidays.
OPTUNA_MAX_K_BY_CLUSTER = {}
TYPEDIST = "pearson"
TYPEREG = "PCR"
SCALE_METHOD = None
OPTUNA_TYPEDIST_CHOICES = ["pearson", "euclidian"]
OPTUNA_SCALE_METHOD_CHOICES = [None, "standard", "minmax"]
N_COMPONENTS = 2
REGRESSOR_PARAMS = {}
LEVELS = [50, 80, 95]
MIN_SPECIAL_POINTS = 24
# NOTE: production / all prior experiments (incl. the 4.03% deep baseline 14_21_46) use 24.
# The working-tree notebook has an uncommitted 12; we anchor to 24 for comparability.
MIN_EVENT_GAP = 24
MAX_EVENTS = None
RECENT_WEEKEND_ANALOGS = 0
MATCH_TARGET_CLUSTER = True
CLUSTER_FILTER_LABEL = "observance_tier"
OPTUNA_N_TRIALS = 25
OPTUNA_TIMEOUT_SEC = 300
OPTUNA_MAX_EVAL_DATES = 19
OPTUNA_RANDOM_SEED = 42
# In[8] overrides this to 4 (intraday bias factor uses up to 4 analogs).
HOURLY_FACTOR_ANALOGS = 4
BIAS_HEAD_HOURS = int(FORECAST_START_OFFSET_HOURS)
BIAS_TAIL_HOURS = int(SEASON_LENGTH - BIAS_HEAD_HOURS)

# ── Deep / Mode-A target subset (by demand-drop depth, not selector cluster) ──
DEEP_TARGET_DATES = [
    ("2025-01-01", "New Year's Day"),
    ("2025-09-16", "Independence Day"),
    ("2025-12-24", "Christmas Eve"),
    ("2025-12-25", "Christmas Day"),
    ("2025-12-31", "New Year's Eve"),
    ("2026-01-01", "New Year's Day"),
]

# selector lookups (per series) -- notebook In[7]
selector_features_df = pd.read_csv(SELECTOR_FEATURES_PATH, parse_dates=["date"])
selector_features_df["date"] = pd.to_datetime(selector_features_df["date"]).dt.normalize()
selector_features_df["unique_id"] = selector_features_df["unique_id"].astype(str)
series_unique_ids = [str(u) for u in UNIQUE_IDS]
selector_cluster_lookup_by_id = {
    uid: (
        selector_features_df.loc[selector_features_df["unique_id"] == uid]
        .dropna(subset=[CLUSTER_COLUMN])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")[CLUSTER_COLUMN]
        .to_dict()
    )
    for uid in series_unique_ids
}
selector_anchor_lookup_by_id = {
    uid: (
        selector_features_df.loc[selector_features_df["unique_id"] == uid]
        .dropna(subset=["anchor_holiday_name"])
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")["anchor_holiday_name"]
        .to_dict()
    )
    for uid in series_unique_ids
}


def _summary_param(summary_df, param_name, default=np.nan):
    matches = summary_df.loc[summary_df["param"] == param_name, "value"]
    return matches.iloc[0] if not matches.empty else default


# ── Panel-metric helpers (notebook In[8]) ─────────────────────────────────────
def _mean_abs_error(actual, forecast):
    return float(np.mean(np.abs(actual - forecast)))


def _mean_ape_pct(actual, forecast):
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    ape_pct = np.abs(actual - forecast) / denom * 100.0
    return float(np.nanmean(ape_pct)) if np.isfinite(ape_pct).any() else np.nan


def _mean_pct_error(actual, forecast):
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    pe_pct = (actual - forecast) / denom * 100.0
    return float(np.nanmean(pe_pct)) if np.isfinite(pe_pct).any() else np.nan


def _mean_bias(actual, forecast):
    return float(np.mean(actual - forecast))


def _build_window_metrics(actual, forecast, label):
    return {
        f"mae_{label}": _mean_abs_error(actual, forecast),
        f"mape_{label}_pct": _mean_ape_pct(actual, forecast),
        f"mpe_{label}_pct": _mean_pct_error(actual, forecast),
        f"bias_{label}": _mean_bias(actual, forecast),
    }


def _apply_hourly_factor_model(forecast_profile, factor_model, expected_hours):
    forecast_profile = np.asarray(forecast_profile, dtype=np.float64)
    resolved_expected_hours = int(expected_hours)
    if forecast_profile.shape[0] != resolved_expected_hours:
        raise ValueError(f"Expected {resolved_expected_hours}-h forecast, got {forecast_profile.shape}.")
    forecast_window_mean = float(np.mean(forecast_profile))
    if factor_model["train_samples"] <= 0:
        return forecast_profile.copy(), forecast_window_mean
    adjusted = forecast_window_mean * (1.0 + factor_model["hourly_factors"])
    return adjusted.astype(np.float64), forecast_window_mean


def _enrich_panel_metrics(rolling_daily_table, rolling_runs):
    """Notebook In[8] core: bias-adjustment + per-band (14/24/38) panel metrics."""
    bias_rows = []
    for unique_id, series_runs in rolling_runs.items():
        for target_date in sorted(series_runs, key=lambda v: pd.Timestamp(v)):
            run = series_runs[target_date]
            if run.actual_profile is None or len(run.forecast_profile) != SEASON_LENGTH:
                continue
            full_actual = run.actual_profile.copy()
            full_forecast = run.forecast_profile.copy()
            head_actual, head_forecast = full_actual[:BIAS_HEAD_HOURS], full_forecast[:BIAS_HEAD_HOURS]
            tail_actual, tail_forecast = full_actual[BIAS_HEAD_HOURS:], full_forecast[BIAS_HEAD_HOURS:]

            factor_model = analog_holidays_module.fit_hourly_bias_factor_model(
                neighbor_profiles=run.neighbors2, window_hours=SEASON_LENGTH, max_analogs=HOURLY_FACTOR_ANALOGS
            )
            adjusted_full_forecast, fwm38 = _apply_hourly_factor_model(full_forecast, factor_model, SEASON_LENGTH)
            adj_head = adjusted_full_forecast[:BIAS_HEAD_HOURS]
            adj_tail = adjusted_full_forecast[BIAS_HEAD_HOURS:]
            pred_bias_profile = adjusted_full_forecast - full_forecast
            pred_head = float(np.mean(pred_bias_profile[:BIAS_HEAD_HOURS]))
            pred_tail = float(np.mean(pred_bias_profile[BIAS_HEAD_HOURS:]))
            pred_full = float(np.mean(pred_bias_profile))

            m14 = _build_window_metrics(head_actual, head_forecast, "14")
            m24 = _build_window_metrics(tail_actual, tail_forecast, "24")
            m38 = _build_window_metrics(full_actual, full_forecast, "38")
            m14a = _build_window_metrics(head_actual, adj_head, "14_bias_adjusted")
            m24a = _build_window_metrics(tail_actual, adj_tail, "24_bias_adjusted")
            m38a = _build_window_metrics(full_actual, adjusted_full_forecast, "38_bias_adjusted")

            bias_rows.append({
                "unique_id": unique_id,
                "target_date": target_date,
                "bias_head_hours": BIAS_HEAD_HOURS,
                "bias_tail_hours": BIAS_TAIL_HOURS,
                "hourly_factor_analog_count": factor_model["train_samples"],
                "hourly_factor_requested_analogs": factor_model["requested_analogs"],
                "hourly_factor_available_neighbors": factor_model["available_neighbor_profiles"],
                "hourly_factor_selected_analogs": factor_model["selected_analogs"],
                "hourly_factor_mean_abs": factor_model["factor_mean_abs"],
                "forecast_window_mean_38": fwm38,
                "forecast_daily_mean_24": float(np.mean(tail_forecast)),
                **m38, **m14, **m24, **m14a, **m24a, **m38a,
                "head_bias_mean": m14["bias_14"],
                "tail_bias_mean": m24["bias_24"],
                "bias_model_method": factor_model["method"],
                "bias_train_samples": factor_model["train_samples"],
                "predicted_head_bias_mean": pred_head,
                "predicted_tail_bias_mean": pred_tail,
                "predicted_full_bias_mean": pred_full,
                "predicted_bias_14": pred_head,
                "predicted_bias_24": pred_tail,
                "predicted_bias_38": pred_full,
                "bias_14_prediction_error": pred_head - m14["bias_14"],
                "bias_24_prediction_error": pred_tail - m24["bias_24"],
                "bias_38_prediction_error": pred_full - m38["bias_38"],
                "mae_head14": m14["mae_14"],
                "mape_head14_pct": m14["mape_14_pct"],
                "mae_head14_bias_adjusted": m14a["mae_14_bias_adjusted"],
                "mape_head14_bias_adjusted_pct": m14a["mape_14_bias_adjusted_pct"],
                "mae_holiday24_raw": m24["mae_24"],
                "mape_holiday24_raw_pct": m24["mape_24_pct"],
                "mae_holiday24_bias_adjusted": m24a["mae_24_bias_adjusted"],
                "mape_holiday24_bias_adjusted_pct": m24a["mape_24_bias_adjusted_pct"],
                "mae_window_bias_adjusted": m38a["mae_38_bias_adjusted"],
                "mape_window_bias_adjusted_pct": m38a["mape_38_bias_adjusted_pct"],
            })

    bias_df = pd.DataFrame(bias_rows)
    if bias_df.empty:
        raise ValueError("No complete rolling runs to build panel metrics.")
    out = rolling_daily_table.merge(bias_df, on=["unique_id", "target_date"], how="left", validate="one_to_one")
    out["mae_holiday24_improvement"] = out["mae_holiday24_raw"] - out["mae_holiday24_bias_adjusted"]
    out["mape_holiday24_improvement_pct"] = out["mape_holiday24_raw_pct"] - out["mape_holiday24_bias_adjusted_pct"]
    out["mape_window_improvement_pct"] = out["mape_window_pct"] - out["mape_window_bias_adjusted_pct"]
    return out


def run_variant(typereg_choices, slug, target_items=None, min_event_gap=None, config_extra=None):
    """Run the rolling loop + enrichment + logging for one config.

    Parametrized so the same machinery serves the deep regression sweep and other
    one-axis experiments (e.g. MIN_EVENT_GAP). Defaults reproduce the deep sweep.
    """
    t0 = time.time()
    items = DEEP_TARGET_DATES if target_items is None else target_items
    gap = MIN_EVENT_GAP if min_event_gap is None else min_event_gap
    rolling_target_items = [
        (pd.Timestamp(d).date().isoformat(), label) for d, label in items
    ]
    rolling_runs, rolling_rows = {}, []

    for unique_id in series_unique_ids:
        series_cluster_lookup = selector_cluster_lookup_by_id.get(unique_id, {})
        series_anchor_lookup = selector_anchor_lookup_by_id.get(unique_id, {})
        series_runs = {}
        for target_date, holiday_label in rolling_target_items:
            target_ts = pd.Timestamp(target_date).normalize()
            target_cluster = series_cluster_lookup.get(target_ts, pd.NA)
            anchor = series_anchor_lookup.get(target_ts, holiday_label)
            anchor = holiday_label if pd.isna(anchor) else str(anchor)
            target_min_k = OPTUNA_MIN_K_BY_CLUSTER.get(str(target_cluster), OPTUNA_MIN_K)
            target_max_k = OPTUNA_MAX_K_BY_CLUSTER.get(str(target_cluster), None)
            try:
                tuning_result = tune_analog_holidays_optuna(
                    unique_id=unique_id, source_path=SOURCE_PATH, train_end=target_ts,
                    season_length=SEASON_LENGTH, forecast_start_offset_hours=FORECAST_START_OFFSET_HOURS,
                    initial_k=K, initial_typedist=TYPEDIST, initial_typereg=TYPEREG,
                    typedist_choices=OPTUNA_TYPEDIST_CHOICES, typereg_choices=typereg_choices,
                    scale_method=SCALE_METHOD, scale_method_choices=OPTUNA_SCALE_METHOD_CHOICES,
                    initial_n_components=N_COMPONENTS, initial_regressor_params=REGRESSOR_PARAMS,
                    optuna_min_k=target_min_k, optuna_max_k=target_max_k,
                    n_trials=OPTUNA_N_TRIALS, timeout_sec=OPTUNA_TIMEOUT_SEC,
                    max_eval_dates=OPTUNA_MAX_EVAL_DATES, random_seed=OPTUNA_RANDOM_SEED,
                    special_labels=SPECIAL_LABELS, min_special_points=MIN_SPECIAL_POINTS,
                    min_event_gap=gap, max_events=MAX_EVENTS,
                    selector_features_path=SELECTOR_FEATURES_PATH, cluster_column=CLUSTER_COLUMN,
                    match_target_cluster=MATCH_TARGET_CLUSTER, recent_weekend_analogs=RECENT_WEEKEND_ANALOGS,
                )
                best = tuning_result.best_config
                best_k_range = tuple(best.get("k_range", (np.nan, np.nan)))
                run = run_analog_holidays(
                    unique_id=unique_id, target_date=target_ts, source_path=SOURCE_PATH,
                    season_length=SEASON_LENGTH, forecast_start_offset_hours=FORECAST_START_OFFSET_HOURS,
                    k=int(best["k"]), typedist=str(best["typedist"]), typereg=str(best["typereg"]),
                    scale_method=best.get("scale_method", SCALE_METHOD), n_components=int(best["n_components"]),
                    regressor_params=dict(best.get("regressor_params", {})), levels=LEVELS,
                    special_labels=SPECIAL_LABELS, min_special_points=MIN_SPECIAL_POINTS,
                    min_event_gap=gap, max_events=MAX_EVENTS, expected_target_label=None,
                    selector_features_path=SELECTOR_FEATURES_PATH, cluster_column=CLUSTER_COLUMN,
                    match_target_cluster=MATCH_TARGET_CLUSTER, recent_weekend_analogs=RECENT_WEEKEND_ANALOGS,
                )
                series_runs[target_date] = run
                mae_w = mape_w = np.nan
                if run.actual_profile is not None:
                    mae_w = float(np.mean(np.abs(run.forecast_profile - run.actual_profile)))
                    denom = np.where(np.abs(run.actual_profile) > 1e-9, np.abs(run.actual_profile), np.nan)
                    ape = np.abs(run.forecast_profile - run.actual_profile) / denom * 100.0
                    mape_w = float(np.nanmean(ape)) if np.isfinite(ape).any() else np.nan
                rolling_rows.append({
                    "unique_id": unique_id, "target_date": target_date, "holiday_label": holiday_label,
                    "analog_cluster": target_cluster, "cluster_filter_label": CLUSTER_FILTER_LABEL,
                    "filter_by_cluster": bool(MATCH_TARGET_CLUSTER), "train_end": target_date,
                    "eligible_tuning_dates": len(tuning_result.eligible_dates),
                    "target_exists": run.target_exists, "target_has_complete_profile": run.target_has_complete_profile,
                    "selected_analogs": len(run.positions), "fail": run.fail,
                    "optuna_k_min": best_k_range[0], "optuna_k_max": best_k_range[1], "k": run.k,
                    "typedist": run.typedist, "typereg": run.typereg, "scale_method": run.scale_method,
                    "n_components": run.n_components, "regressor_params": dict(best.get("regressor_params", {})),
                    "forecast_start": run.forecast_start, "forecast_end": run.forecast_end,
                    "mae_window": mae_w, "mape_window_pct": mape_w,
                    "tuning_best_mean_mae": _summary_param(tuning_result.summary_df, "best_mean_mae"),
                    "tuning_best_mean_mape_pct": _summary_param(tuning_result.summary_df, "best_mean_mape_pct"),
                    "error": None,
                })
                print(f"[{slug}] {unique_id} {target_date} {anchor}: typereg={run.typereg} k={run.k} MAPE38={mape_w:.2f}%", flush=True)
            except Exception as exc:
                rolling_rows.append({
                    "unique_id": unique_id, "target_date": target_date, "holiday_label": holiday_label,
                    "analog_cluster": target_cluster, "cluster_filter_label": CLUSTER_FILTER_LABEL,
                    "filter_by_cluster": bool(MATCH_TARGET_CLUSTER), "train_end": target_date,
                    "fail": True, "k": np.nan, "typedist": pd.NA, "typereg": pd.NA, "scale_method": pd.NA,
                    "n_components": np.nan, "regressor_params": {}, "mae_window": np.nan, "mape_window_pct": np.nan,
                    "error": str(exc),
                })
                print(f"[{slug}] {unique_id} {target_date} ERROR: {exc}", flush=True)
        rolling_runs[unique_id] = series_runs

    rolling_daily_table = pd.DataFrame(rolling_rows)
    rolling_daily_table = _enrich_panel_metrics(rolling_daily_table, rolling_runs)

    batch_results = {}
    for uid in series_unique_ids:
        sub = rolling_daily_table.loc[rolling_daily_table["unique_id"] == uid].reset_index(drop=True)
        if sub.empty:
            continue
        batch_results[uid] = analog_holidays_module.AnalogHolidayBatchResult(
            target_items=rolling_target_items, runs=rolling_runs.get(uid, {}),
            results_df=sub, metric_summary_df=sub.describe(include="all"),
        )

    config = {
        "window": {"season_length": SEASON_LENGTH, "forecast_start_offset_hours": FORECAST_START_OFFSET_HOURS,
                   "special_labels": list(SPECIAL_LABELS)},
        "cluster": {"use_cluster": True, "match_target_cluster": MATCH_TARGET_CLUSTER, "cluster_column": CLUSTER_COLUMN},
        "analog_selection": {"typedist": TYPEDIST, "k": K, "min_special_points": MIN_SPECIAL_POINTS,
                             "min_event_gap": gap, "max_events": MAX_EVENTS,
                             "recent_weekend_analogs": RECENT_WEEKEND_ANALOGS},
        "regression": {"typereg": TYPEREG, "scale_method": SCALE_METHOD, "n_components": N_COMPONENTS,
                       "regressor_params": REGRESSOR_PARAMS, "levels": LEVELS},
        "tuning": {"optuna_min_k": OPTUNA_MIN_K, "optuna_min_k_by_cluster": OPTUNA_MIN_K_BY_CLUSTER,
                   "optuna_max_k_by_cluster": dict(OPTUNA_MAX_K_BY_CLUSTER),
                   "n_trials": OPTUNA_N_TRIALS, "timeout_sec": OPTUNA_TIMEOUT_SEC,
                   "typedist_choices": OPTUNA_TYPEDIST_CHOICES, "typereg_choices": typereg_choices,
                   "scale_method_choices": OPTUNA_SCALE_METHOD_CHOICES, "max_eval_dates": OPTUNA_MAX_EVAL_DATES},
        "training_cutoff": "rolling", "history_start": None, "source_path": str(SOURCE_PATH),
    }
    if config_extra:
        config.update(config_extra)
    exp_dir = save_experiment_run(
        config=config, batch_results=batch_results, figures={},
        selector_features_path=SELECTOR_FEATURES_PATH, slug=slug, verbose=True,
    )
    # quick read-out
    med = pd.to_numeric(rolling_daily_table["mape_24_pct"], errors="coerce").median()
    mean = pd.to_numeric(rolling_daily_table["mape_24_pct"], errors="coerce").mean()
    picks = rolling_daily_table["typereg"].value_counts().to_dict()
    print(f"=== {slug}: median mape_24={med:.3f}% mean={mean:.3f}% | regressor picks={picks} | {time.time()-t0:.0f}s ===", flush=True)
    return exp_dir.name, float(med), float(mean), picks


VARIANTS = [
    (["PCR", "PLS", "RidgeReg", "LassoReg"], "deep_reg_control_baselinesearch"),
    (["PCR"], "deep_reg_PCR"),
    (["PLS"], "deep_reg_PLS"),
    (["RidgeReg"], "deep_reg_RidgeReg"),
    (["LassoReg"], "deep_reg_LassoReg"),
    (["PCR", "PLS", "RidgeReg", "LassoReg", "RF"], "deep_reg_RF_enriched"),
]

_DEEP_CONFIG_EXTRA = {
    "target_dates_var": "DEEP_TARGET_DATES",
    "experiment_family": "deep_regression_sweep",
    "baseline_compare": "experiment_2026_06_14_21_46 (deep subset, median mape_24=4.03%)",
    "deep_subset": [d for d, _ in DEEP_TARGET_DATES],
}

if __name__ == "__main__":
    print(f"Deep regression sweep | {len(series_unique_ids)} series x {len(DEEP_TARGET_DATES)} deep dates", flush=True)
    results = []
    for choices, slug in VARIANTS:
        name, med, mean, picks = run_variant(choices, slug, config_extra=_DEEP_CONFIG_EXTRA)
        results.append((slug, name, med, mean, picks))
    print("\n================ DEEP REGRESSION SWEEP SUMMARY ================")
    print("baseline (14_21_46 deep subset): median 4.028% | mean 5.748%")
    for slug, name, med, mean, picks in results:
        print(f"  {slug:34s} median={med:6.3f}%  mean={mean:6.3f}%  -> {name}")
