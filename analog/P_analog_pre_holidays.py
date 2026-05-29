"""Forecast pre-holiday hourly windows using standard AnalogKNN.

For each holiday timestamp listed by the caller this module:

1. Builds the historical hourly series for every ``unique_id`` available in
   ``pre_holiday_demand_mx.csv`` (one column per series, with ``ds`` as the
   hourly index).
2. Forecasts the ``previously_w_hours`` immediately before the holiday using
   :class:`analog_holidays.analog.analog.AnalogKNN` (the standard analog
   model, *not* :class:`AnalogSpecialDays`).
3. Writes a copy of the source CSV called
   ``pre_holiday_demand_mx_YYYY_MM_DD_HH_MM.csv`` with the forecast values
   (rounded to integers) overwriting the corresponding hourly rows.

The module also exposes a small Optuna helper that tunes ``AnalogKNN``
hyperparameters by back-testing on historical holidays flagged in the
``<unique_id>_holiday`` companion columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analog_special_days import AnalogSpecialDays
from analog_holidays.shared.identify_holidays import load_selector_cluster_lookup


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PACKAGE_ROOT / "holidays" / "pre_holiday_demand_mx.csv"
DEFAULT_SELECTOR_FEATURES_PATH = PACKAGE_ROOT / "holidays" / "holiday_selector_features.csv"


# =====================================================================
# Dataclasses
# =====================================================================


@dataclass
class PreHolidayRun:
    """Single (unique_id, target_date) pre-holiday forecast."""

    unique_id: str
    target_date: pd.Timestamp
    holiday_label: Optional[str]
    pre_holiday_start: pd.Timestamp
    pre_holiday_end: pd.Timestamp
    previously_w_hours: int
    forecast: np.ndarray
    forecast_int: np.ndarray
    actual: Optional[np.ndarray]
    history_length: int
    fail: bool


@dataclass
class PreHolidayBatchResult:
    """Batch result aggregating one PreHolidayRun per (target, unique_id)."""

    runs: Dict[Tuple[str, str], PreHolidayRun]
    results_df: pd.DataFrame
    forecasts_long_df: pd.DataFrame
    output_path: Path
    source_path: Path
    config: Dict[str, object] = field(default_factory=dict)


@dataclass
class PreHolidayOptunaResult:
    best_config: Dict[str, object]
    summary_df: pd.DataFrame
    fold_metrics_df: pd.DataFrame
    eligible_dates: List[pd.Timestamp]
    study: object


# =====================================================================
# Source helpers
# =====================================================================


def load_pre_holiday_source(
    source_path: Path | str = DEFAULT_SOURCE_PATH,
) -> pd.DataFrame:
    """Load the hourly wide pre-holiday CSV."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Pre-holiday source not found: {source_path}")
    if source_path.name == "holiday_demand_mx_analog_cluster.csv":
        raise ValueError(
            "holiday_demand_mx_analog_cluster.csv is no longer a valid historical "
            "demand source for AnalogSpecialDays. Use holiday_demand_mx.csv for demand "
            "history and holiday_selector_features.csv for analog_cluster lookups."
        )

    df = pd.read_csv(source_path, parse_dates=["ds"])
    if "ds" not in df.columns:
        raise ValueError("Source CSV must include a 'ds' column.")
    return df.sort_values("ds").reset_index(drop=True)


def list_unique_ids(df: pd.DataFrame) -> list[str]:
    """Return value-series columns (excluding ``ds`` and helper columns)."""
    return [
        c for c in df.columns
        if c != "ds" and not c.endswith("_holiday") and not c.endswith("_cluster")
    ]


def _resolve_selector_cluster_lookup(
    selector_features_path: Path | str | None,
    cluster_column: str,
    match_target_cluster: bool,
    unique_id: str | None = None,
) -> Optional[dict[pd.Timestamp, object]]:
    if not match_target_cluster:
        return None

    resolved_path = (
        DEFAULT_SELECTOR_FEATURES_PATH
        if selector_features_path is None else Path(selector_features_path)
    )
    return load_selector_cluster_lookup(
        resolved_path,
        cluster_column=cluster_column,
        unique_id=unique_id,
    )


def _get_target_cluster(
    target_date: str | pd.Timestamp,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]],
) -> object:
    target_norm = pd.Timestamp(target_date).normalize()
    target_cluster = (
        pd.NA
        if selector_cluster_lookup is None
        else selector_cluster_lookup.get(target_norm, pd.NA)
    )
    if pd.isna(target_cluster):
        raise ValueError(
            f"No selector analog cluster was found for target_date={target_norm.date()}."
        )
    return target_cluster


def _filter_dates_with_selector_cluster(
    dates: Sequence[str | pd.Timestamp],
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]],
) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """Split dates into those with and without selector clusters."""
    if not selector_cluster_lookup:
        return [pd.Timestamp(date_value) for date_value in dates], []

    kept_dates: List[pd.Timestamp] = []
    skipped_dates: List[pd.Timestamp] = []
    for date_value in dates:
        normalized = pd.Timestamp(date_value).normalize()
        if pd.isna(selector_cluster_lookup.get(normalized, pd.NA)):
            skipped_dates.append(normalized)
            continue
        kept_dates.append(pd.Timestamp(date_value))
    return kept_dates, skipped_dates


def _build_pre_holiday_mask(
    df_hist: pd.DataFrame,
    unique_id: str,
    previously_w_hours: int,
    target_date: str | pd.Timestamp | None = None,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    match_target_cluster: bool = False,
) -> np.ndarray:
    """Binary mask aligned with ``df_hist``: 1 for hours in [holiday − previously_w_hours, holiday).

    ``AnalogSpecialDays`` uses this mask to restrict X2 candidates to
    windows that overlap with a historical pre-holiday period. When
    ``match_target_cluster`` is true, only historical holidays whose selector
    ``analog_cluster`` matches the target date are kept.
    """
    holiday_col = f"{unique_id}_holiday"
    if holiday_col not in df_hist.columns:
        raise ValueError(
            f"Column '{holiday_col}' not found. "
            "Source CSV must include '<uid>_holiday' flag columns."
        )
    df_h = df_hist[["ds", holiday_col]].copy()
    df_h["_date"] = pd.to_datetime(df_h["ds"]).dt.normalize()
    holiday_dates = (
        df_h.groupby("_date")[holiday_col].max()
        .loc[lambda s: s == 1]
        .index
    )

    if match_target_cluster:
        if target_date is None:
            raise ValueError("target_date is required when match_target_cluster=True.")
        target_cluster = _get_target_cluster(target_date, selector_cluster_lookup)
        holiday_dates = [
            pd.Timestamp(date_value)
            for date_value in holiday_dates
            if selector_cluster_lookup is not None
            and pd.notna(
                selector_cluster_lookup.get(pd.Timestamp(date_value).normalize(), pd.NA)
            )
            and selector_cluster_lookup.get(pd.Timestamp(date_value).normalize(), pd.NA) == target_cluster
        ]
        if not holiday_dates:
            raise ValueError(
                f"No historical pre-holiday events share analog cluster {target_cluster!r} "
                f"before target_date={pd.Timestamp(target_date).normalize().date()}."
            )

    mask = np.zeros(len(df_hist), dtype=np.float64)
    for d in holiday_dates:
        pre_end = pd.Timestamp(d)
        pre_start_dt = pre_end - pd.Timedelta(hours=previously_w_hours)
        window = (df_hist["ds"] >= pre_start_dt) & (df_hist["ds"] < pre_end)
        mask[window.to_numpy()] = 1.0
    return mask


# =====================================================================
# Single forecast
# =====================================================================


def run_analog_pre_holiday(
    unique_id: str,
    target_date: str | pd.Timestamp,
    df_source: pd.DataFrame,
    previously_w_hours: int = 14,
    season_length: int = 24,
    k: int = 10,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    n_components: int = 3,
    typedist: str = "pearson",
    typereg: str = "PCR",
    holiday_label: Optional[str] = None,
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
) -> PreHolidayRun:
    """Forecast the ``previously_w_hours`` window before ``target_date``.

    Uses :class:`AnalogSpecialDays`: candidate X2 blocks are restricted to
    historical windows that overlap with a pre-holiday period (hours in
    ``[holiday − previously_w_hours, holiday)``), rather than selecting by
    pure time-series similarity as in the classic ``AnalogKNN``.

    Parameters
    ----------
    min_special_points:
        Minimum number of pre-holiday hours required inside a 24-h X2 block
        for it to be considered a valid candidate.  ``None`` defaults to
        ``previously_w_hours`` (require the full pre-holiday window).
    min_event_gap:
        Minimum number of hours between the starts of two selected events.
        ``None`` uses ``season_length``.
    max_events:
        Maximum number of historical pre-holiday events to use.
    """
    if unique_id not in df_source.columns:
        raise KeyError(f"Series '{unique_id}' not in source CSV columns.")

    target_ts = pd.Timestamp(target_date).normalize()
    pre_end = target_ts  # exclusive: forecast hours strictly before 00:00
    pre_start = pre_end - pd.Timedelta(hours=previously_w_hours)

    history_mask = df_source["ds"] < pre_start
    df_hist = df_source.loc[history_mask].copy()
    history = df_hist[unique_id].to_numpy(dtype=np.float64)
    if np.isnan(history).any():
        history = pd.Series(history).interpolate(limit_direction="both").to_numpy()

    if len(history) < 2 * season_length + 1:
        raise ValueError(
            f"Insufficient history for '{unique_id}' before {pre_start} "
            f"(have {len(history)}, need >= {2 * season_length + 1})."
        )

    effective_min_points = (
        previously_w_hours if min_special_points is None else min_special_points
    )
    resolved_cluster_lookup = selector_cluster_lookup
    if match_target_cluster and resolved_cluster_lookup is None:
        resolved_cluster_lookup = _resolve_selector_cluster_lookup(
            selector_features_path=selector_features_path,
            cluster_column=cluster_column,
            match_target_cluster=match_target_cluster,
            unique_id=unique_id,
        )
    pre_holiday_mask = _build_pre_holiday_mask(
        df_hist=df_hist,
        unique_id=unique_id,
        previously_w_hours=previously_w_hours,
        target_date=target_ts,
        selector_cluster_lookup=resolved_cluster_lookup,
        match_target_cluster=match_target_cluster,
    )

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        min_special_points=effective_min_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )
    model.fit(y=history, special_days=pre_holiday_mask)
    result = model.predict(h=previously_w_hours)
    forecast = np.asarray(result["mean"], dtype=np.float64)[:previously_w_hours]
    forecast_int = np.rint(forecast).astype(np.int64)

    window_mask = (df_source["ds"] >= pre_start) & (df_source["ds"] < pre_end)
    actual = None
    if int(window_mask.sum()) == previously_w_hours:
        candidate = df_source.loc[window_mask, unique_id].to_numpy(dtype=np.float64)
        if not np.isnan(candidate).any():
            actual = candidate

    return PreHolidayRun(
        unique_id=unique_id,
        target_date=target_ts,
        holiday_label=holiday_label,
        pre_holiday_start=pre_start,
        pre_holiday_end=pre_end,
        previously_w_hours=previously_w_hours,
        forecast=forecast,
        forecast_int=forecast_int,
        actual=actual,
        history_length=int(len(history)),
        fail=False,
    )


# =====================================================================
# Batch + CSV writer
# =====================================================================


def _normalize_target_items(
    target_dates: Sequence,
) -> List[Tuple[pd.Timestamp, Optional[str]]]:
    items: List[Tuple[pd.Timestamp, Optional[str]]] = []
    for entry in target_dates:
        if isinstance(entry, (tuple, list)):
            date = entry[0]
            label = entry[1] if len(entry) > 1 else None
        else:
            date, label = entry, None
        items.append((pd.Timestamp(date).normalize(), label))
    return items


def _build_output_path(
    source_path: Path,
    output_dir: Optional[Path | str],
    output_path: Optional[Path | str],
) -> Path:
    if output_path is not None:
        return Path(output_path)
    base_dir = Path(output_dir) if output_dir is not None else source_path.parent
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    return base_dir / f"pre_holiday_demand_mx_{timestamp}.csv"


def run_analog_pre_holidays_batch(
    target_dates: Sequence,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    unique_ids: Optional[Sequence[str]] = None,
    previously_w_hours: int = 14,
    season_length: int = 24,
    k: int = 10,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    n_components: int = 3,
    typedist: str = "pearson",
    typereg: str = "PCR",
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    output_dir: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    write_csv: bool = True,
) -> PreHolidayBatchResult:
    """Forecast pre-holiday windows for every (target_date, unique_id) pair.

    Uses :class:`AnalogSpecialDays`: candidate X2 blocks are restricted to
    historical pre-holiday windows.  A copy of the source CSV is written with
    the rounded integer forecast values overwriting the pre-holiday rows.
    """
    source_path = Path(source_path)
    df_source = load_pre_holiday_source(source_path)

    if unique_ids is None:
        resolved_ids = list_unique_ids(df_source)
    else:
        resolved_ids = [str(u) for u in unique_ids]

    target_items = _normalize_target_items(target_dates)
    selector_cluster_lookup_by_id = {
        uid: _resolve_selector_cluster_lookup(
            selector_features_path=selector_features_path,
            cluster_column=cluster_column,
            match_target_cluster=match_target_cluster,
            unique_id=uid,
        )
        for uid in resolved_ids
    }
    df_out = df_source.copy()
    runs: Dict[Tuple[str, str], PreHolidayRun] = {}
    rows: List[dict] = []
    long_records: List[dict] = []

    for target_ts, holiday_label in target_items:
        date_key = target_ts.strftime("%Y-%m-%d")
        for uid in resolved_ids:
            try:
                run = run_analog_pre_holiday(
                    unique_id=uid,
                    target_date=target_ts,
                    df_source=df_source,
                    previously_w_hours=previously_w_hours,
                    season_length=season_length,
                    k=k,
                    min_special_points=min_special_points,
                    min_event_gap=min_event_gap,
                    max_events=max_events,
                    n_components=n_components,
                    typedist=typedist,
                    typereg=typereg,
                    holiday_label=holiday_label,
                    selector_features_path=selector_features_path,
                    cluster_column=cluster_column,
                    match_target_cluster=match_target_cluster,
                    selector_cluster_lookup=selector_cluster_lookup_by_id[uid],
                )
            except Exception as exc:
                rows.append({
                    "target_date": target_ts,
                    "holiday_label": holiday_label,
                    "unique_id": uid,
                    "fail": True,
                    "error": str(exc),
                    "mae": np.nan,
                    "mape_pct": np.nan,
                })
                continue

            runs[(date_key, uid)] = run

            window_mask = (
                (df_out["ds"] >= run.pre_holiday_start)
                & (df_out["ds"] < run.pre_holiday_end)
            )
            n_match = int(window_mask.sum())
            if n_match > 0:
                df_out.loc[window_mask, uid] = run.forecast_int[:n_match]

            mae = np.nan
            mape = np.nan
            if run.actual is not None:
                diff = run.forecast - run.actual
                mae = float(np.mean(np.abs(diff)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = np.where(run.actual == 0, np.nan, run.actual)
                    mape = float(
                        np.nanmean(np.abs(diff / denom)) * 100.0
                    )

            rows.append({
                "target_date": target_ts,
                "holiday_label": holiday_label,
                "unique_id": uid,
                "fail": False,
                "error": None,
                "mae": mae,
                "mape_pct": mape,
                "hours_written": n_match,
            })

            ds_range = pd.date_range(
                run.pre_holiday_start,
                periods=previously_w_hours,
                freq="h",
            )
            for ts, val_float, val_int in zip(
                ds_range, run.forecast, run.forecast_int
            ):
                long_records.append({
                    "ds": ts,
                    "unique_id": uid,
                    "target_date": target_ts,
                    "forecast": float(val_float),
                    "forecast_int": int(val_int),
                })

    results_df = pd.DataFrame(rows)
    forecasts_long_df = pd.DataFrame(long_records)

    resolved_output_path = _build_output_path(source_path, output_dir, output_path)
    if write_csv:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(resolved_output_path, index=False)

    resolved_selector_path = None
    if match_target_cluster:
        resolved_selector_path = str(
            DEFAULT_SELECTOR_FEATURES_PATH if selector_features_path is None else selector_features_path
        )

    config = {
        "previously_w_hours": previously_w_hours,
        "season_length": season_length,
        "k": k,
        "min_special_points": min_special_points,
        "min_event_gap": min_event_gap,
        "max_events": max_events,
        "n_components": n_components,
        "typedist": typedist,
        "typereg": typereg,
        "unique_ids": resolved_ids,
        "match_target_cluster": match_target_cluster,
        "cluster_column": cluster_column,
        "selector_features_path": resolved_selector_path,
    }

    return PreHolidayBatchResult(
        runs=runs,
        results_df=results_df,
        forecasts_long_df=forecasts_long_df,
        output_path=resolved_output_path,
        source_path=source_path,
        config=config,
    )


# =====================================================================
# Optuna tuner (per unique_id, back-tested on historical holidays)
# =====================================================================


def _detect_historical_holiday_dates(
    df_source: pd.DataFrame,
    unique_id: str,
    train_end: pd.Timestamp,
) -> List[pd.Timestamp]:
    holiday_col = f"{unique_id}_holiday"
    if holiday_col not in df_source.columns:
        raise ValueError(f"Source CSV missing '{holiday_col}' column.")

    df_h = df_source[["ds", holiday_col]].copy()
    df_h["date"] = df_h["ds"].dt.normalize()
    daily = df_h.groupby("date")[holiday_col].max()
    dates = [d for d, flag in daily.items() if flag == 1 and d < train_end]
    return dates


def tune_analog_pre_holidays_optuna(
    unique_id: str,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    train_end: str | pd.Timestamp = "2024-01-01",
    previously_w_hours: int = 14,
    season_length: int = 24,
    initial_n_components: int = 3,
    n_trials: int = 25,
    timeout_sec: Optional[int] = 900,
    max_eval_dates: Optional[int] = 12,
    random_seed: int = 42,
    typedist_choices: Optional[Sequence[str]] = None,
    typereg_choices: Optional[Sequence[str]] = None,
    k_min: int = 2,
    k_max: int = 30,
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = False,
) -> PreHolidayOptunaResult:
    """Tune AnalogKNN hyperparameters by back-testing pre-holiday windows."""
    import importlib.util

    import optuna

    df_source = load_pre_holiday_source(source_path)
    train_end_ts = pd.Timestamp(train_end)
    selector_cluster_lookup = _resolve_selector_cluster_lookup(
        selector_features_path=selector_features_path,
        cluster_column=cluster_column,
        match_target_cluster=match_target_cluster,
        unique_id=unique_id,
    )

    candidate_dates = _detect_historical_holiday_dates(
        df_source, unique_id, train_end_ts
    )

    eligible: List[pd.Timestamp] = []
    for d in candidate_dates:
        pre_start = d - pd.Timedelta(hours=previously_w_hours)
        history_len = int((df_source["ds"] < pre_start).sum())
        window_count = int(
            ((df_source["ds"] >= pre_start) & (df_source["ds"] < d)).sum()
        )
        if history_len < 2 * season_length + 1 or window_count != previously_w_hours:
            continue
        eligible.append(d)

    if max_eval_dates is not None:
        eligible = eligible[-int(max_eval_dates):]

    if match_target_cluster:
        eligible, _ = _filter_dates_with_selector_cluster(
            eligible,
            selector_cluster_lookup,
        )

    if len(eligible) < 2:
        detail = (
            " with selector analog clusters"
            if match_target_cluster else ""
        )
        raise ValueError(
            f"At least 2 historical holidays with valid windows{detail} are required."
        )

    if typedist_choices is None:
        resolved_typedist_choices = ["pearson", "euclidian"]
        if importlib.util.find_spec("dtw") is not None:
            resolved_typedist_choices.append("dtw")
    else:
        resolved_typedist_choices = [str(c) for c in typedist_choices]

    if typereg_choices is None:
        resolved_typereg_choices = [
            "PCR", "PLS", "RidgeReg", "LassoReg", "BayesRidge", "RF", "Boosting",
        ]
    else:
        resolved_typereg_choices = [str(c) for c in typereg_choices]

    def _evaluate_fold(target_date, k, typedist, typereg, n_components):
        pre_start = target_date - pd.Timedelta(hours=previously_w_hours)
        history_mask = df_source["ds"] < pre_start
        df_hist = df_source.loc[history_mask].copy()
        history = df_hist[unique_id].to_numpy(dtype=np.float64)
        if np.isnan(history).any():
            history = pd.Series(history).interpolate(limit_direction="both").to_numpy()
        actual = df_source.loc[
            (df_source["ds"] >= pre_start) & (df_source["ds"] < target_date),
            unique_id,
        ].to_numpy(dtype=np.float64)
        pre_holiday_mask = _build_pre_holiday_mask(
            df_hist=df_hist,
            unique_id=unique_id,
            previously_w_hours=previously_w_hours,
            target_date=target_date,
            selector_cluster_lookup=selector_cluster_lookup,
            match_target_cluster=match_target_cluster,
        )
        model = AnalogSpecialDays(
            season_length=season_length,
            k=k,
            n_components=n_components,
            typedist=typedist,
            typereg=typereg,
            min_special_points=previously_w_hours,
        )
        model.fit(history, special_days=pre_holiday_mask)
        pred = np.asarray(
            model.predict(h=previously_w_hours)["mean"], dtype=np.float64
        )[:previously_w_hours]
        mae = float(np.mean(np.abs(pred - actual)))
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.where(actual == 0, np.nan, actual)
            mape = float(np.nanmean(np.abs((pred - actual) / denom)) * 100.0)
        return {
            "target_date": target_date,
            "mae": mae,
            "mape_pct": mape,
        }

    def objective(trial):
        typereg = trial.suggest_categorical("typereg", resolved_typereg_choices)
        typedist = trial.suggest_categorical("typedist", resolved_typedist_choices)
        k = trial.suggest_int("k", int(k_min), int(k_max))
        if typereg in {"PCR", "PLS"}:
            max_components = max(2, min(k, season_length))
            n_components = trial.suggest_int("n_components", 2, max_components)
        else:
            n_components = int(initial_n_components)

        try:
            folds = [
                _evaluate_fold(d, k, typedist, typereg, n_components)
                for d in eligible
            ]
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return 1e9

        mean_mae = float(np.mean([f["mae"] for f in folds]))
        mape_vals = [f["mape_pct"] for f in folds if np.isfinite(f["mape_pct"])]
        mean_mape = float(np.mean(mape_vals)) if mape_vals else np.nan
        trial.set_user_attr("fold_metrics", folds)
        trial.set_user_attr("mean_mae", mean_mae)
        trial.set_user_attr("mean_mape_pct", mean_mape)
        return mean_mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

    best = study.best_trial
    best_params = study.best_params
    best_config = {
        "k": int(best_params["k"]),
        "typedist": str(best_params["typedist"]),
        "typereg": str(best_params["typereg"]),
        "n_components": int(
            best_params.get("n_components", initial_n_components)
        ),
    }
    summary_df = pd.DataFrame([
        {"param": "unique_id", "value": unique_id},
        {"param": "cutoff_train", "value": train_end_ts.date().isoformat()},
        {"param": "previously_w_hours", "value": previously_w_hours},
        {"param": "eval_dates", "value": len(eligible)},
        {"param": "best_mean_mae",
         "value": round(best.user_attrs.get("mean_mae", float("nan")), 6)},
        {"param": "best_mean_mape_pct",
         "value": round(best.user_attrs.get("mean_mape_pct", float("nan")), 3)},
        {"param": "K", "value": best_config["k"]},
        {"param": "TYPEDIST", "value": best_config["typedist"]},
        {"param": "TYPEREG", "value": best_config["typereg"]},
        {"param": "N_COMPONENTS", "value": best_config["n_components"]},
    ])
    fold_metrics_df = pd.DataFrame(best.user_attrs.get("fold_metrics", []))

    return PreHolidayOptunaResult(
        best_config=best_config,
        summary_df=summary_df,
        fold_metrics_df=fold_metrics_df,
        eligible_dates=eligible,
        study=study,
    )


# =====================================================================
# Plot helpers
# =====================================================================


def plot_pre_holiday_run(run: PreHolidayRun, ax=None):
    """Plot forecast (and actual if present) for a single PreHolidayRun."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    hours = np.arange(-run.previously_w_hours, 0)
    ax.plot(hours, run.forecast, marker="o", label="forecast", color="tab:red")
    if run.actual is not None:
        ax.plot(hours, run.actual, marker="x", label="actual", color="tab:blue")

    holiday_str = run.target_date.strftime("%Y-%m-%d")
    label = run.holiday_label or holiday_str
    window_start_str = run.pre_holiday_start.strftime("%Y-%m-%d %H:%M")
    window_end_str = (
        run.pre_holiday_end - pd.Timedelta(hours=1)
    ).strftime("%H:%M")
    ax.set_title(
        f"{run.unique_id} | {label} ({holiday_str})\n"
        f"last {run.previously_w_hours} h before holiday "
        f"[{window_start_str} → {window_end_str}]"
    )
    ax.set_xlabel(f"hours before {holiday_str} 00:00")
    ax.set_ylabel("demand")
    ax.set_xticks(hours)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_pre_holiday_batch_grid(
    batch_result: PreHolidayBatchResult,
    unique_id: Optional[str] = None,
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (5.5, 3.4),
    title: Optional[str] = None,
):
    """Grid of forecasts for one unique_id across every target date."""
    import matplotlib.pyplot as plt

    if unique_id is None:
        unique_id = batch_result.config.get("unique_ids", [None])[0]
    if unique_id is None:
        raise ValueError("No unique_id available to plot.")

    selected = [
        run for (date_key, uid), run in batch_result.runs.items()
        if uid == unique_id
    ]
    selected.sort(key=lambda r: r.target_date)
    if not selected:
        raise ValueError(f"No runs found for unique_id='{unique_id}'.")

    n = len(selected)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, run in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]
        plot_pre_holiday_run(run, ax=ax)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if title is None:
        title = (
            f"Pre-holiday window forecast | {unique_id} | "
            f"last {batch_result.config.get('previously_w_hours', '?')} h "
            f"before each holiday 00:00"
        )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


# =====================================================================
# Stage 2 — Holiday day (24 h) forecast from the curated source
# =====================================================================


@dataclass
class HolidayDayRun:
    """Single (unique_id, target_date) 24-h holiday-day forecast."""

    unique_id: str
    target_date: pd.Timestamp
    holiday_label: Optional[str]
    forecast: np.ndarray        # length 24
    forecast_int: np.ndarray    # rounded integers
    actual: Optional[np.ndarray]
    history_length: int
    fail: bool


@dataclass
class HolidayDayBatchResult:
    """Aggregated Stage-2 results for all (target_date, unique_id) pairs."""

    runs: Dict[Tuple[str, str], HolidayDayRun]
    results_df: pd.DataFrame
    output_path: Path
    source_path: Path
    config: Dict[str, object]


def _build_holiday_day_mask(
    df_hist: pd.DataFrame,
    unique_id: str,
    target_date: str | pd.Timestamp | None = None,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    match_target_cluster: bool = False,
) -> np.ndarray:
    """Binary mask aligned with ``df_hist``: 1 for hours that belong to historical holiday days.

    Uses the ``<uid>_holiday`` flag column directly — each hour already carries the
    flag of its parent day, so no date-level expansion is necessary. When
    ``match_target_cluster`` is true, only historical holiday dates whose
    selector ``analog_cluster`` matches the target date are kept.
    """
    holiday_col = f"{unique_id}_holiday"
    if holiday_col not in df_hist.columns:
        raise ValueError(
            f"Column '{holiday_col}' not found. "
            "Source CSV must include '<uid>_holiday' flag columns."
        )

    holiday_values = df_hist[holiday_col].fillna(0).astype(bool)
    if not match_target_cluster:
        return holiday_values.to_numpy(dtype=np.float64)

    if target_date is None:
        raise ValueError("target_date is required when match_target_cluster=True.")

    target_cluster = _get_target_cluster(target_date, selector_cluster_lookup)
    date_clusters = pd.to_datetime(df_hist["ds"]).dt.normalize().map(
        selector_cluster_lookup or {}
    )
    matched_mask = holiday_values & date_clusters.eq(target_cluster).fillna(False)
    if int(matched_mask.sum()) == 0:
        raise ValueError(
            f"No historical holiday-day hours share analog cluster {target_cluster!r} "
            f"before target_date={pd.Timestamp(target_date).normalize().date()}."
        )
    return matched_mask.to_numpy(dtype=np.float64)


def run_holiday_day(
    unique_id: str,
    target_date: str | pd.Timestamp,
    df_source: pd.DataFrame,
    season_length: int = 24,
    k: Optional[int] = 10,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    n_components: int = 3,
    typedist: str = "pearson",
    typereg: str = "PCR",
    holiday_label: Optional[str] = None,
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
) -> HolidayDayRun:
    """Forecast the full 24-h holiday day using ``AnalogSpecialDays``.

    Reads the curated Stage-1 source (``pre_holiday_demand_mx_*.csv``) so the
    pre-holiday hours immediately before the holiday are already filled in.

    Parameters
    ----------
    min_special_points:
        Minimum number of holiday hours required inside a 24-h X2 block.
        ``None`` defaults to ``season_length`` (require the full holiday day).
    """
    if unique_id not in df_source.columns:
        raise KeyError(f"Series '{unique_id}' not in source CSV columns.")

    target_ts = pd.Timestamp(target_date).normalize()
    day_start = target_ts
    day_end = day_start + pd.Timedelta(hours=season_length)

    history_mask = df_source["ds"] < day_start
    df_hist = df_source.loc[history_mask].copy()
    history = df_hist[unique_id].to_numpy(dtype=np.float64)
    if np.isnan(history).any():
        history = pd.Series(history).interpolate(limit_direction="both").to_numpy()

    if len(history) < 2 * season_length + 1:
        raise ValueError(
            f"Insufficient history for '{unique_id}' before {day_start} "
            f"(have {len(history)}, need >= {2 * season_length + 1})."
        )

    effective_min_points = season_length if min_special_points is None else min_special_points
    resolved_cluster_lookup = selector_cluster_lookup
    if match_target_cluster and resolved_cluster_lookup is None:
        resolved_cluster_lookup = _resolve_selector_cluster_lookup(
            selector_features_path=selector_features_path,
            cluster_column=cluster_column,
            match_target_cluster=match_target_cluster,
            unique_id=unique_id,
        )
    holiday_mask = _build_holiday_day_mask(
        df_hist,
        unique_id,
        target_date=target_ts,
        selector_cluster_lookup=resolved_cluster_lookup,
        match_target_cluster=match_target_cluster,
    )

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        min_special_points=effective_min_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )
    model.fit(y=history, special_days=holiday_mask)
    result = model.predict(h=season_length)
    forecast = np.asarray(result["mean"], dtype=np.float64)[:season_length]
    forecast_int = np.rint(forecast).astype(np.int64)

    window_mask = (df_source["ds"] >= day_start) & (df_source["ds"] < day_end)
    actual = None
    if int(window_mask.sum()) == season_length:
        candidate = df_source.loc[window_mask, unique_id].to_numpy(dtype=np.float64)
        if not np.isnan(candidate).any():
            actual = candidate

    return HolidayDayRun(
        unique_id=unique_id,
        target_date=target_ts,
        holiday_label=holiday_label,
        forecast=forecast,
        forecast_int=forecast_int,
        actual=actual,
        history_length=int(len(history)),
        fail=False,
    )


def run_holiday_day_batch(
    target_dates: Sequence,
    source_path: Path | str,
    unique_ids: Optional[Sequence[str]] = None,
    season_length: int = 24,
    k: Optional[int] = 10,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    n_components: int = 3,
    typedist: str = "pearson",
    typereg: str = "PCR",
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    output_dir: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    write_csv: bool = True,
) -> HolidayDayBatchResult:
    """Forecast 24-h holiday days for every (target_date, unique_id) pair.

    Reads the Stage-1 curated source (pre-holiday hours already filled).  Writes
    a copy named ``holiday_demand_mx_complete_<timestamp>.csv`` with the holiday-day
    rows overwritten by the integer-rounded forecasts.
    """
    source_path = Path(source_path)
    df_source = load_pre_holiday_source(source_path)

    if unique_ids is None:
        resolved_ids = list_unique_ids(df_source)
    else:
        resolved_ids = [str(u) for u in unique_ids]

    target_items = _normalize_target_items(target_dates)
    selector_cluster_lookup_by_id = {
        uid: _resolve_selector_cluster_lookup(
            selector_features_path=selector_features_path,
            cluster_column=cluster_column,
            match_target_cluster=match_target_cluster,
            unique_id=uid,
        )
        for uid in resolved_ids
    }
    df_out = df_source.copy()
    runs: Dict[Tuple[str, str], HolidayDayRun] = {}
    rows: List[dict] = []

    for target_ts, holiday_label in target_items:
        date_key = target_ts.strftime("%Y-%m-%d")
        day_end = target_ts + pd.Timedelta(hours=season_length)
        for uid in resolved_ids:
            try:
                run = run_holiday_day(
                    unique_id=uid,
                    target_date=target_ts,
                    df_source=df_source,
                    season_length=season_length,
                    k=k,
                    min_special_points=min_special_points,
                    min_event_gap=min_event_gap,
                    max_events=max_events,
                    n_components=n_components,
                    typedist=typedist,
                    typereg=typereg,
                    holiday_label=holiday_label,
                    selector_features_path=selector_features_path,
                    cluster_column=cluster_column,
                    match_target_cluster=match_target_cluster,
                    selector_cluster_lookup=selector_cluster_lookup_by_id[uid],
                )
            except Exception as exc:
                rows.append({
                    "target_date": target_ts,
                    "holiday_label": holiday_label,
                    "unique_id": uid,
                    "fail": True,
                    "error": str(exc),
                    "mae": np.nan,
                    "mape_pct": np.nan,
                })
                continue

            runs[(date_key, uid)] = run

            window_mask = (
                (df_out["ds"] >= target_ts) & (df_out["ds"] < day_end)
            )
            n_match = int(window_mask.sum())
            if n_match > 0:
                df_out.loc[window_mask, uid] = run.forecast_int[:n_match]

            mae = np.nan
            mape = np.nan
            if run.actual is not None:
                diff = run.forecast - run.actual
                mae = float(np.mean(np.abs(diff)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = np.where(run.actual == 0, np.nan, run.actual)
                    mape = float(np.nanmean(np.abs(diff / denom)) * 100.0)

            rows.append({
                "target_date": target_ts,
                "holiday_label": holiday_label,
                "unique_id": uid,
                "fail": False,
                "error": None,
                "mae": mae,
                "mape_pct": mape,
                "hours_written": n_match,
            })

    results_df = pd.DataFrame(rows)

    if output_path is not None:
        resolved_output = Path(output_path)
    else:
        base_dir = Path(output_dir) if output_dir is not None else source_path.parent
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        resolved_output = base_dir / f"holiday_demand_mx_complete_{timestamp}.csv"

    if write_csv:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(resolved_output, index=False)

    resolved_selector_path = None
    if match_target_cluster:
        resolved_selector_path = str(
            DEFAULT_SELECTOR_FEATURES_PATH if selector_features_path is None else selector_features_path
        )

    config = {
        "season_length": season_length,
        "k": k,
        "min_special_points": min_special_points,
        "min_event_gap": min_event_gap,
        "max_events": max_events,
        "n_components": n_components,
        "typedist": typedist,
        "typereg": typereg,
        "unique_ids": resolved_ids,
        "match_target_cluster": match_target_cluster,
        "cluster_column": cluster_column,
        "selector_features_path": resolved_selector_path,
    }

    return HolidayDayBatchResult(
        runs=runs,
        results_df=results_df,
        output_path=resolved_output,
        source_path=source_path,
        config=config,
    )


# =====================================================================
# Stage 2 — Plot helpers
# =====================================================================


def plot_holiday_day_run(run: HolidayDayRun, ax=None):
    """Plot the 24-h holiday-day forecast (and actual if present)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    hours = np.arange(1, len(run.forecast) + 1)
    ax.plot(hours, run.forecast, marker="o", label="forecast", color="tab:red", linewidth=1.5)
    if run.actual is not None:
        ax.plot(hours, run.actual, marker="x", label="actual", color="tab:blue", linewidth=1.5)

    holiday_str = run.target_date.strftime("%Y-%m-%d")
    label = run.holiday_label or holiday_str
    ax.set_title(f"{run.unique_id}\n{label} ({holiday_str})", fontsize=9)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("demand")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_holiday_day_batch_grid(
    batch_result: HolidayDayBatchResult,
    unique_id: Optional[str] = None,
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (5.5, 3.4),
    title: Optional[str] = None,
):
    """Grid of 24-h holiday-day forecasts for one ``unique_id`` across all target dates."""
    import matplotlib.pyplot as plt

    if unique_id is None:
        unique_id = batch_result.config.get("unique_ids", [None])[0]
    if unique_id is None:
        raise ValueError("No unique_id available to plot.")

    selected = [
        run for (date_key, uid), run in batch_result.runs.items()
        if uid == unique_id
    ]
    selected.sort(key=lambda r: r.target_date)
    if not selected:
        raise ValueError(f"No runs found for unique_id='{unique_id}'.")

    n = len(selected)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, run in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]
        plot_holiday_day_run(run, ax=ax)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if title is None:
        title = f"Holiday-day (24 h) forecast | {unique_id}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes
