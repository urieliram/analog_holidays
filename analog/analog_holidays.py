"""Thin orchestration helpers for AnalogSpecialDays holiday runs.

This module normalizes the audit input to a daily wide contract with one row per
``(unique_id, date)`` and hourly columns ``h_00`` ... ``h_23``. It supports:

- the legacy daily parquet cache used by the audit app;
- the hourly wide CSV exported by the audit app.

The forecasting logic stays unchanged once the source is normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analog_special_days import (
    AnalogSpecialDays,
    _normalize_scale_method,
    analog_special_days_core,
    count_special_day_candidates,
)
from analog_holidays.audit.data_loader import HOUR_COLS
from analog_holidays.shared.identify_holidays import load_selector_cluster_lookup


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PACKAGE_ROOT / "audit" / "data"
DEFAULT_LEVELS = [80, 95]
DEFAULT_SELECTOR_FEATURES_PATH = PACKAGE_ROOT / "holidays" / "holiday_selector_features.csv"
POST_HOLIDAY_RECOVERY_HOURS = 24
SELECTOR_CLUSTER_CRITERION_COLUMN = "analog_cluster_criterion"


@dataclass
class AnalogHolidayRun:
    unique_id: str
    target_date: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    forecast_start_offset_hours: int
    target_exists: bool
    target_has_complete_profile: bool
    typedist: str
    typereg: str
    scale_method: Optional[str]
    season_length: int
    k: Optional[int]
    n_components: int
    regressor_params: dict[str, object]
    levels: list[int]
    train_df: pd.DataFrame
    target_row: Optional[pd.Series]
    special_day_daily_mask: pd.Series
    hourly_series: np.ndarray
    special_day_hourly_mask: np.ndarray
    previous_day_profile: np.ndarray
    forecast_profile: np.ndarray
    interval_low: Dict[int, np.ndarray]
    interval_high: Dict[int, np.ndarray]
    actual_profile: Optional[np.ndarray]
    positions: list[int]
    neighbors2: np.ndarray
    selected_days_df: pd.DataFrame
    fail: bool
    t_sel: float
    t_reg: float
    special_labels: tuple[str, ...]
    include_declared_holidays: bool
    include_outliers: bool
    min_special_points: Optional[int]
    min_event_gap: Optional[int]
    max_events: Optional[int]
    recent_weekend_analogs: int
    recent_weekend_like: Optional[str]
    recent_weekend_dates: list[pd.Timestamp]
    label_column: str
    post_holiday_actual_profile: Optional[np.ndarray] = None


@dataclass
class AnalogHolidayOptunaResult:
    best_config: dict[str, object]
    summary_df: pd.DataFrame
    fold_metrics_df: pd.DataFrame
    eligible_dates: list[pd.Timestamp]
    study: object


@dataclass
class AnalogHolidayBatchResult:
    target_items: list[tuple[str, Optional[str]]]
    runs: Dict[str, AnalogHolidayRun]
    results_df: pd.DataFrame
    metric_summary_df: pd.DataFrame


def _coerce_regressor_params(
    regressor_params: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    return dict(regressor_params or {})


def _format_regressor_params(regressor_params: Optional[dict[str, object]] = None) -> str:
    params = _coerce_regressor_params(regressor_params)
    return json.dumps(params, sort_keys=True) if params else "{}"


def _coerce_2d_profiles(profiles: object) -> np.ndarray:
    arr = np.asarray(profiles, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def fit_hourly_bias_factor_model(
    neighbor_profiles: object,
    window_hours: Optional[int] = None,
    max_analogs: int = 4,
    tail_hours: Optional[int] = None,
) -> dict[str, object]:
    """Fit an hourly bias-adjustment factor model from up to the available analogs.

    The factor is learned over the full window passed by the caller. Historical
    neighbors remain the only training source, so the fit does not use target
    actuals and therefore does not introduce leakage.

    ``tail_hours`` is kept as a compatibility alias for older notebook cells.
    """
    resolved_window_hours = window_hours
    if resolved_window_hours is None:
        resolved_window_hours = tail_hours
    elif tail_hours is not None and int(window_hours) != int(tail_hours):
        raise ValueError("window_hours and tail_hours must match when both are provided.")

    if resolved_window_hours is None:
        raise ValueError("window_hours is required.")

    resolved_window_hours = int(resolved_window_hours)
    if resolved_window_hours <= 0:
        raise ValueError("window_hours must be a positive integer.")

    neighbor_matrix = _coerce_2d_profiles(neighbor_profiles)
    requested_analogs = max(0, int(max_analogs))
    available_neighbor_profiles = int(neighbor_matrix.shape[0])
    analogs_to_use = min(requested_analogs, available_neighbor_profiles)
    valid_window_profiles: list[np.ndarray] = []

    for profile in neighbor_matrix[:analogs_to_use]:
        window_profile = np.asarray(profile[-resolved_window_hours:], dtype=np.float64)
        if (
            window_profile.shape[0] != resolved_window_hours
            or not np.isfinite(window_profile).all()
        ):
            continue
        window_mean = float(np.mean(window_profile))
        if not np.isfinite(window_mean) or abs(window_mean) <= 1e-9:
            continue
        valid_window_profiles.append(window_profile)

    if not valid_window_profiles:
        return {
            "method": "none",
            "train_samples": 0,
            "requested_analogs": requested_analogs,
            "available_neighbor_profiles": available_neighbor_profiles,
            "selected_analogs": 0,
            "window_hours": resolved_window_hours,
            "hourly_factors": np.zeros(resolved_window_hours, dtype=np.float64),
            "factor_mean": 0.0,
            "factor_mean_abs": 0.0,
            "window_mean_train": np.nan,
            "daily_mean_train": np.nan,
            "head_tail_corr_train": np.nan,
            "intercept": 0.0,
            "slope": 0.0,
        }

    window_matrix = np.vstack(valid_window_profiles)
    window_means = window_matrix.mean(axis=1, keepdims=True)
    hourly_factors = window_matrix / window_means - 1.0
    mean_hourly_factors = hourly_factors.mean(axis=0)
    mean_hourly_factors = mean_hourly_factors - mean_hourly_factors.mean()

    return {
        "method": "hourly_window_mean_factor_top_available_analogs",
        "train_samples": int(window_matrix.shape[0]),
        "requested_analogs": requested_analogs,
        "available_neighbor_profiles": available_neighbor_profiles,
        "selected_analogs": int(window_matrix.shape[0]),
        "window_hours": resolved_window_hours,
        "hourly_factors": mean_hourly_factors.astype(np.float64),
        "factor_mean": float(mean_hourly_factors.mean()),
        "factor_mean_abs": float(np.mean(np.abs(mean_hourly_factors))),
        "window_mean_train": float(window_means.mean()),
        "daily_mean_train": float(window_means.mean()),
        "head_tail_corr_train": np.nan,
        "intercept": float(window_means.mean()),
        "slope": 0.0,
    }


def _suggest_optuna_regressor_params(trial, typereg: str) -> dict[str, object]:
    if typereg == "RF":
        return {
            # Compact search for moderate training cost and low overfitting risk.
            "n_estimators": int(trial.suggest_categorical("rf_n_estimators", [100, 200, 300])),
            "max_depth": int(trial.suggest_categorical("rf_max_depth", [4, 8, 12])),
            "min_samples_leaf": int(trial.suggest_categorical("rf_min_samples_leaf", [1, 2, 4])),
        }

    if typereg == "LGBM":
        return {
            # Keep only the main LightGBM controls and use a small discrete grid.
            "n_estimators": int(trial.suggest_categorical("lgbm_n_estimators", [100, 200, 300])),
            "learning_rate": float(trial.suggest_categorical("lgbm_learning_rate", [0.03, 0.05, 0.1])),
            "num_leaves": int(trial.suggest_categorical("lgbm_num_leaves", [15, 31, 63])),
            "min_child_samples": int(trial.suggest_categorical("lgbm_min_child_samples", [10, 20, 30])),
        }

    return {}


def _resolve_scale_method_choices(
    scale_method_choices: Optional[Sequence[Optional[str] | str]] = None,
) -> tuple[Optional[str], ...]:
    if scale_method_choices is None:
        return ()

    resolved: list[Optional[str]] = []
    seen: set[Optional[str]] = set()
    for choice in scale_method_choices:
        normalized = _normalize_scale_method(choice)
        if normalized in seen:
            continue
        resolved.append(normalized)
        seen.add(normalized)

    if not resolved:
        raise ValueError(
            "scale_method_choices must contain at least one valid option."
        )
    return tuple(resolved)


def load_audit_source(source_path: Path | str = DEFAULT_SOURCE_PATH) -> pd.DataFrame:
    """Load the audit source and normalize it to the daily wide contract."""
    path = _resolve_audit_source_path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Audit source not found: {path}")

    if path.suffix.lower() == ".csv":
        return _load_hourly_wide_csv(path)
    if path.suffix.lower() == ".parquet":
        return _load_daily_parquet(path)

    raise ValueError(
        f"Unsupported audit source format '{path.suffix}'. "
        "Expected .csv or .parquet."
    )


def _resolve_audit_source_path(source_path: Path | str) -> Path:
    """Resolve a concrete audit source file from a directory or stale CSV path."""
    path = Path(source_path)
    if path.exists():
        if path.is_dir():
            return _find_latest_audit_csv(path)
        if path.name == "holiday_demand_mx_analog_cluster.csv":
            raise ValueError(
                "holiday_demand_mx_analog_cluster.csv is no longer a valid historical "
                "demand source. Use holiday_demand_mx.csv for demand history and "
                "holiday_selector_features.csv for analog_cluster lookups."
            )
        return path

    if path.suffix.lower() == ".csv" and path.name.startswith("holiday_audit_hourly_wide_"):
        parent = path.parent
        if parent.exists():
            return _find_latest_audit_csv(parent)

    raise FileNotFoundError(f"Audit source not found: {path}")


def _find_latest_audit_csv(directory: Path) -> Path:
    """Return the latest hourly wide holiday audit CSV inside a directory."""
    candidates = sorted(directory.glob("holiday_audit_hourly_wide_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No holiday audit CSV files were found in: {directory}"
        )
    return candidates[-1]


def prepare_audit_working_copy(
    source_path: Path | str,
    working_dir: Optional[Path | str] = None,
    prefix: str = "holiday_audit_hourly_wide",
    reuse_today: bool = True,
) -> Path:
    """Copy the canonical audit CSV to a timestamped working copy and return it.

    The canonical file at ``source_path`` (a CSV or a directory containing the
    latest ``holiday_audit_hourly_wide_*.csv``) is never modified. A local copy
    is written to ``working_dir`` with name ``{prefix}_<YYYYMMDD_HHMMSS>.csv``.

    Parameters
    ----------
    source_path
        Canonical CSV file or directory holding ``holiday_audit_hourly_wide_*.csv``.
    working_dir
        Destination directory for the working copy. Defaults to
        ``<package>/holidays/working/``. Created if missing.
    prefix
        File-name prefix for the working copy.
    reuse_today
        If True and a working copy created today (matching ``{prefix}_<YYYYMMDD>_*.csv``)
        already exists in ``working_dir`` and has the same size as the canonical
        source, reuse it instead of recopying.

    Returns
    -------
    Path
        Absolute path to the working copy.
    """
    canonical = _resolve_audit_source_path(source_path)

    if working_dir is None:
        working_dir = PACKAGE_ROOT / "holidays" / "working"
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    if reuse_today:
        same_day = sorted(
            working_dir.glob(f"{prefix}_{today}_*.csv"), reverse=True
        )
        canonical_size = canonical.stat().st_size
        for existing in same_day:
            if existing.stat().st_size == canonical_size:
                return existing.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = working_dir / f"{prefix}_{timestamp}.csv"
    shutil.copy2(canonical, dest)
    return dest.resolve()


def get_available_unique_ids(source_path: Path | str = DEFAULT_SOURCE_PATH) -> list[str]:
    df = load_audit_source(source_path)
    return sorted(df["unique_id"].dropna().unique().tolist())


def list_candidate_special_dates(
    unique_id: str,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    label_column: str = "label",
    n: int = 15,
) -> pd.DataFrame:
    """Return recent labeled dates to help choose a target date."""
    df = load_audit_source(source_path)
    df_region = _get_region_df(df, unique_id)
    complete_mask = _complete_day_mask(df_region)
    mask = build_special_day_daily_mask(
        df_region,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
    )

    cols = [
        "date",
        "label",
        "holiday_name",
        "holiday_type",
        "is_declared_holiday",
        "is_outlier",
        "outlier_score",
    ]
    out = df_region.loc[mask & complete_mask, cols].tail(n).reset_index(drop=True)
    return out


def build_special_day_daily_mask(
    df_region: pd.DataFrame,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    label_column: str = "label",
) -> pd.Series:
    """Build the daily mask used to mark special events."""
    mask = pd.Series(False, index=df_region.index)

    if special_labels:
        mask |= df_region[label_column].fillna("normal_day").isin(tuple(special_labels))
    if include_declared_holidays:
        mask |= df_region["is_declared_holiday"].fillna(False).astype(bool)
    if include_outliers:
        mask |= df_region["is_outlier"].fillna(False).astype(bool)

    return mask.astype(bool)


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


def _resolve_selector_cluster_filter_label(
    selector_features_path: Path | str | None,
    match_target_cluster: bool,
    unique_id: str | None = None,
    criterion_column: str = SELECTOR_CLUSTER_CRITERION_COLUMN,
) -> object:
    if not match_target_cluster:
        return False

    resolved_path = (
        DEFAULT_SELECTOR_FEATURES_PATH
        if selector_features_path is None else Path(selector_features_path)
    )
    selector_df = pd.read_csv(resolved_path, parse_dates=["date"])
    if "unique_id" in selector_df.columns:
        available_unique_ids = pd.Series(selector_df["unique_id"]).dropna().astype(str).unique().tolist()
        if unique_id is not None:
            selector_df = selector_df[selector_df["unique_id"].astype(str) == str(unique_id)].copy()
            if selector_df.empty:
                raise ValueError(
                    f"Selector CSV {resolved_path} has no rows for unique_id={unique_id!r}."
                )
        elif len(available_unique_ids) > 1:
            preview = ", ".join(sorted(available_unique_ids)[:5])
            raise ValueError(
                "Selector CSV contains multiple unique_id values. "
                f"Pass unique_id to select one series: {preview}"
            )

    if criterion_column not in selector_df.columns:
        return True

    criterion_values = (
        pd.Series(selector_df[criterion_column])
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    criterion_values = [value for value in criterion_values.unique().tolist() if value]
    if not criterion_values:
        return True
    if len(criterion_values) > 1:
        raise ValueError(
            "Selector CSV contains multiple analog cluster criteria for the same series: "
            f"{criterion_values}"
        )
    return criterion_values[0]


def _normalize_recent_weekend_analogs(recent_weekend_analogs: int) -> int:
    resolved = int(recent_weekend_analogs)
    if resolved < 0:
        raise ValueError("recent_weekend_analogs must be >= 0.")
    return resolved


def _normalize_weekend_like_value(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()
    if normalized in {"saturday", "saturday-like"}:
        return "Saturday"
    if normalized in {"sunday", "sunday-like"}:
        return "Sunday"
    return None


def _resolve_selector_weekend_like_lookup(
    selector_features_path: Path | str | None,
    recent_weekend_analogs: int,
    unique_id: str | None = None,
) -> Optional[dict[pd.Timestamp, str]]:
    if _normalize_recent_weekend_analogs(recent_weekend_analogs) == 0:
        return None

    resolved_path = (
        DEFAULT_SELECTOR_FEATURES_PATH
        if selector_features_path is None else Path(selector_features_path)
    )
    selector_df = pd.read_csv(resolved_path, parse_dates=["date"])
    if "unique_id" in selector_df.columns:
        available_unique_ids = pd.Series(selector_df["unique_id"]).dropna().astype(str).unique().tolist()
        if unique_id is not None:
            selector_df = selector_df[selector_df["unique_id"].astype(str) == str(unique_id)].copy()
            if selector_df.empty:
                raise ValueError(
                    f"Selector CSV {resolved_path} has no rows for unique_id={unique_id!r}."
                )
        elif len(available_unique_ids) > 1:
            preview = ", ".join(sorted(available_unique_ids)[:5])
            raise ValueError(
                "Selector CSV contains multiple unique_id values. "
                f"Pass unique_id to select one series: {preview}"
            )
    selector_df["date"] = pd.to_datetime(selector_df["date"]).dt.normalize()

    if "best_matching_weekday" not in selector_df.columns:
        selector_df["best_matching_weekday"] = pd.NA
    if "daily_profile_archetype" not in selector_df.columns:
        selector_df["daily_profile_archetype"] = pd.NA

    selector_df["weekend_like"] = [
        _normalize_weekend_like_value(best_matching_weekday)
        or _normalize_weekend_like_value(daily_profile_archetype)
        for best_matching_weekday, daily_profile_archetype in zip(
            selector_df["best_matching_weekday"],
            selector_df["daily_profile_archetype"],
        )
    ]
    weekend_df = (
        selector_df.loc[selector_df["weekend_like"].notna(), ["date", "weekend_like"]]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
    )
    return dict(zip(weekend_df["date"], weekend_df["weekend_like"]))


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


def _restrict_daily_mask_to_target_cluster(
    mask: pd.Series,
    df_region: pd.DataFrame,
    target_date: str | pd.Timestamp,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]],
    raise_if_empty: bool = True,
) -> pd.Series:
    target_cluster = _get_target_cluster(target_date, selector_cluster_lookup)
    date_clusters = pd.to_datetime(df_region["date"]).dt.normalize().map(
        selector_cluster_lookup or {}
    )
    restricted_mask = mask.astype(bool) & date_clusters.eq(target_cluster).fillna(False)
    if int(restricted_mask.sum()) == 0 and raise_if_empty:
        raise ValueError(
            f"No historical special days share analog cluster {target_cluster!r} "
            f"before target_date={pd.Timestamp(target_date).normalize().date()}."
        )
    return restricted_mask.astype(bool)


def _build_recent_weekend_analog_mask(
    train_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    selector_weekend_lookup: Optional[dict[pd.Timestamp, str]],
    recent_weekend_analogs: int,
) -> tuple[pd.Series, Optional[str], list[pd.Timestamp]]:
    resolved_recent_weekends = _normalize_recent_weekend_analogs(recent_weekend_analogs)
    empty_mask = pd.Series(False, index=train_df.index, dtype=bool)
    if resolved_recent_weekends == 0 or selector_weekend_lookup is None:
        return empty_mask, None, []

    target_ts = pd.Timestamp(target_date).normalize()
    weekend_like = selector_weekend_lookup.get(target_ts)
    if weekend_like not in {"Saturday", "Sunday"}:
        return empty_mask, weekend_like, []

    weekday_number = 5 if weekend_like == "Saturday" else 6
    train_dates = pd.to_datetime(train_df["date"]).dt.normalize()
    selected_dates = (
        train_dates.loc[train_dates.dt.dayofweek == weekday_number]
        .drop_duplicates()
        .sort_values()
        .tail(resolved_recent_weekends)
        .tolist()
    )
    if not selected_dates:
        return empty_mask, weekend_like, []

    recent_weekend_mask = train_dates.isin(selected_dates)
    resolved_dates = [pd.Timestamp(date_value).normalize() for date_value in selected_dates]
    return recent_weekend_mask.astype(bool), weekend_like, resolved_dates


def _build_analog_candidate_daily_mask(
    train_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    special_labels: Sequence[str],
    include_declared_holidays: bool,
    include_outliers: bool,
    label_column: str,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    match_target_cluster: bool = False,
    selector_weekend_lookup: Optional[dict[pd.Timestamp, str]] = None,
    recent_weekend_analogs: int = 0,
) -> tuple[pd.Series, Optional[str], list[pd.Timestamp]]:
    resolved_recent_weekends = _normalize_recent_weekend_analogs(recent_weekend_analogs)
    candidate_mask = build_special_day_daily_mask(
        train_df,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
    )
    if match_target_cluster:
        candidate_mask = _restrict_daily_mask_to_target_cluster(
            candidate_mask,
            train_df,
            target_date,
            selector_cluster_lookup,
            raise_if_empty=resolved_recent_weekends == 0,
        )

    recent_weekend_mask, recent_weekend_like, recent_weekend_dates = _build_recent_weekend_analog_mask(
        train_df,
        target_date,
        selector_weekend_lookup,
        resolved_recent_weekends,
    )
    candidate_mask = candidate_mask.astype(bool) | recent_weekend_mask.astype(bool)
    if int(candidate_mask.sum()) == 0:
        raise ValueError(
            f"No prior candidate days exist before {pd.Timestamp(target_date).normalize().date()} "
            "to build analogs."
        )
    return candidate_mask.astype(bool), recent_weekend_like, recent_weekend_dates


def _normalize_forecast_start_offset_hours(
    forecast_start_offset_hours: int,
) -> int:
    offset_hours = int(forecast_start_offset_hours)
    if offset_hours < 0:
        raise ValueError("forecast_start_offset_hours must be >= 0.")
    return offset_hours


def _count_realizable_analog_positions(
    history_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    season_length: int,
    special_labels: Sequence[str],
    include_declared_holidays: bool,
    include_outliers: bool,
    min_special_points: Optional[int],
    min_event_gap: Optional[int],
    max_events: Optional[int],
    forecast_start_offset_hours: int,
    label_column: str,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    match_target_cluster: bool = False,
    selector_weekend_lookup: Optional[dict[pd.Timestamp, str]] = None,
    recent_weekend_analogs: int = 0,
) -> int:
    """Count analog windows that remain after all upstream filters except k-ranking."""
    target_ts = pd.Timestamp(target_date).normalize()
    forecast_start_offset_hours = _normalize_forecast_start_offset_hours(
        forecast_start_offset_hours
    )
    train_df = history_df.loc[history_df["date"] < target_ts].copy()

    if train_df.empty:
        raise ValueError(
            f"No prior history exists before {target_ts.date()} to build analogs."
        )

    hourly_series = _truncate_hourly_history(
        _flatten_daily_profiles(train_df),
        forecast_start_offset_hours,
    )
    if len(hourly_series) < 2 * season_length + 1:
        raise ValueError(
            f"Insufficient history before {target_ts.date()} for season_length={season_length}."
        )

    special_mask, _, _ = _build_analog_candidate_daily_mask(
        train_df,
        target_ts,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
        selector_cluster_lookup=selector_cluster_lookup,
        match_target_cluster=match_target_cluster,
        selector_weekend_lookup=selector_weekend_lookup,
        recent_weekend_analogs=recent_weekend_analogs,
    )

    special_hourly_mask = np.repeat(
        special_mask.astype(float).to_numpy(),
        len(HOUR_COLS),
    )
    special_hourly_mask = _truncate_hourly_history(
        special_hourly_mask,
        forecast_start_offset_hours,
    )
    analog_count = count_special_day_candidates(
        special_days=special_hourly_mask,
        vsele=season_length,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )
    if analog_count < 1:
        raise ValueError(
            f"No realizable analog windows remain before {target_ts.date()} after filtering."
        )

    return int(analog_count)


def _truncate_hourly_history(
    hourly_values: np.ndarray,
    forecast_start_offset_hours: int,
) -> np.ndarray:
    offset_hours = _normalize_forecast_start_offset_hours(forecast_start_offset_hours)
    if offset_hours == 0:
        return hourly_values
    if offset_hours >= len(hourly_values):
        return np.asarray([], dtype=np.float64)
    return hourly_values[:-offset_hours]


def _extract_hour_window(
    df_region: pd.DataFrame,
    window_start: pd.Timestamp,
    length_hours: int,
) -> Optional[np.ndarray]:
    if length_hours <= 0:
        raise ValueError("length_hours must be > 0.")

    start_ts = pd.Timestamp(window_start)
    start_day = start_ts.normalize()
    start_hour = int((start_ts - start_day) / pd.Timedelta(hours=1))
    if start_hour < 0 or start_hour >= len(HOUR_COLS):
        raise ValueError(
            f"window_start must be aligned to a valid hourly slot, got {start_ts!s}."
        )

    days_needed = math.ceil((start_hour + int(length_hours)) / len(HOUR_COLS))
    expected_dates = pd.date_range(start_day, periods=days_needed, freq="D")
    daily_profiles = (
        df_region[["date", *HOUR_COLS]]
        .drop_duplicates(subset="date")
        .set_index("date")
        .sort_index()
        .reindex(expected_dates)
    )
    if daily_profiles.isna().any().any():
        return None

    flattened = daily_profiles.to_numpy(dtype=np.float64).reshape(-1)
    window = flattened[start_hour:start_hour + int(length_hours)]
    if len(window) != int(length_hours) or not np.isfinite(window).all():
        return None
    return window


def run_analog_holidays(
    unique_id: str,
    target_date: str | pd.Timestamp,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    season_length: int = 24,
    forecast_start_offset_hours: int = 0,
    k: Optional[int] = None,
    typedist: str = "pearson",
    typereg: str = "PCR",
    scale_method: Optional[str] = None,
    n_components: int = 3,
    regressor_params: Optional[dict[str, object]] = None,
    levels: Optional[Sequence[int]] = None,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
    expected_target_label: Optional[str] = "holiday",
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    selector_weekend_lookup: Optional[dict[pd.Timestamp, str]] = None,
    recent_weekend_analogs: int = 0,
) -> AnalogHolidayRun:
    """Run an offset-aware AnalogSpecialDays forecast for a holiday date."""
    target_ts = pd.Timestamp(target_date).normalize()
    regressor_params = _coerce_regressor_params(regressor_params)
    recent_weekend_analogs = _normalize_recent_weekend_analogs(recent_weekend_analogs)
    forecast_start_offset_hours = _normalize_forecast_start_offset_hours(
        forecast_start_offset_hours
    )
    forecast_start = target_ts - pd.Timedelta(hours=forecast_start_offset_hours)
    forecast_end = forecast_start + pd.Timedelta(hours=season_length)
    levels = list(DEFAULT_LEVELS if levels is None else levels)

    df = load_audit_source(source_path)
    df_region = _get_region_df(df, unique_id)
    complete_mask = _complete_day_mask(df_region)
    train_df = df_region[(df_region["date"] < target_ts) & complete_mask].copy()
    target_rows = df_region[df_region["date"] == target_ts].copy()
    target_row = target_rows.iloc[0] if not target_rows.empty else None

    if expected_target_label is not None:
        if target_row is None:
            warnings.warn(
                f"{unique_id} has no row for target_date={target_ts.date()} in the audit source.",
                stacklevel=2,
            )
        else:
            target_label = target_row.get(label_column, "normal_day")
            target_label = "normal_day" if pd.isna(target_label) else str(target_label)
            if target_label != expected_target_label:
                warnings.warn(
                    f"target_date={target_ts.date()} for {unique_id} is labeled "
                    f"'{target_label}', not '{expected_target_label}'.",
                    stacklevel=2,
                )

    if train_df.empty:
        raise ValueError(f"No history found for {unique_id} before {target_ts.date()}")

    resolved_cluster_lookup = selector_cluster_lookup
    if match_target_cluster and resolved_cluster_lookup is None:
        resolved_cluster_lookup = _resolve_selector_cluster_lookup(
            selector_features_path=selector_features_path,
            cluster_column=cluster_column,
            match_target_cluster=match_target_cluster,
            unique_id=unique_id,
        )
    resolved_weekend_lookup = selector_weekend_lookup
    if recent_weekend_analogs > 0 and resolved_weekend_lookup is None:
        resolved_weekend_lookup = _resolve_selector_weekend_like_lookup(
            selector_features_path=selector_features_path,
            recent_weekend_analogs=recent_weekend_analogs,
            unique_id=unique_id,
        )

    hourly_series = _truncate_hourly_history(
        _flatten_daily_profiles(train_df),
        forecast_start_offset_hours,
    )
    if len(hourly_series) < 2 * season_length + 1:
        raise ValueError(
            "History is too short for AnalogSpecialDays. "
            f"At least {2 * season_length + 1} hourly points are required before "
            f"forecast_start={forecast_start}."
        )

    special_day_daily_mask, recent_weekend_like, recent_weekend_dates = _build_analog_candidate_daily_mask(
        train_df,
        target_ts,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
        selector_cluster_lookup=resolved_cluster_lookup,
        match_target_cluster=match_target_cluster,
        selector_weekend_lookup=resolved_weekend_lookup,
        recent_weekend_analogs=recent_weekend_analogs,
    )

    special_day_hourly_mask = np.repeat(
        special_day_daily_mask.astype(float).to_numpy(),
        len(HOUR_COLS),
    )
    special_day_hourly_mask = _truncate_hourly_history(
        special_day_hourly_mask,
        forecast_start_offset_hours,
    )

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        regressor_params=regressor_params,
        scale_method=scale_method,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )
    model.fit(y=hourly_series, special_days=special_day_hourly_mask)
    result = model.predict(h=season_length, level=levels)

    forecast_profile = result["mean"]
    interval_low = {lv: result[f"lo-{lv}"] for lv in levels if f"lo-{lv}" in result}
    interval_high = {lv: result[f"hi-{lv}"] for lv in levels if f"hi-{lv}" in result}

    _, t_sel, t_reg, fail, positions, neighbors2 = analog_special_days_core(
        serie=hourly_series,
        special_days=special_day_hourly_mask,
        vsele=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        regressor_params=regressor_params,
        scale_method=scale_method,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )

    selected_days_df = build_selected_days_table(
        train_df=train_df,
        positions=positions,
        neighbors2=neighbors2,
        season_length=season_length,
        forecast_start_offset_hours=forecast_start_offset_hours,
    )

    previous_day_profile = hourly_series[-season_length:].copy()
    actual_profile = _extract_hour_window(
        df_region,
        window_start=forecast_start,
        length_hours=season_length,
    )
    post_holiday_actual_profile = _extract_hour_window(
        df_region,
        window_start=target_ts + pd.Timedelta(hours=24),
        length_hours=POST_HOLIDAY_RECOVERY_HOURS,
    )
    target_has_complete_profile = actual_profile is not None

    return AnalogHolidayRun(
        unique_id=unique_id,
        target_date=target_ts,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        forecast_start_offset_hours=forecast_start_offset_hours,
        target_exists=target_row is not None,
        target_has_complete_profile=target_has_complete_profile,
        typedist=typedist,
        typereg=typereg,
        scale_method=scale_method,
        season_length=season_length,
        k=k,
        n_components=n_components,
        regressor_params=regressor_params,
        levels=levels,
        train_df=train_df,
        target_row=target_row,
        special_day_daily_mask=special_day_daily_mask,
        hourly_series=hourly_series,
        special_day_hourly_mask=special_day_hourly_mask,
        previous_day_profile=previous_day_profile,
        forecast_profile=forecast_profile,
        interval_low=interval_low,
        interval_high=interval_high,
        actual_profile=actual_profile,
        positions=positions,
        neighbors2=neighbors2,
        selected_days_df=selected_days_df,
        fail=fail,
        t_sel=t_sel,
        t_reg=t_reg,
        special_labels=tuple(special_labels),
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
        recent_weekend_analogs=recent_weekend_analogs,
        recent_weekend_like=recent_weekend_like,
        recent_weekend_dates=recent_weekend_dates,
        label_column=label_column,
        post_holiday_actual_profile=post_holiday_actual_profile,
    )


def tune_analog_holidays_optuna(
    unique_id: str,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    train_end: str | pd.Timestamp = "2024-01-01",
    season_length: int = 24,
    forecast_start_offset_hours: int = 0,
    initial_k: Optional[int] = None,
    initial_typedist: str = "pearson",
    initial_typereg: str = "PCR",
    scale_method: Optional[str] = None,
    scale_method_choices: Optional[Sequence[Optional[str] | str]] = None,
    initial_n_components: int = 3,
    initial_regressor_params: Optional[dict[str, object]] = None,
    optuna_min_k: int = 3,
    n_trials: int = 25,
    timeout_sec: Optional[int] = 900,
    max_eval_dates: Optional[int] = 12,
    random_seed: int = 42,
    special_labels: Sequence[str] = ("holiday",),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
    typedist_choices: Optional[Sequence[str]] = None,
    typereg_choices: Optional[Sequence[str]] = None,
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = False,
    recent_weekend_analogs: int = 0,
) -> AnalogHolidayOptunaResult:
    """Tune AnalogSpecialDays hyperparameters on historical special dates."""
    import optuna

    train_end_ts = pd.Timestamp(train_end)
    scale_method = _normalize_scale_method(scale_method)
    recent_weekend_analogs = _normalize_recent_weekend_analogs(recent_weekend_analogs)
    resolved_scale_method_choices = _resolve_scale_method_choices(scale_method_choices)
    initial_regressor_params = _coerce_regressor_params(initial_regressor_params)
    df = load_audit_source(source_path)
    df_region = _get_region_df(df, unique_id)
    complete_mask = _complete_day_mask(df_region)
    selector_cluster_lookup = _resolve_selector_cluster_lookup(
        selector_features_path=selector_features_path,
        cluster_column=cluster_column,
        match_target_cluster=match_target_cluster,
        unique_id=unique_id,
    )
    selector_weekend_lookup = _resolve_selector_weekend_like_lookup(
        selector_features_path=selector_features_path,
        recent_weekend_analogs=recent_weekend_analogs,
        unique_id=unique_id,
    )
    history_df = df_region.loc[
        (df_region["date"] < train_end_ts) & complete_mask
    ].copy()

    if history_df.empty:
        raise ValueError(
            f"No complete history is available for {unique_id} before {train_end_ts.date()}."
        )

    # ── Build search spaces ───────────────────────────────────────────────────
    # typedist_choices: keep the default distance search compact for fast
    # rolling tuning. DTW remains available only via an explicit override.
    if typedist_choices is None:
        resolved_typedist_choices = ["pearson", "euclidian"]
    else:
        resolved_typedist_choices = [str(choice) for choice in typedist_choices]

    # typereg_choices: compact default search focused on the main moderate-cost
    # regressors. Broader searches, including LGBM, remain available only via
    # explicit overrides.
    lightgbm_available = importlib.util.find_spec("lightgbm") is not None
    resolved_typereg_choices = [
        "PCR",
        "PLS",
    ] if typereg_choices is None else [str(choice) for choice in typereg_choices]
    if "LGBM" in resolved_typereg_choices and not lightgbm_available:
        raise ImportError("Install lightgbm to include typereg='LGBM' in Optuna tuning.")

    special_mask_history = build_special_day_daily_mask(
        history_df,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
    )
    candidate_dates = (
        history_df.loc[special_mask_history, "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    optuna_min_k = int(optuna_min_k)
    if optuna_min_k < 1:
        raise ValueError(f"optuna_min_k must be >= 1, got {optuna_min_k}.")

    available_special_days = int(special_mask_history.sum())
    if available_special_days < 1:
        raise ValueError(
            "At least 1 labeled special day is required before the cutoff "
            "to tune hyperparameters."
        )

    baseline_k = optuna_min_k
    eligible_pairs: list[tuple[pd.Timestamp, int]] = []
    for candidate_date in candidate_dates:
        try:
            _evaluate_analog_holiday_fold(
                history_df=history_df,
                target_date=candidate_date,
                season_length=season_length,
                k=baseline_k,
                typedist=initial_typedist,
                typereg=initial_typereg,
                scale_method=scale_method,
                n_components=initial_n_components,
                regressor_params=initial_regressor_params,
                special_labels=special_labels,
                include_declared_holidays=include_declared_holidays,
                include_outliers=include_outliers,
                min_special_points=min_special_points,
                min_event_gap=min_event_gap,
                max_events=max_events,
                forecast_start_offset_hours=forecast_start_offset_hours,
                label_column=label_column,
                selector_cluster_lookup=selector_cluster_lookup,
                match_target_cluster=match_target_cluster,
                selector_weekend_lookup=selector_weekend_lookup,
                recent_weekend_analogs=recent_weekend_analogs,
            )
            realizable_analogs = _count_realizable_analog_positions(
                history_df=history_df,
                target_date=candidate_date,
                season_length=season_length,
                special_labels=special_labels,
                include_declared_holidays=include_declared_holidays,
                include_outliers=include_outliers,
                min_special_points=min_special_points,
                min_event_gap=min_event_gap,
                max_events=max_events,
                forecast_start_offset_hours=forecast_start_offset_hours,
                label_column=label_column,
                selector_cluster_lookup=selector_cluster_lookup,
                match_target_cluster=match_target_cluster,
                selector_weekend_lookup=selector_weekend_lookup,
                recent_weekend_analogs=recent_weekend_analogs,
            )
            if realizable_analogs < optuna_min_k:
                continue
            eligible_pairs.append((pd.Timestamp(candidate_date), realizable_analogs))
        except ValueError:
            continue

    if max_eval_dates is not None:
        eligible_pairs = eligible_pairs[-int(max_eval_dates):]

    eligible_dates = [target_date for target_date, _ in eligible_pairs]
    eligible_k_caps = [analog_count for _, analog_count in eligible_pairs]

    if len(eligible_dates) < 2:
        raise ValueError(
            f"At least 2 eligible special events with >= {optuna_min_k} realizable "
            "analogs are required before the cutoff to tune hyperparameters."
        )

    final_target_analog_cap = _count_realizable_analog_positions(
        history_df=history_df,
        target_date=train_end_ts,
        season_length=season_length,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
        forecast_start_offset_hours=forecast_start_offset_hours,
        label_column=label_column,
        selector_cluster_lookup=selector_cluster_lookup,
        match_target_cluster=match_target_cluster,
        selector_weekend_lookup=selector_weekend_lookup,
        recent_weekend_analogs=recent_weekend_analogs,
    )
    if final_target_analog_cap < optuna_min_k:
        raise ValueError(
            f"At least {optuna_min_k} realizable analogs are required for "
            f"target_date={train_end_ts.date()} after applying the special-day filters."
        )

    # Search bounds for k: lower bound comes from optuna_min_k, while the upper bound
    # is the smallest realizable analog pool across retained evaluation folds
    # and the final target run, capped at 24.
    optuna_max_k = min(final_target_analog_cap, min(eligible_k_caps), 24)

    def objective(trial) -> float:
        # ── Search grid ───────────────────────────────────────────────────────
        # typereg   – regression model used to map X-neighbors onto Y.
        #             The default grid is intentionally compact; broader menus can
        #             still be passed explicitly via typereg_choices.
        typereg = trial.suggest_categorical("typereg", resolved_typereg_choices)

        # typedist  – distance / similarity metric for ranking analog candidates.
        #             The default grid avoids DTW to keep runtime bounded.
        typedist = trial.suggest_categorical("typedist", resolved_typedist_choices)

        if resolved_scale_method_choices:
            resolved_scale_method = _normalize_scale_method(
                trial.suggest_categorical("scale_method", resolved_scale_method_choices)
            )
        else:
            resolved_scale_method = scale_method

        # k         – number of nearest special-day analogs to keep.
        #             Lower bound comes from optuna_min_k; upper bound is the smallest
        #             realizable analog pool across the retained folds and the
        #             final target run, capped at 24.
        k = trial.suggest_int("k", optuna_min_k, optuna_max_k)

        # n_components – latent dimensions for PCR / PLS only.
        #                Conditioned on typereg so that tree/linear models are not
        #                penalized by an irrelevant parameter.
        if typereg in {"PCR", "PLS"}:
            max_components = max(2, min(k, season_length))
            n_components = trial.suggest_int("n_components", 2, max_components)
        else:
            n_components = int(initial_n_components)

        regressor_params = _suggest_optuna_regressor_params(trial, typereg)

        # dtw_window_frac – Sakoe-Chiba band width expressed as a fraction of
        #                   the season length (vsele).  A value of 1.0 means no
        #                   constraint (full DTW), while 0.05 restricts warping
        #                   to 5 % of the window length.
        #                   This parameter is only active when typedist == "dtw";
        #                   for other distance metrics it is set to None and has
        #                   no effect.
        if typedist == "dtw":
            dtw_window = trial.suggest_float("dtw_window_frac", 0.05, 1.0)
        else:
            dtw_window = None

        fold_metrics = []
        try:
            for target_date in eligible_dates:
                fold_metrics.append(
                    _evaluate_analog_holiday_fold(
                        history_df=history_df,
                        target_date=target_date,
                        season_length=season_length,
                        k=k,
                        typedist=typedist,
                        typereg=typereg,
                        scale_method=resolved_scale_method,
                        n_components=n_components,
                        regressor_params=regressor_params,
                        special_labels=special_labels,
                        include_declared_holidays=include_declared_holidays,
                        include_outliers=include_outliers,
                        min_special_points=min_special_points,
                        min_event_gap=min_event_gap,
                        max_events=max_events,
                        forecast_start_offset_hours=forecast_start_offset_hours,
                        label_column=label_column,
                        selector_cluster_lookup=selector_cluster_lookup,
                        match_target_cluster=match_target_cluster,
                            selector_weekend_lookup=selector_weekend_lookup,
                            recent_weekend_analogs=recent_weekend_analogs,
                        dtw_window=dtw_window,
                    )
                )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return 1e9

        mean_mae = float(np.mean([fold["mae"] for fold in fold_metrics]))
        mape_values = [
            fold["mape_pct"] for fold in fold_metrics if np.isfinite(fold["mape_pct"])
        ]
        mean_mape = float(np.mean(mape_values)) if mape_values else np.nan
        fail_rate = float(np.mean([fold["fail"] for fold in fold_metrics]))

        trial.set_user_attr("fold_metrics", fold_metrics)
        trial.set_user_attr("mean_mae", mean_mae)
        trial.set_user_attr("mean_mape_pct", mean_mape)
        trial.set_user_attr("fail_rate", fail_rate)
        trial.set_user_attr("regressor_params", regressor_params)
        return mean_mae + fail_rate * 1000.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec,
    )

    best_trial = study.best_trial
    best_params = study.best_params.copy()
    best_scale_method = _normalize_scale_method(best_params.get("scale_method", scale_method))
    best_config = {
        "k": int(best_params["k"]),
        "k_range": (int(optuna_min_k), int(optuna_max_k)),
        "typedist": str(best_params["typedist"]),
        "typereg": str(best_params["typereg"]),
        "scale_method": best_scale_method,
        "n_components": int(best_params.get("n_components", initial_n_components)),
        "dtw_window": float(best_params["dtw_window_frac"]) if "dtw_window_frac" in best_params else None,
        "regressor_params": _coerce_regressor_params(best_trial.user_attrs.get("regressor_params")),
    }
    fold_metrics_df = pd.DataFrame(best_trial.user_attrs.get("fold_metrics", []))
    summary_df = pd.DataFrame(
        [
            {"param": "cutoff_train", "value": train_end_ts.date().isoformat()},
            {"param": "eval_dates", "value": len(eligible_dates)},
            {"param": "FORECAST_START_OFFSET_HOURS", "value": int(forecast_start_offset_hours)},
            {"param": "best_mean_mae", "value": round(best_trial.user_attrs["mean_mae"], 6)},
            {"param": "best_mean_mape_pct", "value": round(best_trial.user_attrs["mean_mape_pct"], 3)},
            {"param": "best_fail_rate", "value": round(best_trial.user_attrs["fail_rate"], 3)},
            {"param": "OPTUNA_MIN_K", "value": int(optuna_min_k)},
            {"param": "OPTUNA_MAX_K", "value": int(optuna_max_k)},
            {"param": "SCALE_METHOD_CHOICES", "value": list(resolved_scale_method_choices) if resolved_scale_method_choices else None},
            {"param": "K", "value": best_config["k"]},
            {"param": "TYPEDIST", "value": best_config["typedist"]},
            {"param": "TYPEREG", "value": best_config["typereg"]},
            {"param": "SCALE_METHOD", "value": best_config["scale_method"]},
            {"param": "N_COMPONENTS", "value": best_config["n_components"]},
            {"param": "DTW_WINDOW", "value": best_config["dtw_window"]},
            {"param": "REGRESSOR_PARAMS", "value": _format_regressor_params(best_config["regressor_params"])},
        ]
    )

    return AnalogHolidayOptunaResult(
        best_config=best_config,
        summary_df=summary_df,
        fold_metrics_df=fold_metrics_df,
        eligible_dates=eligible_dates,
        study=study,
    )


def run_analog_holidays_batch(
    target_dates: Sequence[str | pd.Timestamp | tuple[str | pd.Timestamp, Optional[str]]],
    unique_id: str,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    season_length: int = 24,
    forecast_start_offset_hours: int = 0,
    k: Optional[int] = None,
    typedist: str = "pearson",
    typereg: str = "PCR",
    scale_method: Optional[str] = None,
    n_components: int = 3,
    regressor_params: Optional[dict[str, object]] = None,
    levels: Optional[Sequence[int]] = None,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
    expected_target_label: Optional[str] = None,
    selector_features_path: Path | str | None = None,
    cluster_column: str = "analog_cluster",
    match_target_cluster: bool = True,
    recent_weekend_analogs: int = 0,
) -> AnalogHolidayBatchResult:
    """Run AnalogSpecialDays over a batch of target dates and summarize results."""
    regressor_params = _coerce_regressor_params(regressor_params)
    target_items = _normalize_batch_target_items(target_dates)
    selector_cluster_lookup = _resolve_selector_cluster_lookup(
        selector_features_path=selector_features_path,
        cluster_column=cluster_column,
        match_target_cluster=match_target_cluster,
        unique_id=unique_id,
    )
    cluster_filter_label = _resolve_selector_cluster_filter_label(
        selector_features_path=selector_features_path,
        match_target_cluster=match_target_cluster,
        unique_id=unique_id,
    )
    selector_weekend_lookup = _resolve_selector_weekend_like_lookup(
        selector_features_path=selector_features_path,
        recent_weekend_analogs=recent_weekend_analogs,
        unique_id=unique_id,
    )
    runs: Dict[str, AnalogHolidayRun] = {}
    rows = []

    for target_date, holiday_label in target_items:
        try:
            run = run_analog_holidays(
                unique_id=unique_id,
                target_date=target_date,
                source_path=source_path,
                season_length=season_length,
                forecast_start_offset_hours=forecast_start_offset_hours,
                k=k,
                typedist=typedist,
                typereg=typereg,
                scale_method=scale_method,
                n_components=n_components,
                regressor_params=regressor_params,
                levels=levels,
                special_labels=special_labels,
                include_declared_holidays=include_declared_holidays,
                include_outliers=include_outliers,
                min_special_points=min_special_points,
                min_event_gap=min_event_gap,
                max_events=max_events,
                label_column=label_column,
                expected_target_label=expected_target_label,
                selector_features_path=selector_features_path,
                cluster_column=cluster_column,
                match_target_cluster=match_target_cluster,
                selector_cluster_lookup=selector_cluster_lookup,
                selector_weekend_lookup=selector_weekend_lookup,
                recent_weekend_analogs=recent_weekend_analogs,
            )
            runs[target_date] = run

            mae_24h = np.nan
            mape_24h_pct = np.nan
            mae_window = np.nan
            mape_window_pct = np.nan
            if run.actual_profile is not None:
                mae_window = float(np.mean(np.abs(run.forecast_profile - run.actual_profile)))
                ape_pct = _absolute_percentage_error(run.actual_profile, run.forecast_profile)
                if np.isfinite(ape_pct).any():
                    mape_window_pct = float(np.nanmean(ape_pct))
                mae_24h = mae_window
                mape_24h_pct = mape_window_pct

            rows.append(
                {
                    "target_date": target_date,
                    "holiday_label": holiday_label,
                    "analog_cluster": selector_cluster_lookup.get(pd.Timestamp(target_date).normalize(), pd.NA)
                    if selector_cluster_lookup is not None else pd.NA,
                    "cluster_filter_label": cluster_filter_label,
                    "filter_by_cluster": bool(match_target_cluster),
                    "forecast_start": run.forecast_start,
                    "forecast_end": run.forecast_end,
                    "forecast_start_offset_hours": run.forecast_start_offset_hours,
                    "forecast_hours": run.season_length,
                    "target_exists": run.target_exists,
                    "target_has_complete_profile": run.target_has_complete_profile,
                    "selected_analogs": len(run.positions),
                    "fail": run.fail,
                    "mae_window": mae_window,
                    "mape_window_pct": mape_window_pct,
                    "mae_24h": mae_24h,
                    "mape_24h_pct": mape_24h_pct,
                    "t_sel_sec": run.t_sel,
                    "t_reg_sec": run.t_reg,
                    "k": run.k,
                    "typedist": run.typedist,
                    "typereg": run.typereg,
                    "n_components": run.n_components,
                    "regressor_params": _format_regressor_params(run.regressor_params),
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "target_date": target_date,
                    "holiday_label": holiday_label,
                    "analog_cluster": selector_cluster_lookup.get(pd.Timestamp(target_date).normalize(), pd.NA)
                    if selector_cluster_lookup is not None else pd.NA,
                    "cluster_filter_label": cluster_filter_label,
                    "filter_by_cluster": bool(match_target_cluster),
                    "forecast_start": pd.NaT,
                    "forecast_end": pd.NaT,
                    "forecast_start_offset_hours": int(forecast_start_offset_hours),
                    "forecast_hours": int(season_length),
                    "target_exists": False,
                    "target_has_complete_profile": False,
                    "selected_analogs": 0,
                    "fail": True,
                    "mae_window": np.nan,
                    "mape_window_pct": np.nan,
                    "mae_24h": np.nan,
                    "mape_24h_pct": np.nan,
                    "t_sel_sec": np.nan,
                    "t_reg_sec": np.nan,
                    "k": k,
                    "typedist": typedist,
                    "typereg": typereg,
                    "n_components": n_components,
                    "regressor_params": _format_regressor_params(regressor_params),
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(rows)
    metric_summary_df = results_df[["mae_window", "mape_window_pct"]].describe(include="all")
    return AnalogHolidayBatchResult(
        target_items=target_items,
        runs=runs,
        results_df=results_df,
        metric_summary_df=metric_summary_df,
    )


def _load_daily_parquet(path: Path) -> pd.DataFrame:
    """Load the legacy daily parquet cache without changing its schema."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["unique_id", "date"]).reset_index(drop=True)


def _load_hourly_wide_csv(path: Path) -> pd.DataFrame:
    """Normalize the hourly export CSV to the daily wide contract."""
    raw_df = pd.read_csv(path, parse_dates=["ds"])
    if "ds" not in raw_df.columns:
        raise ValueError("Hourly audit CSV must include a 'ds' column")

    value_columns = [
        col for col in raw_df.columns
        if col != "ds" and not col.endswith("_holiday") and not col.endswith("_cluster")
    ]
    if not value_columns:
        raise ValueError("Hourly audit CSV does not contain any series columns")

    daily_frames = []
    for unique_id in value_columns:
        holiday_column = f"{unique_id}_holiday"
        if holiday_column not in raw_df.columns:
            raise ValueError(
                f"Hourly audit CSV is missing the companion holiday flag column "
                f"for '{unique_id}'"
            )

        region_df = raw_df[["ds", unique_id, holiday_column]].copy()
        region_df["date"] = region_df["ds"].dt.normalize()
        region_df["hour_col"] = region_df["ds"].dt.strftime("h_%H")

        daily_profiles = region_df.pivot_table(
            index="date",
            columns="hour_col",
            values=unique_id,
            aggfunc="first",
        )
        daily_profiles = daily_profiles.reindex(columns=HOUR_COLS)
        holiday_flags = (
            region_df.groupby("date")[holiday_column]
            .max()
            .reindex(daily_profiles.index)
            .fillna(0)
            .astype(int)
        )

        daily_frame = daily_profiles.reset_index()
        daily_frame.insert(0, "unique_id", unique_id)
        daily_frame["label"] = np.where(
            holiday_flags.to_numpy() == 1,
            "holiday",
            "normal_day",
        )
        daily_frames.append(daily_frame)

    df = pd.concat(daily_frames, ignore_index=True)
    df["holiday_name"] = pd.NA
    df["holiday_type"] = pd.NA
    df["is_declared_holiday"] = False
    df["is_outlier"] = False
    df["outlier_score"] = np.nan

    ordered_columns = [
        "unique_id",
        "date",
        "label",
        "holiday_name",
        "holiday_type",
        "is_declared_holiday",
        "is_outlier",
        "outlier_score",
        *HOUR_COLS,
    ]
    return df[ordered_columns].sort_values(["unique_id", "date"]).reset_index(drop=True)


def build_selected_days_table(
    train_df: pd.DataFrame,
    positions: Sequence[int],
    neighbors2: np.ndarray,
    season_length: int,
    forecast_start_offset_hours: int = 0,
) -> pd.DataFrame:
    """Map selected hourly positions back to daily context metadata."""
    if len(positions) == 0:
        return pd.DataFrame(
            columns=[
                "context_date", "special_date", "label", "holiday_name",
                "is_declared_holiday", "is_outlier", "outlier_score",
            ]
        )

    steps_per_day = len(HOUR_COLS)
    records = []
    for idx, pos in enumerate(positions):
        context_day_idx = int(pos // steps_per_day)
        special_day_idx = int(
            (pos + season_length + int(forecast_start_offset_hours)) // steps_per_day
        )
        if special_day_idx >= len(train_df):
            continue
        context_row = train_df.iloc[context_day_idx]
        special_row = train_df.iloc[special_day_idx]

        records.append(
            {
                "context_date": pd.Timestamp(context_row["date"]),
                "special_date": pd.Timestamp(special_row["date"]),
                "label": special_row.get("label"),
                "holiday_name": special_row.get("holiday_name"),
                "is_declared_holiday": bool(special_row.get("is_declared_holiday", False)),
                "is_outlier": bool(special_row.get("is_outlier", False)),
                "outlier_score": special_row.get("outlier_score"),
                "profile_max": float(neighbors2[idx].max()) if idx < len(neighbors2) else np.nan,
            }
        )

    return pd.DataFrame(records)


def _absolute_percentage_error(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Return absolute percentage error in percent with zero-safe denominators."""
    denom = np.where(np.abs(reference) > 1e-9, np.abs(reference), np.nan)
    return np.abs(prediction - reference) / denom * 100.0


def build_run_summary(run: AnalogHolidayRun) -> pd.DataFrame:
    """Build a compact summary table for a completed run."""
    mape = np.nan
    target_label = None
    target_name = None

    if run.target_row is not None:
        target_label = run.target_row.get("label")
        target_name = run.target_row.get("holiday_name")

    if run.actual_profile is not None:
        ape_pct = _absolute_percentage_error(run.actual_profile, run.forecast_profile)
        if np.isfinite(ape_pct).any():
            mape = float(np.nanmean(ape_pct))

    rows = [
        ("unique_id", run.unique_id),
        ("target_date", run.target_date.date().isoformat()),
        ("forecast_start", run.forecast_start.strftime("%Y-%m-%d %H:%M")),
        ("forecast_end", run.forecast_end.strftime("%Y-%m-%d %H:%M")),
        ("forecast_start_offset_hours", run.forecast_start_offset_hours),
        ("forecast_window_hours", run.season_length),
        ("target_exists", run.target_exists),
        ("target_has_complete_profile", run.target_has_complete_profile),
        ("target_label", target_label),
        ("target_holiday_name", target_name),
        ("typedist", run.typedist),
        ("typereg", run.typereg),
        ("scale_method", run.scale_method),
        ("regressor_params", _format_regressor_params(run.regressor_params)),
        ("k_neighbors", run.k),
        ("train_days", len(run.train_df)),
        ("special_days_train", int(run.special_day_daily_mask.sum())),
        ("recent_weekend_like", run.recent_weekend_like),
        ("recent_weekend_analogs_requested", run.recent_weekend_analogs),
        ("recent_weekend_analogs_added", len(run.recent_weekend_dates)),
        (
            "recent_weekend_dates",
            ", ".join(date_value.date().isoformat() for date_value in run.recent_weekend_dates),
        ),
        ("selected_analogs", len(run.positions)),
        ("fail", run.fail),
        ("t_sel_sec", round(run.t_sel, 6)),
        ("t_reg_sec", round(run.t_reg, 6)),
        ("mape_window_pct", None if np.isnan(mape) else round(mape, 3)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_analog_ranking_table(run: AnalogHolidayRun) -> pd.DataFrame:
    """Rank selected analog profiles against the best available reference."""
    columns = [
        "analog_index",
        "context_date",
        "special_date",
        "label",
        "mape_vs_reference_pct",
        "profile_max",
        "reference_name",
    ]
    if len(run.neighbors2) == 0:
        return pd.DataFrame(columns=columns)

    reference_profile = run.actual_profile
    reference_name = "actual"
    if reference_profile is None:
        reference_profile = run.forecast_profile
        reference_name = "forecast"

    ranking_df = run.selected_days_df.iloc[: len(run.neighbors2)].copy().reset_index(drop=True)
    ranking_df["analog_index"] = np.arange(len(ranking_df))
    ranking_df["mape_vs_reference_pct"] = [
        float(np.nanmean(_absolute_percentage_error(reference_profile, profile)))
        for profile in run.neighbors2[: len(ranking_df)]
    ]
    ranking_df["reference_name"] = reference_name

    return ranking_df[columns].sort_values(
        ["mape_vs_reference_pct", "special_date"]
    ).reset_index(drop=True)


def _relative_hour_axis(run: AnalogHolidayRun) -> np.ndarray:
    return np.arange(run.season_length) - int(run.forecast_start_offset_hours)


def _pair_relative_hour_axis(
    run: AnalogHolidayRun,
    extra_hours: int = 0,
) -> np.ndarray:
    horizon = run.season_length * 2 + int(extra_hours)
    return np.arange(horizon) - (run.season_length + int(run.forecast_start_offset_hours))


def _hour_tick_step(length: int) -> int:
    if length <= 12:
        return 1
    if length <= 24:
        return 2
    if length <= 48:
        return 4
    return 6


def _row_metric_value(row: pd.Series, *keys: str) -> float:
    for key in keys:
        if key not in row.index:
            continue
        value = row.get(key)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            return value_float
    return np.nan


def _format_metric_value(value: float, decimals: int = 2, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _format_panel_config_value(value: object, default: str = "n/a") -> str:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, float) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _resolve_panel_cluster_filter_value(row: pd.Series) -> object:
    cluster_filter_label = row.get("cluster_filter_label", pd.NA)
    if pd.notna(cluster_filter_label):
        return cluster_filter_label
    return row.get("filter_by_cluster", row.get("match_target_cluster", pd.NA))


def _build_panel_config_row(row: pd.Series) -> str:
    k_value = row.get("k", np.nan)
    total_available = row.get("optuna_k_max", np.nan)
    if pd.isna(total_available):
        total_available = row.get("selected_analogs", np.nan)

    if pd.notna(k_value) and pd.notna(total_available):
        k_text = f"k={int(float(k_value))}/{int(float(total_available))}"
    elif pd.notna(k_value):
        k_text = f"k={int(float(k_value))}/n/a"
    else:
        k_text = "k=n/a"

    scale_method = _format_panel_config_value(row.get("scale_method", pd.NA), default="None")
    typedist = _format_panel_config_value(row.get("typedist", pd.NA))
    typereg = _format_panel_config_value(row.get("typereg", pd.NA))
    n_components = _format_panel_config_value(row.get("n_components", pd.NA))
    cluster_filter = _resolve_panel_cluster_filter_value(row)
    cluster_filter_text = _format_panel_config_value(cluster_filter)

    return (
        f"{k_text} | scale_method={scale_method} |\n"
        f"typedist={typedist} |\n"
        f"typereg={typereg} | n_components={n_components}\n"
        f"cluster={cluster_filter_text}"
    )


def _build_panel_metric_rows(row: pd.Series) -> list[str]:
    mape_38 = _row_metric_value(row, "mape_38_pct", "mape_window_pct", "mape_24h_pct")
    mape_14 = _row_metric_value(row, "mape_14_pct", "mape_head14_pct")
    mape_24 = _row_metric_value(row, "mape_24_pct", "mape_holiday24_raw_pct")
    mae_38 = _row_metric_value(row, "mae_38", "mae_window", "mae_24h")
    mae_14 = _row_metric_value(row, "mae_14", "mae_head14")
    mpe_24 = _row_metric_value(row, "mpe_24_pct")
    bias_38 = _row_metric_value(row, "bias_38")
    bias_14 = _row_metric_value(row, "bias_14", "head_bias_mean")
    bias_24 = _row_metric_value(row, "bias_24", "tail_bias_mean")

    return [
        " | ".join([
            f"MAPE_38={_format_metric_value(mape_38, suffix='%')}",
            f"MAPE_14={_format_metric_value(mape_14, suffix='%')}",
            f"MAPE_24={_format_metric_value(mape_24, suffix='%')}",
        ]),
        " | ".join([
            f"MAE_38={_format_metric_value(mae_38)}",
            f"MAE_14={_format_metric_value(mae_14)}",
            f"MPE_24={_format_metric_value(mpe_24, suffix='%')}",
        ]),
        " | ".join([
            f"BIAS_38={_format_metric_value(bias_38)}",
            f"BIAS_14={_format_metric_value(bias_14)}",
            f"BIAS_24={_format_metric_value(bias_24)}",
        ]),
    ]


def plot_forecast_diagnostics(run: AnalogHolidayRun):
    """Plot a two-panel forecast diagnostic figure."""
    hours = _relative_hour_axis(run)
    tick_step = _hour_tick_step(run.season_length)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax = axes[0]
    ax.plot(hours, run.forecast_profile, color="#d62828", linewidth=2.4, label="Forecast")

    if run.actual_profile is not None:
        ax.plot(hours, run.actual_profile, color="#000000", linewidth=2.1, label="Actual")

    widest = max(run.levels) if run.levels else None
    if widest is not None and widest in run.interval_low and widest in run.interval_high:
        ax.fill_between(
            hours,
            run.interval_low[widest],
            run.interval_high[widest],
            color="#f77f00",
            alpha=0.16,
            label=f"Prediction interval {widest}%",
        )

    ax.set_title(
        f"Forecast diagnostics\n{run.unique_id} | {run.target_date.date()} | "
        f"{run.typedist} | {run.typereg}"
    )
    ax.set_ylabel("Demand")
    ax.set_xticks(hours[::tick_step])
    ax.set_xlim(hours[0], hours[-1])
    ax.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    if run.actual_profile is not None:
        ape_pct = _absolute_percentage_error(run.actual_profile, run.forecast_profile)
        pct_error = np.full(run.season_length, np.nan)
        denom = np.where(np.abs(run.actual_profile) > 1e-9, np.abs(run.actual_profile), np.nan)
        pct_error = (run.forecast_profile - run.actual_profile) / denom * 100.0
        ax2.bar(hours, ape_pct, color="#fcbf49", edgecolor="#8d5524", alpha=0.85, label="APE %")
        ax2.plot(hours, pct_error, color="#003049", marker="o", linewidth=1.2, label="Percentage error %")
        ax2.axhline(0.0, color="#6c757d", linewidth=1.0, linestyle=":")
        ax2.set_title("Hourly percentage error")
        ax2.set_ylabel("Percent")
    else:
        band_width = np.full(run.season_length, np.nan)
        if widest is not None and widest in run.interval_low and widest in run.interval_high:
            band_width = run.interval_high[widest] - run.interval_low[widest]
        ax2.bar(hours, band_width, color="#fcbf49", edgecolor="#8d5524", alpha=0.85, label="Interval width")
        ax2.set_title("Hourly uncertainty")
        ax2.set_ylabel("Demand")

    ax2.set_xlabel("Hour relative to holiday start")
    ax2.set_xticks(hours[::tick_step])
    ax2.set_xlim(hours[0], hours[-1])
    ax2.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    return fig, axes


def plot_ranked_analog_profiles(
    run: AnalogHolidayRun,
    top_n: int = 10,
):
    """Plot the best-ranked analog profiles and their ranking metric."""
    ranking_df = build_analog_ranking_table(run)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    if ranking_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No analog profiles were selected", ha="center", va="center")
            ax.axis("off")
        return fig, axes

    top_profiles = ranking_df.head(top_n).copy()
    hours = _relative_hour_axis(run)
    tick_step = _hour_tick_step(run.season_length)
    reference_profile = run.actual_profile if run.actual_profile is not None else run.forecast_profile
    reference_label = "Actual" if run.actual_profile is not None else "Forecast"
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_profiles)))

    ax = axes[0]
    for (_, row), color in zip(top_profiles.iterrows(), colors):
        profile = run.neighbors2[int(row["analog_index"])]
        special_date = pd.Timestamp(row["special_date"]).date()
        ax.plot(
            hours,
            profile,
            color=color,
            alpha=0.8,
            linewidth=1.3,
            label=f"{special_date} (MAPE={row['mape_vs_reference_pct']:.2f}%)",
        )

    ax.plot(hours, run.forecast_profile, color="#d62828", linewidth=2.2, label="Forecast")
    ax.plot(hours, reference_profile, color="#000000", linewidth=2.1, label=reference_label)
    ax.set_title(f"Top ranked analog profiles\n{run.unique_id} | {run.target_date.date()}")
    ax.set_ylabel("Demand")
    ax.set_xticks(hours[::tick_step])
    ax.set_xlim(hours[0], hours[-1])
    ax.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    bar_df = top_profiles.iloc[::-1].copy()
    bar_labels = [str(pd.Timestamp(value).date()) for value in bar_df["special_date"]]
    bar_colors = list(colors)[::-1]
    ax2.barh(bar_labels, bar_df["mape_vs_reference_pct"], color=bar_colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel(f"MAPE vs {bar_df['reference_name'].iloc[0]} (%)")
    ax2.set_ylabel("Special date")
    ax2.set_title("Ranking by profile MAPE")
    ax2.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    return fig, axes


def plot_batch_inference_grid(
    batch_result: AnalogHolidayBatchResult,
    n_cols: int = 4,
    figsize_per_panel: tuple[float, float] = (5.5, 3.6),
    title: Optional[str] = None,
):
    """Plot batch analog forecasts as a grid of comparable subplots."""
    if not batch_result.target_items:
        raise ValueError("batch_result does not contain target dates to plot.")

    n_items = len(batch_result.target_items)
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    summary_by_date = batch_result.results_df.set_index("target_date")

    legend_handles = None
    legend_labels = None
    interval_fill_color = "#2a9d8f"
    for ax, (target_date, holiday_label) in zip(axes, batch_result.target_items):
        run = batch_result.runs.get(target_date)
        if run is None:
            row = summary_by_date.loc[target_date]
            ax.text(0.5, 0.5, "Run unavailable", ha="center", va="center")
            label_text = holiday_label or "unlabeled"
            ax.set_title(f"{target_date} | {label_text}\nerror: {row.get('error', 'unknown')}")
            ax.axis("off")
            continue

        row = summary_by_date.loc[target_date]
        hours = _relative_hour_axis(run)
        tick_step = _hour_tick_step(run.season_length)

        ax.plot(
            hours,
            run.forecast_profile,
            color="#ef6c00",
            linewidth=2.0,
            label="Forecast",
        )

        levels = sorted(set(run.interval_low) & set(run.interval_high))
        if levels:
            widest = max(levels)
            for level in levels:
                alpha = 0.08 if level == widest else 0.14
                ax.fill_between(
                    hours,
                    run.interval_low[level],
                    run.interval_high[level],
                    color=interval_fill_color,
                    alpha=alpha,
                    label=f"Prediction interval {level}%",
                )

        if run.actual_profile is not None:
            ax.plot(
                hours,
                run.actual_profile,
                color="#1f1f1f",
                linewidth=1.8,
                label="Actual",
            )

        cluster_value = row.get("analog_cluster", pd.NA)
        config_row = _build_panel_config_row(row)
        metric_rows = _build_panel_metric_rows(row)

        label_text = holiday_label or "unlabeled"
        cluster_text = (
            f"analog_cluster={cluster_value}"
            if pd.notna(cluster_value) else "analog_cluster=n/a"
        )
        ax.set_title(
            f"{target_date} | {label_text} | {cluster_text}\n{config_row}\n" + "\n".join(metric_rows),
            fontsize=12,
        )
        ax.grid(alpha=0.2)
        ax.set_xticks(hours[::tick_step])
        ax.set_xlim(hours[0], hours[-1])
        ax.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes[n_items:]:
        ax.axis("off")

    if legend_handles and legend_labels:
        unique_legend = dict(zip(legend_labels, legend_handles))
        fig.legend(
            list(unique_legend.values()),
            list(unique_legend.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.968),
            ncol=min(5, len(unique_legend)),
            frameon=False,
        )

    if title is None:
        sample_run = next(iter(batch_result.runs.values()), None)
        if sample_run is None:
            title = "Batch inference"
        else:
            title = (
                f"Batch inference | {sample_run.unique_id} | {sample_run.typereg} | "
                f"{sample_run.typedist} | k={sample_run.k}"
            )
    top_margin = 0.89 if legend_handles and legend_labels else 0.93
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_margin))
    fig.suptitle(title, fontsize=16, y=0.995)
    return fig, axes


def plot_batch_pair_sequences_grid(
    batch_result: AnalogHolidayBatchResult,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.5, 5.0),
    title: Optional[str] = None,
    adjusted_forecasts_by_date: Optional[Dict[str, np.ndarray]] = None,
    post_holiday_actuals_by_date: Optional[Dict[str, np.ndarray]] = None,
):
    """Grid of X/X2 and Y/Y2 pair-sequence plots, one panel per forecasted date."""
    if not batch_result.target_items:
        raise ValueError("batch_result does not contain target dates to plot.")

    n_items = len(batch_result.target_items)
    n_rows = math.ceil(n_items / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    metrics_by_date = {}
    if not batch_result.results_df.empty:
        metrics_by_date = {
            row["target_date"]: row
            for _, row in batch_result.results_df.iterrows()
        }

    for ax, (target_date, holiday_label) in zip(axes, batch_result.target_items):
        run = batch_result.runs.get(target_date)
        label_text = holiday_label or "unlabeled"

        metric_row = metrics_by_date.get(target_date)
        if metric_row is not None:
            cluster_val = metric_row.get("analog_cluster", pd.NA)
            config_row = _build_panel_config_row(metric_row)
            metric_rows = _build_panel_metric_rows(metric_row)
        else:
            cluster_val = pd.NA
            config_row = (
                f"k={getattr(run, 'k', 'n/a')}/n/a | scale_method={getattr(run, 'scale_method', None) or 'None'} |\n"
                f"typedist={getattr(run, 'typedist', 'n/a')} |\n"
                f"typereg={getattr(run, 'typereg', 'n/a')} | n_components={getattr(run, 'n_components', 'n/a')}\n"
                f"cluster=n/a"
            ) if run is not None else "k=n/a | scale_method=None |\ntypedist=n/a |\ntypereg=n/a | n_components=n/a\ncluster=n/a"
            metric_rows = [
                "MAPE_38=n/a | MAPE_14=n/a | MAPE_24=n/a",
                "MAE_38=n/a | MAE_14=n/a | MPE_24=n/a",
                "BIAS_38=n/a | BIAS_14=n/a | BIAS_24=n/a",
            ]

        cluster_text = (
            f"analog_cluster={cluster_val}"
            if pd.notna(cluster_val) else "analog_cluster=n/a"
        )

        if run is not None:
            window_start = run.forecast_start.strftime("%m-%d %H:%M")
            window_end = (run.forecast_end - pd.Timedelta(hours=1)).strftime("%m-%d %H:%M")
            date_pair_text = f"{window_start} → {window_end}"
        else:
            target_ts = pd.Timestamp(target_date)
            prev_ts = target_ts - pd.Timedelta(days=1)
            date_pair_text = f"{prev_ts.date()}  |  {target_ts.date()}"

        panel_title = (
            f"{label_text} | {cluster_text}\n"
            f"{config_row}\n"
            f"{date_pair_text}\n"
            f"{metric_rows[0]}\n"
            f"{metric_rows[1]}\n"
            f"{metric_rows[2]}\n"
            f"<-pre-holiday (X / Y)   holiday (X' / Y')   post-holiday recovery (+24h real) ->"
        )

        if run is None:
            ax.text(0.5, 0.5, "Run unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel_title, fontsize=11)
            ax.axis("off")
            continue

        adjusted_forecast_profile = None
        if adjusted_forecasts_by_date is not None:
            adjusted_forecast_profile = adjusted_forecasts_by_date.get(target_date)
        post_holiday_actual_profile = None
        if post_holiday_actuals_by_date is not None:
            post_holiday_actual_profile = post_holiday_actuals_by_date.get(target_date)

        plot_analog_pair_sequences(
            run,
            ax=ax,
            adjusted_forecast_profile=adjusted_forecast_profile,
            post_holiday_actual_profile=post_holiday_actual_profile,
        )

        ax.set_title(panel_title, fontsize=11, color="#ff8000")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n_items:]:
        ax.axis("off")

    if title is None:
        sample_run = next(iter(batch_result.runs.values()), None)
        if sample_run is not None:
            title = (
                f"X/X' and Y/Y' by forecast date | {sample_run.unique_id} | "
                f"{sample_run.typereg} | {sample_run.typedist} | k={sample_run.k}"
            )
        else:
            title = "X/X' and Y/Y' by forecast date"

    fig.suptitle(title, fontsize=15, y=1.01, color="#ff8000")
    fig.tight_layout()
    return fig, axes


def plot_recent_context(
    run: AnalogHolidayRun,
    lookback_days: int = 45,
    ax=None,
):
    """Plot recent daily totals and highlight special days and target date."""
    recent = run.train_df.tail(lookback_days).copy()
    recent["daily_total"] = recent[HOUR_COLS].sum(axis=1)
    recent["is_special"] = run.special_day_daily_mask.reindex(recent.index).astype(bool)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    ax.plot(recent["date"], recent["daily_total"], color="#355070", linewidth=1.5, label="Daily total")
    special = recent[recent["is_special"]]
    if not special.empty:
        ax.scatter(
            special["date"],
            special["daily_total"],
            color="#c1121f",
            s=50,
            label="Training special days",
            zorder=3,
        )

    ax.axvline(run.target_date, color="#ff7f11", linestyle="--", linewidth=1.5, label="Target date")
    ax.set_title(f"Recent context: {run.unique_id} toward {run.target_date.date()}")
    ax.set_ylabel("24h sum")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    return fig, ax


def plot_forecast_vs_actual(run: AnalogHolidayRun, ax=None):
    """Plot forecast interval and actual target profile when available."""
    hours = _relative_hour_axis(run)
    tick_step = _hour_tick_step(run.season_length)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    ax.plot(hours, run.forecast_profile, color="#d62828", linewidth=2.4, label="AnalogSpecialDays forecast")

    if run.actual_profile is not None:
        ax.plot(hours, run.actual_profile, color="#003049", linewidth=2.0, label="Actual target")
    else:
        ax.text(
            0.02,
            0.95,
            "Actual profile unavailable or incomplete in the audit source",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#6c757d",
        )

    if run.levels:
        widest = max(run.levels)
        if widest in run.interval_low and widest in run.interval_high:
            ax.fill_between(
                hours,
                run.interval_low[widest],
                run.interval_high[widest],
                color="#f77f00",
                alpha=0.18,
                label=f"Prediction interval {widest}%",
            )

    subtitle = f"{run.unique_id} | {run.target_date.date()} | {run.typedist} | {run.typereg}"
    ax.set_title(f"Hourly forecast vs actual\n{subtitle}")
    ax.set_xlabel("Hour relative to holiday start")
    ax.set_ylabel("Demand")
    ax.set_xticks(hours[::tick_step])
    ax.set_xlim(hours[0], hours[-1])
    ax.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_selected_analog_profiles(
    run: AnalogHolidayRun,
    max_profiles: int = 12,
    ax=None,
):
    """Plot selected analog profiles together with forecast and actual."""
    hours = _relative_hour_axis(run)
    tick_step = _hour_tick_step(run.season_length)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    plotted = 0
    for idx, profile in enumerate(run.neighbors2[:max_profiles]):
        label = None
        if idx < len(run.selected_days_df):
            special_date = pd.Timestamp(run.selected_days_df.iloc[idx]["special_date"]).date()
            label = f"Analog {special_date}"
        ax.plot(hours, profile, alpha=0.35, linewidth=1.2, color="#669bbc", label=label)
        plotted += 1

    ax.plot(hours, run.forecast_profile, color="#d62828", linewidth=2.4, label="Forecast")
    if run.actual_profile is not None:
        ax.plot(hours, run.actual_profile, color="#003049", linewidth=2.0, label="Actual")

    ax.set_title(f"Selected special profiles ({plotted})\n{run.unique_id} | {run.target_date.date()}")
    ax.set_xlabel("Hour relative to holiday start")
    ax.set_ylabel("Demand")
    ax.set_xticks(hours[::tick_step])
    ax.set_xlim(hours[0], hours[-1])
    ax.axvline(0, color="#6c757d", linewidth=1.0, linestyle=":")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", ncol=2)
    return fig, ax


def plot_analog_pair_sequences(
    run: AnalogHolidayRun,
    max_pairs: Optional[int] = None,
    ax=None,
    adjusted_forecast_profile: Optional[np.ndarray] = None,
    adjusted_forecast_label: str = "Adjusted forecast Y'",
    post_holiday_actual_profile: Optional[np.ndarray] = None,
):
    """Plot historical X/X' pairs together with the forecast Y' sequence."""
    pair_length = run.season_length
    recovery_hours_count = POST_HOLIDAY_RECOVERY_HOURS
    horizon = pair_length * 2 + recovery_hours_count
    hours = _pair_relative_hour_axis(run, extra_hours=recovery_hours_count)
    tick_step = _hour_tick_step(horizon)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure

    valid_positions = [
        pos
        for pos in run.positions
        if pos + 2 * pair_length <= len(run.hourly_series)
    ]
    context_profiles = np.array(
        [run.hourly_series[pos:pos + pair_length] for pos in valid_positions],
        dtype=np.float64,
    )
    future_profiles = run.neighbors2[: len(valid_positions)]

    if max_pairs is not None:
        valid_positions = valid_positions[:max_pairs]
        context_profiles = context_profiles[:max_pairs]
        future_profiles = future_profiles[:max_pairs]

    pair_count = min(len(valid_positions), len(future_profiles))
    valid_positions = valid_positions[:pair_count]
    if pair_count > 0:
        pair_paths = np.hstack([context_profiles[:pair_count], future_profiles[:pair_count]])
        pair_hours = hours[: pair_length * 2]
        for idx, pair in enumerate(pair_paths):
            label = "Historical X/X' pairs" if idx == 0 else None
            ax.plot(pair_hours, pair, color="#669bbc", alpha=0.28, linewidth=1.4, label=label)

    context_hours = hours[:pair_length]
    forecast_hours = hours[pair_length: pair_length * 2]
    recovery_hours = hours[pair_length * 2:]

    recovery_label_added = False
    for pos in valid_positions:
        recovery_start = pos + 2 * pair_length
        recovery_end = recovery_start + recovery_hours_count
        if recovery_end > len(run.hourly_series):
            continue
        recovery_profile = run.hourly_series[recovery_start:recovery_end]
        ax.plot(
            recovery_hours,
            recovery_profile,
            color="#669bbc",
            alpha=0.24,
            linewidth=1.2,
            linestyle="--",
            label="Historical recovery +24h" if not recovery_label_added else None,
        )
        recovery_label_added = True

    # Y: context window immediately before the forecast start.
    if run.previous_day_profile is not None and len(run.previous_day_profile) == pair_length:
        ax.plot(
            context_hours,
            run.previous_day_profile,
            color="#000000",
            linewidth=2.2,
            label="Y (pre-holiday)",
        )

    ax.plot(
        forecast_hours,
        run.forecast_profile,
        color="#d62828",
        linewidth=2.8,
        label="Forecast Y'",
    )

    if adjusted_forecast_profile is not None:
        adjusted_forecast_profile = np.asarray(adjusted_forecast_profile, dtype=np.float64)
        if adjusted_forecast_profile.shape[0] == pair_length:
            ax.plot(
                forecast_hours,
                adjusted_forecast_profile,
                color="#1d4ed8",
                linewidth=2.4,
                linestyle="--",
                label=adjusted_forecast_label,
            )

    if run.actual_profile is not None:
        ax.plot(
            forecast_hours,
            run.actual_profile,
            color="#000000",
            linewidth=2.0,
            label="Actual Y'",
        )

    if post_holiday_actual_profile is None:
        post_holiday_actual_profile = getattr(run, "post_holiday_actual_profile", None)

    if (
        post_holiday_actual_profile is not None
        and len(post_holiday_actual_profile) == recovery_hours_count
    ):
        ax.plot(
            recovery_hours,
            post_holiday_actual_profile,
            color="#000000",
            linewidth=1.9,
            linestyle="--",
            label="Actual recovery +24h",
        )

    forecast_boundary = forecast_hours[0] - 0.5
    recovery_boundary = recovery_hours[0] - 0.5
    ax.axvline(forecast_boundary, color="#ff8000", linestyle=":", linewidth=1.2)
    ax.axvline(recovery_boundary, color="#2a9d8f", linestyle=":", linewidth=1.0)
    ax.axvspan(hours[0] - 0.5, forecast_boundary, color="#ff8000", alpha=0.04)
    ax.axvspan(forecast_boundary, recovery_boundary, color="#d62828", alpha=0.035)
    ax.axvspan(recovery_boundary, hours[-1] + 0.5, color="#2a9d8f", alpha=0.03)
    if run.forecast_start_offset_hours > 0:
        ax.axvline(-0.5, color="#6c757d", linestyle="--", linewidth=1.0, label="Holiday start")

    ax.set_title("X/X' selection, Y' forecast, and recovery day", fontsize="x-large", color="#ff8000")
    ax.set_xlabel("Hour relative to holiday start", color="#ff8000", fontsize="large")
    ax.set_ylabel("Demand", color="#ff8000", fontsize="large")
    ax.set_xticks(hours[::tick_step])
    ax.set_xlim(hours[0], hours[-1])
    ax.tick_params(colors="#ff8000", which="both")
    ax.spines["bottom"].set_color("#ff8000")
    ax.spines["top"].set_color("#ff8000")
    ax.spines["right"].set_color("#ff8000")
    ax.spines["left"].set_color("#ff8000")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig, ax


def _get_region_df(df: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    df_region = df[df["unique_id"] == unique_id].copy()
    if df_region.empty:
        raise ValueError(f"unique_id '{unique_id}' is not present in the audit source")
    return df_region.sort_values("date").reset_index(drop=True)


def _flatten_daily_profiles(df_region: pd.DataFrame) -> np.ndarray:
    return df_region[HOUR_COLS].to_numpy(dtype=np.float64).reshape(-1)


def _complete_day_mask(df_region: pd.DataFrame) -> pd.Series:
    return df_region[HOUR_COLS].notna().all(axis=1)


def _evaluate_analog_holiday_fold(
    history_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    season_length: int,
    k: Optional[int],
    typedist: str,
    typereg: str,
    scale_method: Optional[str],
    n_components: int,
    regressor_params: Optional[dict[str, object]],
    special_labels: Sequence[str],
    include_declared_holidays: bool,
    include_outliers: bool,
    min_special_points: Optional[int],
    min_event_gap: Optional[int],
    max_events: Optional[int],
    forecast_start_offset_hours: int,
    label_column: str,
    selector_cluster_lookup: Optional[dict[pd.Timestamp, object]] = None,
    match_target_cluster: bool = False,
    selector_weekend_lookup: Optional[dict[pd.Timestamp, str]] = None,
    recent_weekend_analogs: int = 0,
    dtw_window: Optional[float] = None,
) -> dict[str, object]:
    """Evaluate a single historical special date as a tuning fold."""
    target_ts = pd.Timestamp(target_date).normalize()
    regressor_params = _coerce_regressor_params(regressor_params)
    forecast_start_offset_hours = _normalize_forecast_start_offset_hours(
        forecast_start_offset_hours
    )
    forecast_start = target_ts - pd.Timedelta(hours=forecast_start_offset_hours)
    train_df = history_df.loc[history_df["date"] < target_ts].copy()
    target_row = history_df.loc[history_df["date"] == target_ts]

    if train_df.empty or target_row.empty:
        raise ValueError(f"Invalid fold for {target_ts.date()}.")

    hourly_series = _truncate_hourly_history(
        _flatten_daily_profiles(train_df),
        forecast_start_offset_hours,
    )
    if len(hourly_series) < 2 * season_length + 1:
        raise ValueError(
            f"Insufficient history before {forecast_start} for season_length={season_length}."
        )

    special_mask, _, _ = _build_analog_candidate_daily_mask(
        train_df,
        target_ts,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
        selector_cluster_lookup=selector_cluster_lookup,
        match_target_cluster=match_target_cluster,
        selector_weekend_lookup=selector_weekend_lookup,
        recent_weekend_analogs=recent_weekend_analogs,
    )

    special_hourly_mask = np.repeat(
        special_mask.astype(float).to_numpy(),
        len(HOUR_COLS),
    )
    special_hourly_mask = _truncate_hourly_history(
        special_hourly_mask,
        forecast_start_offset_hours,
    )
    actual_profile = _extract_hour_window(
        history_df,
        window_start=forecast_start,
        length_hours=season_length,
    )
    if actual_profile is None:
        raise ValueError(
            f"Incomplete actual window for {target_ts.date()} and season_length={season_length}."
        )

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        regressor_params=regressor_params,
        scale_method=scale_method,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
        dtw_window=dtw_window,
    )
    prediction, _, _, fail, positions = model.predict_single(
        y=hourly_series,
        special_days=special_hourly_mask,
    )

    mae = float(np.mean(np.abs(prediction - actual_profile)))
    ape_pct = _absolute_percentage_error(actual_profile, prediction)
    mape_pct = float(np.nanmean(ape_pct)) if np.isfinite(ape_pct).any() else np.nan

    return {
        "target_date": target_ts.date().isoformat(),
        "mae": mae,
        "mape_pct": mape_pct,
        "selected_analogs": int(len(positions)),
        "fail": bool(fail),
        "train_days": int(len(train_df)),
        "train_special_days": int(special_mask.sum()),
    }


def _normalize_batch_target_items(
    target_dates: Sequence[str | pd.Timestamp | tuple[str | pd.Timestamp, Optional[str]]],
) -> list[tuple[str, Optional[str]]]:
    """Normalize batch target dates into (iso_date, label) tuples."""
    normalized = []
    for item in target_dates:
        if isinstance(item, tuple):
            target_date, holiday_label = item
        else:
            target_date, holiday_label = item, None
        normalized.append((pd.Timestamp(target_date).date().isoformat(), holiday_label))
    return normalized