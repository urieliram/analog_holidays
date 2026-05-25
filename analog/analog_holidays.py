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
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analog_special_days import AnalogSpecialDays, analog_special_days_core
from analog_holidays.audit.data_loader import HOUR_COLS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PACKAGE_ROOT / "audit" / "data"
DEFAULT_LEVELS = [80, 95]


@dataclass
class AnalogHolidayRun:
    unique_id: str
    target_date: pd.Timestamp
    target_exists: bool
    target_has_complete_profile: bool
    typedist: str
    typereg: str
    season_length: int
    k: Optional[int]
    n_components: int
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
    label_column: str


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


def run_analog_holidays(
    unique_id: str,
    target_date: str | pd.Timestamp,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    season_length: int = 24,
    k: Optional[int] = None,
    typedist: str = "pearson",
    typereg: str = "PCR",
    n_components: int = 3,
    levels: Optional[Sequence[int]] = None,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
    expected_target_label: Optional[str] = "holiday",
) -> AnalogHolidayRun:
    """Run a 24-hour AnalogSpecialDays forecast for a target date."""
    target_ts = pd.Timestamp(target_date)
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

    hourly_series = _flatten_daily_profiles(train_df)
    if len(hourly_series) < 2 * season_length + 1:
        raise ValueError(
            "History is too short for AnalogSpecialDays. "
            f"At least {2 * season_length + 1} hourly points are required."
        )

    special_day_daily_mask = build_special_day_daily_mask(
        train_df,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
    )

    special_day_hourly_mask = np.repeat(
        special_day_daily_mask.astype(float).to_numpy(),
        len(HOUR_COLS),
    )

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
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
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )

    selected_days_df = build_selected_days_table(
        train_df=train_df,
        positions=positions,
        neighbors2=neighbors2,
        season_length=season_length,
    )

    previous_day_profile = train_df.iloc[-1][HOUR_COLS].to_numpy(dtype=np.float64)
    target_has_complete_profile = False
    actual_profile = None
    if target_row is not None:
        candidate_actual = target_row[HOUR_COLS].to_numpy(dtype=np.float64)
        if np.isfinite(candidate_actual).all():
            actual_profile = candidate_actual
            target_has_complete_profile = True

    return AnalogHolidayRun(
        unique_id=unique_id,
        target_date=target_ts,
        target_exists=target_row is not None,
        target_has_complete_profile=target_has_complete_profile,
        typedist=typedist,
        typereg=typereg,
        season_length=season_length,
        k=k,
        n_components=n_components,
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
        label_column=label_column,
    )


def tune_analog_holidays_optuna(
    unique_id: str,
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    train_end: str | pd.Timestamp = "2024-01-01",
    season_length: int = 24,
    initial_k: Optional[int] = None,
    initial_typedist: str = "pearson",
    initial_typereg: str = "PCR",
    initial_n_components: int = 3,
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
) -> AnalogHolidayOptunaResult:
    """Tune AnalogSpecialDays hyperparameters on historical special dates."""
    import importlib.util

    import optuna

    train_end_ts = pd.Timestamp(train_end)
    df = load_audit_source(source_path)
    df_region = _get_region_df(df, unique_id)
    complete_mask = _complete_day_mask(df_region)
    history_df = df_region.loc[
        (df_region["date"] < train_end_ts) & complete_mask
    ].copy()

    if history_df.empty:
        raise ValueError(
            f"No complete history is available for {unique_id} before {train_end_ts.date()}."
        )

    # ── Build search spaces ───────────────────────────────────────────────────
    # typedist_choices: always include Pearson correlation and Euclidean distance;
    # DTW is appended only when dtw-python is installed so that missing the
    # optional dependency does not break the tuning run.
    if typedist_choices is None:
        resolved_typedist_choices = ["pearson", "euclidian"]
        if importlib.util.find_spec("dtw") is not None:
            resolved_typedist_choices.append("dtw")
    else:
        resolved_typedist_choices = [str(choice) for choice in typedist_choices]

    # typereg_choices: full regressor menu by default; override via typereg_choices
    # to restrict the search to a specific subset (e.g. ["PCR", "RF"]).
    resolved_typereg_choices = [
        "PCR",
        "PLS",
        "RidgeReg",
        "LassoReg",
        "RF",
        "Boosting",
    ] if typereg_choices is None else [str(choice) for choice in typereg_choices]

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

    baseline_k = min(3, max(1, int(initial_k) if initial_k is not None else 3))
    eligible_dates: list[pd.Timestamp] = []
    for candidate_date in candidate_dates:
        try:
            _evaluate_analog_holiday_fold(
                history_df=history_df,
                target_date=candidate_date,
                season_length=season_length,
                k=baseline_k,
                typedist=initial_typedist,
                typereg=initial_typereg,
                n_components=initial_n_components,
                special_labels=special_labels,
                include_declared_holidays=include_declared_holidays,
                include_outliers=include_outliers,
                min_special_points=min_special_points,
                min_event_gap=min_event_gap,
                max_events=max_events,
                label_column=label_column,
            )
            eligible_dates.append(pd.Timestamp(candidate_date))
        except ValueError:
            continue

    if max_eval_dates is not None:
        eligible_dates = eligible_dates[-int(max_eval_dates):]

    if len(eligible_dates) < 2:
        raise ValueError(
            "At least 2 eligible special events are required before the cutoff "
            "to tune hyperparameters."
        )

    # Upper bound for k: at most as many candidates as labeled special days, capped
    # at 24 to keep evaluation time reasonable.
    optuna_max_k = max(1, min(int(special_mask_history.sum()), 24))

    def objective(trial) -> float:
        # ── Search grid ───────────────────────────────────────────────────────
        # typereg   – regression model used to map X-neighbors onto Y.
        #             All sklearn-compatible regressors in _REGRESSORS are eligible.
        typereg = trial.suggest_categorical("typereg", resolved_typereg_choices)

        # typedist  – distance / similarity metric for ranking analog candidates.
        #             "dtw" is only included when dtw-python is installed.
        typedist = trial.suggest_categorical("typedist", resolved_typedist_choices)

        # k         – number of nearest special-day analogs to keep.
        #             Upper bound is the total number of labeled special days in
        #             the training window, capped at 24.
        k = trial.suggest_int("k", 1, optuna_max_k)

        # n_components – latent dimensions for PCR / PLS only.
        #                Conditioned on typereg so that tree/linear models are not
        #                penalized by an irrelevant parameter.
        if typereg in {"PCR", "PLS"}:
            max_components = max(2, min(k, season_length))
            n_components = trial.suggest_int("n_components", 2, max_components)
        else:
            n_components = int(initial_n_components)

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
                        n_components=n_components,
                        special_labels=special_labels,
                        include_declared_holidays=include_declared_holidays,
                        include_outliers=include_outliers,
                        min_special_points=min_special_points,
                        min_event_gap=min_event_gap,
                        max_events=max_events,
                        label_column=label_column,
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
    best_config = {
        "k": int(best_params["k"]),
        "typedist": str(best_params["typedist"]),
        "typereg": str(best_params["typereg"]),
        "n_components": int(best_params.get("n_components", initial_n_components)),
        "dtw_window": float(best_params["dtw_window_frac"]) if "dtw_window_frac" in best_params else None,
    }
    fold_metrics_df = pd.DataFrame(best_trial.user_attrs.get("fold_metrics", []))
    summary_df = pd.DataFrame(
        [
            {"param": "cutoff_train", "value": train_end_ts.date().isoformat()},
            {"param": "eval_dates", "value": len(eligible_dates)},
            {"param": "best_mean_mae", "value": round(best_trial.user_attrs["mean_mae"], 6)},
            {"param": "best_mean_mape_pct", "value": round(best_trial.user_attrs["mean_mape_pct"], 3)},
            {"param": "best_fail_rate", "value": round(best_trial.user_attrs["fail_rate"], 3)},
            {"param": "K", "value": best_config["k"]},
            {"param": "TYPEDIST", "value": best_config["typedist"]},
            {"param": "TYPEREG", "value": best_config["typereg"]},
            {"param": "N_COMPONENTS", "value": best_config["n_components"]},
            {"param": "DTW_WINDOW", "value": best_config["dtw_window"]},
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
    k: Optional[int] = None,
    typedist: str = "pearson",
    typereg: str = "PCR",
    n_components: int = 3,
    levels: Optional[Sequence[int]] = None,
    special_labels: Sequence[str] = ("holiday", "special_day"),
    include_declared_holidays: bool = False,
    include_outliers: bool = False,
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
    expected_target_label: Optional[str] = None,
) -> AnalogHolidayBatchResult:
    """Run AnalogSpecialDays over a batch of target dates and summarize results."""
    target_items = _normalize_batch_target_items(target_dates)
    runs: Dict[str, AnalogHolidayRun] = {}
    rows = []

    for target_date, holiday_label in target_items:
        try:
            run = run_analog_holidays(
                unique_id=unique_id,
                target_date=target_date,
                source_path=source_path,
                season_length=season_length,
                k=k,
                typedist=typedist,
                typereg=typereg,
                n_components=n_components,
                levels=levels,
                special_labels=special_labels,
                include_declared_holidays=include_declared_holidays,
                include_outliers=include_outliers,
                min_special_points=min_special_points,
                min_event_gap=min_event_gap,
                max_events=max_events,
                label_column=label_column,
                expected_target_label=expected_target_label,
            )
            runs[target_date] = run

            mae_24h = np.nan
            mape_24h_pct = np.nan
            if run.actual_profile is not None:
                mae_24h = float(np.mean(np.abs(run.forecast_profile - run.actual_profile)))
                ape_pct = _absolute_percentage_error(run.actual_profile, run.forecast_profile)
                if np.isfinite(ape_pct).any():
                    mape_24h_pct = float(np.nanmean(ape_pct))

            rows.append(
                {
                    "target_date": target_date,
                    "holiday_label": holiday_label,
                    "target_exists": run.target_exists,
                    "target_has_complete_profile": run.target_has_complete_profile,
                    "selected_analogs": len(run.positions),
                    "fail": run.fail,
                    "mae_24h": mae_24h,
                    "mape_24h_pct": mape_24h_pct,
                    "t_sel_sec": run.t_sel,
                    "t_reg_sec": run.t_reg,
                    "k": run.k,
                    "typedist": run.typedist,
                    "typereg": run.typereg,
                    "n_components": run.n_components,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "target_date": target_date,
                    "holiday_label": holiday_label,
                    "target_exists": False,
                    "target_has_complete_profile": False,
                    "selected_analogs": 0,
                    "fail": True,
                    "mae_24h": np.nan,
                    "mape_24h_pct": np.nan,
                    "t_sel_sec": np.nan,
                    "t_reg_sec": np.nan,
                    "k": k,
                    "typedist": typedist,
                    "typereg": typereg,
                    "n_components": n_components,
                    "error": str(exc),
                }
            )

    results_df = pd.DataFrame(rows)
    metric_summary_df = results_df[["mae_24h", "mape_24h_pct"]].describe(include="all")
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
        col for col in raw_df.columns if col != "ds" and not col.endswith("_holiday")
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
        special_day_idx = int((pos + season_length) // steps_per_day)
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
        ("target_exists", run.target_exists),
        ("target_has_complete_profile", run.target_has_complete_profile),
        ("target_label", target_label),
        ("target_holiday_name", target_name),
        ("typedist", run.typedist),
        ("typereg", run.typereg),
        ("k_neighbors", run.k),
        ("train_days", len(run.train_df)),
        ("special_days_train", int(run.special_day_daily_mask.sum())),
        ("selected_analogs", len(run.positions)),
        ("fail", run.fail),
        ("t_sel_sec", round(run.t_sel, 6)),
        ("t_reg_sec", round(run.t_reg, 6)),
        ("mape_24h_pct", None if np.isnan(mape) else round(mape, 3)),
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


def plot_forecast_diagnostics(run: AnalogHolidayRun):
    """Plot a two-panel forecast diagnostic figure."""
    hours = np.arange(run.season_length)
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
    ax.set_xticks(hours)
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

    ax2.set_xlabel("Hour")
    ax2.set_xticks(hours)
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
    hours = np.arange(run.season_length)
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
    ax.set_xticks(hours)
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
        hours = np.arange(run.season_length)

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

        metric_parts = []
        if pd.notna(row.get("mape_24h_pct", np.nan)):
            metric_parts.append(f"MAPE {row['mape_24h_pct']:.2f}%")
        if pd.notna(row.get("mae_24h", np.nan)):
            metric_parts.append(f"MAE {row['mae_24h']:.0f}")
        if pd.notna(row.get("k", np.nan)):
            metric_parts.append(f"k={int(row['k'])}")

        label_text = holiday_label or "unlabeled"
        ax.set_title(
            f"{target_date} | {label_text}\n" + " | ".join(metric_parts),
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        ax.set_xticks(hours[::6])
        ax.set_xlim(0, run.season_length - 1)

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
            title = "Inferencias batch"
        else:
            title = (
                f"Inferencias batch | {sample_run.unique_id} | {sample_run.typereg} | "
                f"{sample_run.typedist} | k={sample_run.k}"
            )
    top_margin = 0.89 if legend_handles and legend_labels else 0.93
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_margin))
    fig.suptitle(title, fontsize=14, y=0.995)
    return fig, axes


def plot_batch_pair_sequences_grid(
    batch_result: AnalogHolidayBatchResult,
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.5, 5.0),
    title: Optional[str] = None,
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
        target_ts = pd.Timestamp(target_date)
        prev_ts = target_ts - pd.Timedelta(days=1)
        date_pair_text = f"{prev_ts.date()}  |  {target_ts.date()}"
        label_text = holiday_label or "sin etiqueta"

        metric_row = metrics_by_date.get(target_date)
        if metric_row is not None:
            mape_val = metric_row.get("mape_24h_pct", np.nan)
            mae_val = metric_row.get("mae_24h", np.nan)
            k_val = metric_row.get("k", None)
            typereg_val = metric_row.get("typereg", "")
            typedist_val = metric_row.get("typedist", "")
        else:
            mape_val = np.nan
            mae_val = np.nan
            k_val = getattr(run, "k", None) if run is not None else None
            typereg_val = getattr(run, "typereg", "") if run is not None else ""
            typedist_val = getattr(run, "typedist", "") if run is not None else ""

        mape_text = f"MAPE={mape_val:.2f}%" if np.isfinite(mape_val) else "MAPE=n/a"
        mae_text = f"MAE={mae_val:.2f}" if np.isfinite(mae_val) else "MAE=n/a"
        k_text = f"k={int(k_val)}" if k_val is not None and not (isinstance(k_val, float) and np.isnan(k_val)) else "k=n/a"
        metrics_text = f"{mape_text} | {mae_text} | {k_text} | {typereg_val} | {typedist_val}"

        panel_title = (
            f"{label_text}\n"
            f"{date_pair_text}\n"
            f"← día previo            día objetivo →\n"
            f"{metrics_text}"
        )

        if run is None:
            ax.text(0.5, 0.5, "Run unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel_title, fontsize=9)
            ax.axis("off")
            continue

        plot_analog_pair_sequences(run, ax=ax)

        ax.set_title(panel_title, fontsize=9, color="#ff8000")
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n_items:]:
        ax.axis("off")

    if title is None:
        sample_run = next(iter(batch_result.runs.values()), None)
        if sample_run is not None:
            title = (
                f"X/X2 y Y/Y2 por fecha pronosticada | {sample_run.unique_id} | "
                f"{sample_run.typereg} | {sample_run.typedist} | k={sample_run.k}"
            )
        else:
            title = "X/X2 y Y/Y2 por fecha pronosticada"

    fig.suptitle(title, fontsize=13, y=1.01, color="#ff8000")
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
    hours = np.arange(run.season_length)

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
    ax.set_xlabel("Hour")
    ax.set_ylabel("Demand")
    ax.set_xticks(hours)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_selected_analog_profiles(
    run: AnalogHolidayRun,
    max_profiles: int = 12,
    ax=None,
):
    """Plot selected analog profiles together with forecast and actual."""
    hours = np.arange(run.season_length)

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
    ax.set_xlabel("Hour")
    ax.set_ylabel("Demand")
    ax.set_xticks(hours)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", ncol=2)
    return fig, ax


def plot_analog_pair_sequences(
    run: AnalogHolidayRun,
    max_pairs: Optional[int] = None,
    ax=None,
):
    """Plot historical X/X2 pairs together with the forecast Y2 sequence."""
    pair_length = run.season_length
    horizon = pair_length * 2
    hours = np.arange(horizon)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure

    context_profiles = np.array(
        [
            run.hourly_series[pos:pos + pair_length]
            for pos in run.positions
            if pos + 2 * pair_length <= len(run.hourly_series)
        ],
        dtype=np.float64,
    )
    future_profiles = run.neighbors2[: len(context_profiles)]

    if max_pairs is not None:
        context_profiles = context_profiles[:max_pairs]
        future_profiles = future_profiles[:max_pairs]

    pair_count = min(len(context_profiles), len(future_profiles))
    if pair_count > 0:
        pair_paths = np.hstack([context_profiles[:pair_count], future_profiles[:pair_count]])
        for idx, pair in enumerate(pair_paths):
            label = "Historical X/X2 pairs" if idx == 0 else None
            ax.plot(hours, pair, color="#669bbc", alpha=0.28, linewidth=1.4, label=label)

    # Y: context window of the target date (day immediately before the forecast)
    if run.previous_day_profile is not None and len(run.previous_day_profile) == pair_length:
        ax.plot(
            np.arange(pair_length),
            run.previous_day_profile,
            color="#000000",
            linewidth=2.2,
            label="Y (context)",
        )

    ax.plot(
        np.arange(pair_length, horizon),
        run.forecast_profile,
        color="#d62828",
        linewidth=2.8,
        label="Forecast Y2",
    )

    if run.actual_profile is not None:
        ax.plot(
            np.arange(pair_length, horizon),
            run.actual_profile,
            color="#000000",
            linewidth=2.0,
            label="Actual Y2",
        )

    ax.axvline(pair_length - 0.5, color="#ff8000", linestyle=":", linewidth=1.2)
    ax.axvspan(-0.5, pair_length - 0.5, color="#ff8000", alpha=0.04)
    ax.axvspan(pair_length - 0.5, horizon - 0.5, color="#d62828", alpha=0.035)

    ymax = ax.get_ylim()[1]
    ax.text(pair_length / 2 - 0.5, ymax, "X", color="#ff8000", ha="center", va="bottom")
    ax.text(pair_length + pair_length / 2 - 0.5, ymax, "X2 / Y2", color="#d62828", ha="center", va="bottom")

    ax.set_title("X/X2 selection and Y2 forecast", fontsize="x-large", color="#ff8000")
    ax.set_xlabel("Time", color="#ff8000", fontsize="large")
    ax.set_ylabel("Demand", color="#ff8000", fontsize="large")
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
    n_components: int,
    special_labels: Sequence[str],
    include_declared_holidays: bool,
    include_outliers: bool,
    min_special_points: Optional[int],
    min_event_gap: Optional[int],
    max_events: Optional[int],
    label_column: str,
    dtw_window: Optional[float] = None,
) -> dict[str, object]:
    """Evaluate a single historical special date as a tuning fold."""
    target_ts = pd.Timestamp(target_date)
    train_df = history_df.loc[history_df["date"] < target_ts].copy()
    target_row = history_df.loc[history_df["date"] == target_ts]

    if train_df.empty or target_row.empty:
        raise ValueError(f"Invalid fold for {target_ts.date()}.")

    hourly_series = _flatten_daily_profiles(train_df)
    if len(hourly_series) < 2 * season_length + 1:
        raise ValueError(
            f"Insufficient history before {target_ts.date()} for season_length={season_length}."
        )

    special_mask = build_special_day_daily_mask(
        train_df,
        special_labels=special_labels,
        include_declared_holidays=include_declared_holidays,
        include_outliers=include_outliers,
        label_column=label_column,
    )
    if int(special_mask.sum()) == 0:
        raise ValueError(
            f"No prior special days exist before {target_ts.date()} to build analogs."
        )

    special_hourly_mask = np.repeat(
        special_mask.astype(float).to_numpy(),
        len(HOUR_COLS),
    )
    actual_profile = target_row.iloc[0][HOUR_COLS].to_numpy(dtype=np.float64)

    model = AnalogSpecialDays(
        season_length=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
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