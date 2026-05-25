"""Conditional / chained analog forecasting around holidays.

Standard ``run_analog_holidays`` produces an independent 24-hour forecast for a
single target date. In real operations the demand profile of a holiday day is
strongly correlated with the days that surround it (eves, the holiday itself,
post-holidays, the recovery day). This module chains forecasts so that:

    H1  → forecast using the actual day before H1 as Y context
    H2  → forecast using the H1 forecast as Y context
    H3  → forecast using the H2 forecast as Y context
    H4  → forecast using the H3 forecast as Y context

At each stage, the historical analogs are restricted to the matching day-type so
that the regression maps the *correct* preceding profile shape into the *correct*
next-day profile shape.

Stage to historical mask mapping
--------------------------------
For every stage S, the analog algorithm looks for windows where the *future*
24-hour block in history is flagged as a historical day of type S. The 24 hours
*preceding* such a window are then matched against Y. So:

* H1: mask historical H1 days → X = day before historical H1 (pre-eve, normal)
* H2: mask historical H2 days → X = day before historical H2 (= historical H1
      when the holiday is preceded by an eve, or a normal day otherwise)
* H3: mask historical H3 days → X = day before historical H3 (= historical H2)
* H4: mask historical H4 days → X = day before historical H4 (= historical H3)

The conditional chain therefore turns the analog model into a Markov-style
walk through the holiday block: each step's analog X-profile is the previous
step's forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analog_holidays.audit.data_loader import HOUR_COLS

from .analog_holidays import (
    DEFAULT_LEVELS,
    DEFAULT_SOURCE_PATH,
    _complete_day_mask,
    _flatten_daily_profiles,
    _get_region_df,
    build_selected_days_table,
    build_special_day_daily_mask,
    load_audit_source,
)
from .analog_special_days import AnalogSpecialDays, analog_special_days_core


HOLIDAY_STAGES: tuple[str, ...] = ("H1", "H2", "H3", "H4")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HolidayStageRun:
    """Forecast outputs for a single stage of the chain."""

    stage: str
    target_date: pd.Timestamp
    context_source: str  # "actual" or "H1_forecast" / "H2_forecast" / ...
    previous_day_profile: np.ndarray  # the Y context used (real or appended forecast)
    forecast_profile: np.ndarray
    interval_low: Dict[int, np.ndarray]
    interval_high: Dict[int, np.ndarray]
    actual_profile: Optional[np.ndarray]
    positions: List[int]
    neighbors2: np.ndarray
    selected_days_df: pd.DataFrame
    special_day_daily_mask: pd.Series
    extended_series_len: int
    fail: bool
    t_sel: float
    t_reg: float


@dataclass
class ConditionalChainResult:
    """All stage forecasts for one chain plus metadata."""

    unique_id: str
    chain_dates: Dict[str, pd.Timestamp]
    runs: Dict[str, HolidayStageRun]
    classified_df: pd.DataFrame
    season_length: int
    k: Optional[int]
    typedist: str
    typereg: str
    n_components: int
    levels: List[int]


# ---------------------------------------------------------------------------
# Historical classification (H1 / H2 / H3 / H4 from raw "holiday" label)
# ---------------------------------------------------------------------------


def classify_historical_holidays(
    df_region: pd.DataFrame,
    special_labels: Sequence[str] = ("holiday",),
    label_column: str = "label",
) -> pd.DataFrame:
    """Augment ``df_region`` with a ``holiday_class`` column.

    Uses the adjacency rule documented in the ``Holiday Expert`` agent:

    * H1: normal day immediately before a holiday (prev is normal, next is holiday)
    * H2: holiday whose previous day is NOT a holiday
    * H3: holiday whose previous day IS a holiday
    * H4: normal day after a run of 2+ consecutive holidays

    Anything else gets ``None``.
    """
    df_region = df_region.sort_values("date").reset_index(drop=True)
    is_holiday = (
        df_region[label_column].fillna("").isin(tuple(special_labels)).to_numpy()
    )
    prev_h = np.concatenate([[False], is_holiday[:-1]])
    next_h = np.concatenate([is_holiday[1:], [False]])
    prev_prev_h = np.concatenate([[False, False], is_holiday[:-2]])

    klass = np.array([None] * len(df_region), dtype=object)
    klass[(~is_holiday) & next_h & (~prev_h)] = "H1"
    klass[is_holiday & (~prev_h)] = "H2"
    klass[is_holiday & prev_h] = "H3"
    klass[(~is_holiday) & prev_h & prev_prev_h] = "H4"

    out = df_region.copy()
    out["holiday_class"] = klass
    return out


def build_stage_daily_mask(classified_df: pd.DataFrame, stage: str) -> pd.Series:
    """Return a boolean per-day Series flagging only ``stage`` rows (H1/H2/H3/H4)."""
    if stage not in HOLIDAY_STAGES:
        raise ValueError(f"stage must be one of {HOLIDAY_STAGES}, got {stage!r}")
    return (classified_df["holiday_class"] == stage).astype(bool)


# ---------------------------------------------------------------------------
# Single-stage runner (internal helper)
# ---------------------------------------------------------------------------


def _run_single_stage(
    base_train_df: pd.DataFrame,
    classified_train_df: pd.DataFrame,
    stage: str,
    target_date: pd.Timestamp,
    target_row: Optional[pd.Series],
    previous_context: np.ndarray,
    context_source: str,
    season_length: int,
    k: Optional[int],
    typedist: str,
    typereg: str,
    n_components: int,
    levels: List[int],
    min_special_points: Optional[int],
    min_event_gap: Optional[int],
    max_events: Optional[int],
) -> HolidayStageRun:
    """Execute one chain step.

    The hourly series is ``flatten(base_train_df) + previous_context``. The
    mask only flags historical days of the requested ``stage``. Y (last 24 h
    of the series) is therefore ``previous_context`` — either the real day
    before H1 or the forecast emitted by the previous stage.
    """
    if previous_context.shape != (season_length,):
        raise ValueError(
            f"previous_context must have shape ({season_length},), "
            f"got {previous_context.shape}"
        )

    base_series = _flatten_daily_profiles(base_train_df)
    extended_series = np.concatenate(
        [base_series, previous_context.astype(np.float64)]
    )

    stage_mask = build_stage_daily_mask(classified_train_df, stage)
    base_hourly_mask = np.repeat(
        stage_mask.astype(float).to_numpy(), len(HOUR_COLS)
    )
    # The appended context block is the Y target — it must NOT be flagged as
    # special (otherwise the algorithm would treat it as a candidate event).
    appended_mask = np.zeros(len(HOUR_COLS), dtype=float)
    extended_mask = np.concatenate([base_hourly_mask, appended_mask])

    if len(extended_series) < 2 * season_length + 1:
        raise ValueError(
            "Extended history too short for AnalogSpecialDays. "
            f"Need at least {2 * season_length + 1} hourly points, "
            f"got {len(extended_series)}."
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
    model.fit(y=extended_series, special_days=extended_mask)
    result = model.predict(h=season_length, level=levels)

    forecast_profile = result["mean"]
    interval_low = {
        lv: result[f"lo-{lv}"] for lv in levels if f"lo-{lv}" in result
    }
    interval_high = {
        lv: result[f"hi-{lv}"] for lv in levels if f"hi-{lv}" in result
    }

    _, t_sel, t_reg, fail, positions, neighbors2 = analog_special_days_core(
        serie=extended_series,
        special_days=extended_mask,
        vsele=season_length,
        k=k,
        typedist=typedist,
        n_components=n_components,
        typereg=typereg,
        min_special_points=min_special_points,
        min_event_gap=min_event_gap,
        max_events=max_events,
    )

    # Map selected positions back to historical day metadata. The appended
    # context block has no entry in classified_train_df, so any positions
    # pointing into it would be invalid — but the mask there is zero so no
    # selection can land there.
    try:
        selected_days_df = build_selected_days_table(
            train_df=classified_train_df,
            positions=positions,
            neighbors2=neighbors2,
            season_length=season_length,
        )
    except Exception:  # pragma: no cover - defensive
        selected_days_df = pd.DataFrame()

    actual_profile: Optional[np.ndarray] = None
    if target_row is not None:
        candidate_actual = target_row[HOUR_COLS].to_numpy(dtype=np.float64)
        if np.isfinite(candidate_actual).all():
            actual_profile = candidate_actual

    return HolidayStageRun(
        stage=stage,
        target_date=target_date,
        context_source=context_source,
        previous_day_profile=previous_context.astype(np.float64),
        forecast_profile=forecast_profile,
        interval_low=interval_low,
        interval_high=interval_high,
        actual_profile=actual_profile,
        positions=list(positions),
        neighbors2=np.asarray(neighbors2),
        selected_days_df=selected_days_df,
        special_day_daily_mask=stage_mask,
        extended_series_len=len(extended_series),
        fail=fail,
        t_sel=t_sel,
        t_reg=t_reg,
    )


# ---------------------------------------------------------------------------
# Public API: chain runner
# ---------------------------------------------------------------------------


def run_analog_holiday_chain(
    unique_id: str,
    chain_dates: Dict[str, str | pd.Timestamp],
    source_path: Path | str = DEFAULT_SOURCE_PATH,
    season_length: int = 24,
    k: Optional[int] = None,
    typedist: str = "pearson",
    typereg: str = "PCR",
    n_components: int = 3,
    levels: Optional[Sequence[int]] = None,
    special_labels: Sequence[str] = ("holiday",),
    min_special_points: Optional[int] = None,
    min_event_gap: Optional[int] = None,
    max_events: Optional[int] = None,
    label_column: str = "label",
) -> ConditionalChainResult:
    """Run the chained H1 → H2 → H3 → H4 forecast.

    Parameters
    ----------
    chain_dates : dict
        Mapping stage name → target date, e.g.
        ``{"H1": "2025-12-31", "H2": "2026-01-01"}``. Stages run in the order
        ``H1, H2, H3, H4``; missing stages are skipped, and the next stage will
        use the most recent forecast as its context (or the actual day-before
        profile if no prior forecast exists).
    """
    levels_list = list(DEFAULT_LEVELS if levels is None else levels)
    stages_requested = [s for s in HOLIDAY_STAGES if s in chain_dates]
    if not stages_requested:
        raise ValueError(
            "chain_dates must contain at least one of "
            f"{HOLIDAY_STAGES}, got keys {list(chain_dates)}"
        )

    chain_dates_ts: Dict[str, pd.Timestamp] = {
        s: pd.Timestamp(chain_dates[s]) for s in stages_requested
    }

    # ── Load and classify the full region history ────────────────────────────
    df = load_audit_source(source_path)
    df_region = _get_region_df(df, unique_id)
    classified_full = classify_historical_holidays(
        df_region,
        special_labels=special_labels,
        label_column=label_column,
    )

    first_target = chain_dates_ts[stages_requested[0]]
    complete_mask = _complete_day_mask(classified_full)
    base_train_df = classified_full[
        (classified_full["date"] < first_target) & complete_mask
    ].copy()
    if base_train_df.empty:
        raise ValueError(
            f"No complete history found for {unique_id} before {first_target.date()}"
        )

    # The day immediately before the first target is the bootstrap Y context.
    bootstrap_context = base_train_df.iloc[-1][HOUR_COLS].to_numpy(dtype=np.float64)
    bootstrap_source = (
        f"actual_{base_train_df.iloc[-1]['date'].strftime('%Y-%m-%d')}"
    )

    # ── Iterate through stages, chaining forecasts ───────────────────────────
    runs: Dict[str, HolidayStageRun] = {}
    previous_context = bootstrap_context
    previous_source = bootstrap_source

    for stage in stages_requested:
        target_ts = chain_dates_ts[stage]
        target_rows = classified_full[classified_full["date"] == target_ts]
        target_row = target_rows.iloc[0] if not target_rows.empty else None

        stage_run = _run_single_stage(
            base_train_df=base_train_df,
            classified_train_df=base_train_df,
            stage=stage,
            target_date=target_ts,
            target_row=target_row,
            previous_context=previous_context,
            context_source=previous_source,
            season_length=season_length,
            k=k,
            typedist=typedist,
            typereg=typereg,
            n_components=n_components,
            levels=levels_list,
            min_special_points=min_special_points,
            min_event_gap=min_event_gap,
            max_events=max_events,
        )
        runs[stage] = stage_run

        # Set up context for the next stage
        previous_context = stage_run.forecast_profile.astype(np.float64)
        previous_source = f"{stage}_forecast"

    return ConditionalChainResult(
        unique_id=unique_id,
        chain_dates=chain_dates_ts,
        runs=runs,
        classified_df=classified_full,
        season_length=season_length,
        k=k,
        typedist=typedist,
        typereg=typereg,
        n_components=n_components,
        levels=levels_list,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def build_chain_summary(result: ConditionalChainResult) -> pd.DataFrame:
    """Compact summary of every stage of a conditional chain."""
    rows: List[dict] = []
    for stage, run in result.runs.items():
        mape = np.nan
        if run.actual_profile is not None and np.all(np.abs(run.actual_profile) > 1e-9):
            mape = float(
                np.mean(
                    np.abs(
                        (run.actual_profile - run.forecast_profile)
                        / run.actual_profile
                    )
                )
                * 100.0
            )
        rows.append(
            {
                "stage": stage,
                "target_date": run.target_date.date(),
                "context_source": run.context_source,
                "n_analogs": len(run.positions),
                "extended_series_len": run.extended_series_len,
                "forecast_max": float(np.max(run.forecast_profile)),
                "forecast_mean": float(np.mean(run.forecast_profile)),
                "actual_available": run.actual_profile is not None,
                "mape_24h_pct": mape,
                "t_sel_s": run.t_sel,
                "t_reg_s": run.t_reg,
                "fail": run.fail,
            }
        )
    return pd.DataFrame(rows)


def plot_chain(
    result: ConditionalChainResult,
    figsize: Optional[tuple] = None,
):
    """Plot all stages of a chain as a horizontal grid (one subplot per stage)."""
    import matplotlib.pyplot as plt

    stages = list(result.runs.keys())
    n = len(stages)
    fig, axes = plt.subplots(
        1, n, figsize=figsize or (4.5 * n, 3.5), sharey=True, squeeze=False
    )
    axes = axes[0]

    widest = max(result.levels)
    for ax, stage in zip(axes, stages):
        run = result.runs[stage]
        hours = np.arange(1, run.forecast_profile.size + 1)

        # Y context (previous day used)
        ax.plot(
            hours,
            run.previous_day_profile,
            color="tab:gray",
            linestyle="--",
            alpha=0.7,
            label=f"Y context ({run.context_source})",
        )
        # Forecast + interval
        ax.fill_between(
            hours,
            run.interval_low.get(widest, run.forecast_profile),
            run.interval_high.get(widest, run.forecast_profile),
            alpha=0.18,
            color="tab:red",
            label=f"{widest}% PI",
        )
        ax.plot(
            hours,
            run.forecast_profile,
            color="tab:red",
            linewidth=2.0,
            label=f"{stage} forecast",
        )
        # Actual if available
        if run.actual_profile is not None:
            ax.plot(
                hours,
                run.actual_profile,
                color="tab:blue",
                linewidth=1.5,
                label="actual",
            )

        ax.set_title(
            f"{stage} | {run.target_date.date()} | n_analogs={len(run.positions)}"
        )
        ax.set_xlabel("hour")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("demand")
    fig.suptitle(
        f"Conditional chain | {result.unique_id} | "
        f"{result.typereg} | {result.typedist} | k={result.k}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes
