"""Diagnostic: was a labeled holiday actually observed as inhábil?

Read-only analysis (step 1 of the "substituted-holiday" hypothesis). For each
(unique_id, target_date) it compares the day's real 24-h demand against:

  * a *working baseline* -- the median of same-weekday NON-holiday days within
    +/-`WORK_WINDOW_DAYS` of the target (controls for weekday shape & season), and
  * a *holiday baseline*  -- the median of the SAME `anchor_holiday_name` +
    `holiday_day_type` in OTHER years (the typically-observed holiday profile).

From these it derives, per region/date:

  actual_drop_pct  = (work_mean - actual_mean) / work_mean * 100   # how inhábil the day really was
  typical_drop_pct = (work_mean - hol_mean)    / work_mean * 100   # how inhábil this holiday usually is
  observed_strength = actual_drop_pct / typical_drop_pct           # ~1 normal, ~0 looks like a workday, >1 deeper

`observed_strength ~ 0` is the signal the user hypothesized: the day was barely a
holiday (substituted / puente moved the rest day elsewhere). It then merges the
model error from experiments/*/metrics.csv to test whether weak observance lines
up with the model's worst (under-predicted) dates.

Source files are treated as read-only. Nothing here writes to holiday_demand_mx.csv.

Run:  python -m analog_holidays.shared.observed_strength
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[1]
_REPO = _PROJ_ROOT  # holidays/ and experiments/ live here
DEMAND_PATH = _REPO / "holidays" / "holiday_demand_mx.csv"
FEATURES_PATH = _REPO / "holidays" / "holiday_selector_features.csv"
EXPERIMENTS_GLOB = str(_REPO / "experiments" / "experiment_*" / "metrics.csv")
OUTPUT_PATH = _REPO / "holidays" / "working" / "observed_strength_diagnostic.csv"

WORK_WINDOW_DAYS = 28  # +/- window around the target for the working baseline
MIN_WORK_DAYS = 2      # need at least this many same-weekday workdays to trust the baseline
REGIONS = [
    "SEN_demand_CEL", "SEN_demand_NES", "SEN_demand_NOR", "SEN_demand_NTE",
    "SEN_demand_OCC", "SEN_demand_ORI", "SEN_demand_PEN", "SEN_demand_SIN",
]


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(DEMAND_PATH, parse_dates=["ds"])
    df["date"] = df["ds"].dt.normalize()
    df["hour"] = df["ds"].dt.hour
    return df


def load_features() -> pd.DataFrame:
    f = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    return f[["unique_id", "date", "anchor_holiday_name", "holiday_day_type"]].drop_duplicates()


def day_vector(df: pd.DataFrame, region: str, day: pd.Timestamp) -> np.ndarray | None:
    """24-h demand vector (hours 0..23) for a region/day, or None if incomplete."""
    sub = df.loc[df["date"] == day, ["hour", region]].dropna()
    if len(sub) < 24:
        return None
    vec = sub.sort_values("hour")[region].to_numpy()[:24]
    return vec if np.isfinite(vec).all() else None


def working_baseline(df: pd.DataFrame, region: str, day: pd.Timestamp) -> tuple[np.ndarray | None, int]:
    """Median 24-h profile of same-weekday, non-holiday days near `day`."""
    flag = f"{region}_holiday"
    lo, hi = day - pd.Timedelta(days=WORK_WINDOW_DAYS), day + pd.Timedelta(days=WORK_WINDOW_DAYS)
    win = df[(df["ds"] >= lo) & (df["ds"] < hi + pd.Timedelta(days=1))]
    # same weekday, not the target, not flagged holiday
    win = win[(win["ds"].dt.weekday == day.weekday()) & (win["date"] != day) & (win[flag] == 0)]
    vecs = [v for d in win["date"].unique() if (v := day_vector(df, region, d)) is not None]
    if len(vecs) < MIN_WORK_DAYS:
        return None, len(vecs)
    return np.median(np.vstack(vecs), axis=0), len(vecs)


def holiday_baseline(df, feats, region, anchor, day_type, target_year) -> tuple[np.ndarray | None, int]:
    """Median 24-h profile of the same anchor+type in OTHER years, same region."""
    peers = feats[(feats["unique_id"] == region)
                  & (feats["anchor_holiday_name"] == anchor)
                  & (feats["holiday_day_type"] == day_type)
                  & (feats["date"].dt.year != target_year)]
    vecs = [v for d in peers["date"] if (v := day_vector(df, region, pd.Timestamp(d).normalize())) is not None]
    if not vecs:
        return None, 0
    return np.median(np.vstack(vecs), axis=0), len(vecs)


def load_model_error() -> pd.DataFrame:
    files = glob.glob(EXPERIMENTS_GLOB)
    if not files:
        return pd.DataFrame(columns=["unique_id", "target_date", "model_mape24_med", "model_mpe24_med"])
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = (m.groupby(["unique_id", "target_date"])
           .agg(model_mape24_med=("mape_24_pct", "median"),
                model_mpe24_med=("mpe_24_pct", "median"))
           .reset_index())
    g["target_date"] = pd.to_datetime(g["target_date"]).dt.normalize()
    return g


def _safe_div(a, b):
    return np.nan if (b is None or abs(b) < 1e-9) else a / b


def build() -> pd.DataFrame:
    demand = load_demand()
    feats = load_features()
    err = load_model_error()

    # target dates = the dates the experiments actually evaluated
    targets = err[["unique_id", "target_date"]].drop_duplicates() if len(err) else None
    if targets is None or targets.empty:
        # fall back: every flagged anchor in the features within demand coverage
        targets = feats[["unique_id", "date"]].rename(columns={"date": "target_date"})

    rows = []
    for _, t in targets.iterrows():
        region, day = t["unique_id"], pd.Timestamp(t["target_date"]).normalize()
        if region not in REGIONS:
            continue
        actual = day_vector(demand, region, day)
        if actual is None:
            continue  # future / incomplete -> can't diagnose
        meta = feats[(feats["unique_id"] == region) & (feats["date"] == day)]
        anchor = meta["anchor_holiday_name"].iloc[0] if len(meta) else "?"
        dtype = meta["holiday_day_type"].iloc[0] if len(meta) else "?"
        work, n_work = working_baseline(demand, region, day)
        hol, n_hol = holiday_baseline(demand, feats, region, anchor, dtype, day.year)

        a_mean = float(np.mean(actual))
        w_mean = float(np.mean(work)) if work is not None else np.nan
        h_mean = float(np.mean(hol)) if hol is not None else np.nan
        actual_drop = _safe_div(w_mean - a_mean, w_mean) * 100 if work is not None else np.nan
        typical_drop = _safe_div(w_mean - h_mean, w_mean) * 100 if (work is not None and hol is not None) else np.nan
        strength = _safe_div(actual_drop, typical_drop) if np.isfinite(typical_drop) else np.nan
        # shape closeness: which baseline does the real day resemble more?
        corr_work = float(np.corrcoef(actual, work)[0, 1]) if work is not None else np.nan
        corr_hol = float(np.corrcoef(actual, hol)[0, 1]) if hol is not None else np.nan

        rows.append(dict(
            unique_id=region, target_date=day.date().isoformat(), anchor=anchor, day_type=dtype,
            weekday=day.day_name(), actual_mean=round(a_mean, 1), work_mean=round(w_mean, 1),
            hol_mean=round(h_mean, 1), actual_drop_pct=round(actual_drop, 2),
            typical_drop_pct=round(typical_drop, 2),
            observed_strength=round(strength, 3) if np.isfinite(strength) else np.nan,
            corr_work=round(corr_work, 3), corr_hol=round(corr_hol, 3),
            n_work_days=n_work, n_hol_years=n_hol,
        ))
    out = pd.DataFrame(rows)
    if len(err):
        out = out.merge(err.assign(target_date=err["target_date"].dt.date.astype(str)),
                        on=["unique_id", "target_date"], how="left")
    return out


def main() -> None:
    out = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    print(f"observed_strength diagnostic  (rows={len(out)})  -> {OUTPUT_PATH}")
    if out.empty:
        print("No diagnosable dates (no observed demand for evaluated targets).")
        return

    # Per-date summary: median observance + model error, weakest observance first
    per_date = (out.groupby(["target_date", "anchor", "day_type", "weekday"])
                  .agg(strength_med=("observed_strength", "median"),
                       actual_drop_med=("actual_drop_pct", "median"),
                       typical_drop_med=("typical_drop_pct", "median"),
                       model_mape24=("model_mape24_med", "median"),
                       model_mpe24=("model_mpe24_med", "median"))
                  .reset_index().sort_values("strength_med"))
    print("\n=== Per-date observance (weakest first) ===")
    print(per_date.round(2).to_string(index=False))

    # Hypothesis test: does weak observance track model error?
    d = out.dropna(subset=["observed_strength", "model_mape24_med"])
    if len(d) > 3:
        print("\n=== Hypothesis check (per region x date) ===")
        print(f"corr(observed_strength, model_mape24) = {d['observed_strength'].corr(d['model_mape24_med']):+.3f}")
        print(f"corr(observed_strength, model_mpe24)  = {d['observed_strength'].corr(d['model_mpe24_med']):+.3f}"
              "   (+ => weaker observance pairs with model under-prediction)")
        weak = d[d["observed_strength"] < 0.5]
        print(f"\nWeakly-observed cases (strength<0.5): n={len(weak)}  "
              f"median model_mape24={weak['model_mape24_med'].median():.2f}%  "
              f"median model_mpe24={weak['model_mpe24_med'].median():+.2f}%")
        strong = d[d["observed_strength"] >= 0.5]
        print(f"Normally-observed (strength>=0.5):   n={len(strong)}  "
              f"median model_mape24={strong['model_mape24_med'].median():.2f}%  "
              f"median model_mpe24={strong['model_mpe24_med'].median():+.2f}%")


if __name__ == "__main__":
    main()
