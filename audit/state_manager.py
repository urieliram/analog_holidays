"""
state_manager.py
================
Persist label decisions and maintain an append-only audit log.

Public API
----------
update_label(df, region, date, new_label, user_id) → pd.DataFrame
save_labels(df)                                      → None
load_audit_log()                                     → pd.DataFrame
compute_flag_cols(df)                                → pd.DataFrame
export_wide(df)                                      → pd.DataFrame
export_nixtla_long(df)                               → pd.DataFrame
export_exogenous_db(df)                              → pd.DataFrame
export_hourly_wide(df)                               → pd.DataFrame
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_loader import HOUR_COLS, LABELS_PATH, REGIONS, VALID_LABELS

_AUDIT_LOG_PATH = LABELS_PATH.parent / "audit_log.csv"
_AUDIT_COLS     = ["timestamp", "region", "date", "previous_label", "new_label", "user_id"]


# ── Label update ──────────────────────────────────────────────────────────────
def update_label(
    df: pd.DataFrame,
    region: str,
    date,
    new_label: str,
    user_id: str = "user",
) -> pd.DataFrame:
    """
    Change the label for a (region, date) pair in the in-memory DataFrame.

    * Enforces mutual exclusivity: only one of holiday / special_day / normal_day.
    * Updates ALL rows that share the same (unique_id, date) — i.e. all 24 hours
      if the cache ever expands to hourly rows.
    * Writes a single row to the audit log when the label actually changes.
    * Returns the modified DataFrame (in-place mutation + return).
    """
    if new_label not in VALID_LABELS:
        raise ValueError(f"new_label must be one of {VALID_LABELS}, got: {new_label!r}")

    ts   = pd.Timestamp(date)
    mask = (df["unique_id"] == region) & (df["date"] == ts)

    if not mask.any():
        return df

    prev_label = str(df.loc[mask, "label"].iloc[0])

    if prev_label != new_label:
        df.loc[mask, "label"] = new_label
        _append_audit_row(region, ts, prev_label, new_label, user_id)

    return df


def _append_audit_row(
    region: str,
    date: pd.Timestamp,
    prev_label: str,
    new_label: str,
    user_id: str,
) -> None:
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "region":         region,
                "date":           date.date().isoformat(),
                "previous_label": prev_label,
                "new_label":      new_label,
                "user_id":        user_id,
            }
        ]
    )
    write_header = not _AUDIT_LOG_PATH.exists()
    row.to_csv(_AUDIT_LOG_PATH, mode="a", header=write_header, index=False)


# ── Persistence ───────────────────────────────────────────────────────────────
def save_labels(df: pd.DataFrame) -> None:
    """Persist the current label column to Parquet (overwrites previous save)."""
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["unique_id", "date", "label"]].copy().to_parquet(LABELS_PATH, index=False)


def load_audit_log() -> pd.DataFrame:
    """Return the full audit log. Empty DataFrame if no changes have been made yet."""
    if not _AUDIT_LOG_PATH.exists():
        return pd.DataFrame(columns=_AUDIT_COLS)
    return pd.read_csv(_AUDIT_LOG_PATH)


# ── Export helpers ────────────────────────────────────────────────────────────
def compute_flag_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary flag columns for each region:
        {region}_holidays    : 1 iff label == 'holiday'
        {region}_special_day : 1 iff label == 'special_day'
    Returns a copy of df with the new columns appended.
    """
    result = df.copy()
    for region in df["unique_id"].unique():
        mask = result["unique_id"] == region
        result.loc[mask, f"{region}_holidays"]    = (result.loc[mask, "label"] == "holiday").astype(int)
        result.loc[mask, f"{region}_special_day"] = (result.loc[mask, "label"] == "special_day").astype(int)
    return result


def export_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to wide format: one row per date, regions as columns.
    Returns a DataFrame suitable for downstream pipeline consumption.
    """
    flag_df = compute_flag_cols(df)
    records = []
    for date, grp in flag_df.groupby("date"):
        row: dict = {"date": date}
        for _, r in grp.iterrows():
            uid = r["unique_id"]
            row[f"{uid}_holidays"]    = int(r.get(f"{uid}_holidays",    0))
            row[f"{uid}_special_day"] = int(r.get(f"{uid}_special_day", 0))
            row[f"{uid}_label"]       = r["label"]
        records.append(row)
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def export_nixtla_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export in Nixtla long format: 2 binary series × 24 hours per (region, date).

    unique_id : {region}_holiday  |  {region}_special_day
    ds        : datetime with hour  (e.g. 2020-01-02 03:00:00)
    y         : 1 if the label matches that series, else 0
    """
    records: list[dict] = []
    for _, row in df.iterrows():
        region = row["unique_id"]
        date   = pd.Timestamp(row["date"])
        label  = row["label"]
        for h in range(24):
            ds = date + pd.Timedelta(hours=h)
            records.append({"unique_id": f"{region}_holiday",     "ds": ds, "y": int(label == "holiday")})
            records.append({"unique_id": f"{region}_special_day", "ds": ds, "y": int(label == "special_day")})
    result = pd.DataFrame(records)
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return result


def export_exogenous_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export in EXOGENOUS table bulk-load format.

    Columns:
        exogenous_name : str   — matches CAT_EXOGENOUS.exogenous_name
        ds             : str   — 'YYYY-MM-DD HH:MM'  (format expected by the loader)
        value          : int   — 0 or 1

    One row per (region × label_type × hour).
    Ready for UPSERT into EXOGENOUS after mapping exogenous_name → cat_exogenous_id.
    """
    long_df = export_nixtla_long(df)
    long_df = long_df.rename(columns={"unique_id": "exogenous_name", "y": "value"})
    long_df["ds"] = long_df["ds"].dt.strftime("%Y-%m-%d %H:%M")
    return long_df[["exogenous_name", "ds", "value"]]


def export_hourly_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export the audit cache in wide hourly format.

    Output columns:
        ds,
        <unique_id_1>, <unique_id_1>_holiday,
        <unique_id_2>, <unique_id_2>_holiday, ...

    Where:
        - ds is the hourly timestamp
        - <unique_id> is the hourly demand value from the audit cache
        - <unique_id>_holiday is a 0/1 flag derived from the daily label
    """
    base_df = df[["unique_id", "date", "label", *HOUR_COLS]].copy()
    long_df = base_df.melt(
        id_vars=["unique_id", "date", "label"],
        value_vars=HOUR_COLS,
        var_name="hour_col",
        value_name="y",
    )
    long_df["hour"] = long_df["hour_col"].str.removeprefix("h_").astype(int)
    long_df["ds"] = pd.to_datetime(long_df["date"]) + pd.to_timedelta(long_df["hour"], unit="h")
    long_df["holiday_flag"] = (long_df["label"] == "holiday").astype(int)

    value_wide = long_df.pivot(index="ds", columns="unique_id", values="y")
    holiday_wide = long_df.pivot(index="ds", columns="unique_id", values="holiday_flag")
    holiday_wide = holiday_wide.rename(columns={col: f"{col}_holiday" for col in holiday_wide.columns})

    result = pd.concat([value_wide, holiday_wide], axis=1).reset_index().sort_values("ds")

    present_regions = [region for region in REGIONS if region in value_wide.columns]
    extra_regions = sorted(col for col in value_wide.columns if col not in REGIONS)
    ordered_regions = present_regions + extra_regions

    ordered_cols = ["ds"]
    for region in ordered_regions:
        ordered_cols.append(region)
        ordered_cols.append(f"{region}_holiday")

    result = result[ordered_cols]
    result["ds"] = pd.to_datetime(result["ds"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    for region in ordered_regions:
        result[f"{region}_holiday"] = result[f"{region}_holiday"].fillna(0).astype(int)

    return result
