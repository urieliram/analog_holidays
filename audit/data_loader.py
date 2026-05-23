"""
data_loader.py
==============
Load and merge the precomputed audit cache with any persisted user labels.

Public API
----------
load_audit_df()          → pd.DataFrame   (complete merged DataFrame)
get_region_dates(df, r)  → list[Timestamp]
get_day_row(df, r, d)    → pd.Series | None
CACHE_PATH               (Path)
LABELS_PATH              (Path)
HOUR_COLS                (list[str])
REGIONS                  (list[str])
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from analog_holidays.shared.dataset_config import list_dataset_regions

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_ROOT   = Path(__file__).resolve().parents[2]
_DATA_DIR   = PROJ_ROOT / "analog_holidays" / "audit" / "data"
CACHE_PATH  = _DATA_DIR / "audit_cache.parquet"
LABELS_PATH = _DATA_DIR / "audit_labels.parquet"

HOUR_COLS = [f"h_{h:02d}" for h in range(24)]

REGIONS = list_dataset_regions()

VALID_LABELS = {"holiday", "special_day", "normal_day"}


# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Audit cache not found at:\n  {CACHE_PATH}\n\n"
            "Run the precompute script first:\n"
            "  python audit/build_cache.py"
        )
    df = pd.read_parquet(CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_persisted_labels() -> pd.DataFrame | None:
    """Return previously saved labels, or None if no file exists yet."""
    if not LABELS_PATH.exists():
        return None
    df = pd.read_parquet(LABELS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _merge_labels(df_cache: pd.DataFrame, df_labels: pd.DataFrame | None) -> pd.DataFrame:
    """Override computed default labels with any persisted user edits."""
    if df_labels is None:
        return df_cache.copy()

    lookup = df_labels.set_index(["unique_id", "date"])["label"].to_dict()

    df = df_cache.copy()
    df["label"] = df.apply(
        lambda row: lookup.get((row["unique_id"], row["date"]), row["label"]),
        axis=1,
    )
    return df


# ── Public entry point ────────────────────────────────────────────────────────
def load_audit_df() -> pd.DataFrame:
    """
    Load the precomputed cache and overlay any persisted user labels.
    Returned DataFrame is the single source of truth for the Streamlit app.
    """
    df_cache  = _load_cache()
    df_labels = _load_persisted_labels()
    return _merge_labels(df_cache, df_labels)


# ── Navigation helpers ────────────────────────────────────────────────────────
def get_region_dates(df: pd.DataFrame, region: str) -> list[pd.Timestamp]:
    """Sorted list of dates available for a given region."""
    return sorted(df.loc[df["unique_id"] == region, "date"].unique())


def get_day_row(df: pd.DataFrame, region: str, date) -> pd.Series | None:
    """Return the single row for (region, date). None if not found."""
    ts   = pd.Timestamp(date)
    mask = (df["unique_id"] == region) & (df["date"] == ts)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else None
