"""Build the Parquet cache consumed by the holiday audit app."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_MODULE_DIR = Path(__file__).resolve().parent
PROJ_ROOT = _MODULE_DIR.parents[1]

if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from analog_holidays.shared.identify_holidays import (
    build_wide_df,
    cluster_atypical_profiles,
    compare_outliers_holidays,
    compute_distances,
    detect_outliers,
    get_date_sets,
    get_hour_cols,
    load_holidays_catalog,
    load_results_data,
)
from analog_holidays.shared.dataset_config import ACTIVE_CONFIG, list_dataset_regions

DEMAND_PATH = ACTIVE_CONFIG.demand_path
HOLIDAYS_PATH = ACTIVE_CONFIG.audit_holidays_path
OUT_DIR = PROJ_ROOT / "analog_holidays" / "audit" / "data"
CACHE_PATH = OUT_DIR / "audit_cache.parquet"

REGIONS = list_dataset_regions()

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_SEGMENTS = [{"label": m, "months": [i + 1]} for i, m in enumerate(_MONTH_NAMES)]
_MONTH_TO_SEGMENT = {m: s["label"] for s in _SEGMENTS for m in s["months"]}
_SEGMENT_LABELS = [s["label"] for s in _SEGMENTS]
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

OUTLIER_PERCENTILE = 97
DISTANCE_METRIC = "PEARSON"
N_CLUSTERS = 4
EXCLUDE_YEARS = [2022]
KNOWN_HOLIDAY_MIN_GROUP_PERCENTILE = 75


def _build_holiday_type_map(holidays_path: Path) -> dict[str, str]:
    """Map each holiday name to its date type."""
    with open(holidays_path, "r", encoding="utf-8") as file_obj:
        holidays_json = json.load(file_obj)

    return {
        holiday["name"]: holiday.get("date_type", "explicit")
        for holiday in holidays_json["holidays"]
        if "name" in holiday
    }


def _process_region(unique_id: str, holiday_type_map: dict) -> pd.DataFrame:
    print(f"  [{unique_id}] loading raw data...")
    df_raw = load_results_data(DEMAND_PATH, unique_id)
    if df_raw.empty:
        print(f"  [{unique_id}] no data, skipped")
        return pd.DataFrame()

    df_wide = build_wide_df(df_raw, _MONTH_TO_SEGMENT, exclude_years=EXCLUDE_YEARS)
    hour_cols = get_hour_cols(df_wide)

    year_min = int(df_wide.index.year.min())
    year_max = int(df_wide.index.year.max())

    df_holidays = load_holidays_catalog(HOLIDAYS_PATH, year_min, year_max)
    known_holiday_dates = set(pd.to_datetime(df_holidays["date"]).dt.normalize())

    print(f"  [{unique_id}] {len(df_wide)} days ({year_min}-{year_max}) -> computing distances...")

    df_dist = compute_distances(
        df_wide,
        hour_cols,
        _SEGMENT_LABELS,
        _WEEKDAY_NAMES,
        distance_metric=DISTANCE_METRIC,
        reference_exclude_dates=known_holiday_dates,
    )
    df_dist, df_outliers = detect_outliers(
        df_dist,
        OUTLIER_PERCENTILE,
        threshold_reference_only=True,
        promote_dates=known_holiday_dates,
        promote_min_group_percentile=KNOWN_HOLIDAY_MIN_GROUP_PERCENTILE,
    )

    df_match, _ = compare_outliers_holidays(df_outliers, df_holidays)
    all_dates = set(df_wide.index.normalize())
    date_sets = get_date_sets(df_match, df_holidays, all_dates)

    match_dates_set = date_sets["match_dates_set"]
    unknown_dates_set = date_sets["unknown_dates_set"]
    outlier_dates_set = date_sets["outlier_dates_set"]

    cluster_map: dict = {}
    try:
        print(f"  [{unique_id}] clustering {len(outlier_dates_set)} atypical days...")
        cluster_results = cluster_atypical_profiles(
            df_wide,
            match_dates_set,
            unknown_dates_set,
            outlier_dates_set,
            N_CLUSTERS,
            hour_cols,
            df_holidays,
        )
        df_atyp = cluster_results["df_atyp"]
        for dt, row in df_atyp.iterrows():
            cluster_map[pd.Timestamp(dt).normalize()] = int(row["cluster"])
    except Exception as exc:
        warnings.warn(f"[{unique_id}] clustering failed: {exc}")

    df_day = df_wide.copy()
    df_day.index = df_day.index.normalize()

    h_rename = {h: f"h_{h:02d}" for h in hour_cols}
    df_day = df_day.rename(columns=h_rename)
    h_cols = [f"h_{h:02d}" for h in range(24)]

    df_day["unique_id"] = unique_id
    df_day["date"] = df_day.index
    df_day["year"] = df_day.index.year
    df_day["dow_name"] = df_day["dow"].map(lambda d: _WEEKDAY_NAMES[int(d)])

    df_dist_merge = (
        df_dist[["date", "is_outlier", "distance"]]
        .copy()
        .rename(columns={"distance": "outlier_score"})
    )
    df_dist_merge["date"] = pd.to_datetime(df_dist_merge["date"]).dt.normalize()
    df_day = df_day.reset_index(drop=True).merge(df_dist_merge, on="date", how="left")

    df_holidays_merge = df_holidays.copy()
    df_holidays_merge["date"] = pd.to_datetime(df_holidays_merge["date"]).dt.normalize()
    df_day = df_day.merge(df_holidays_merge, on="date", how="left")
    df_day["is_declared_holiday"] = df_day["holiday_name"].notna()

    df_day["holiday_type"] = df_day["holiday_name"].map(holiday_type_map)

    df_day["cluster_id"] = df_day["date"].map(
        lambda d: cluster_map.get(pd.Timestamp(d).normalize(), np.nan)
    )

    df_day["event_description"] = None

    def _init_label(row: pd.Series) -> str:
        if row["is_declared_holiday"]:
            return "holiday"
        if row.get("is_outlier", False):
            return "special_day"
        return "normal_day"

    df_day["label"] = df_day.apply(_init_label, axis=1)
    df_day["outlier_method"] = DISTANCE_METRIC

    meta_cols = [
        "unique_id", "date", "year", "month", "dow", "dow_name", "segment",
        "holiday_name", "holiday_type", "is_declared_holiday",
        "is_outlier", "outlier_score", "outlier_method",
        "cluster_id", "event_description", "label",
    ]
    available_h = [c for c in h_cols if c in df_day.columns]
    result = df_day[meta_cols + available_h].copy()

    result["is_outlier"] = result["is_outlier"].fillna(False).astype(bool)
    result["is_declared_holiday"] = result["is_declared_holiday"].astype(bool)

    print(f"  [{unique_id}] {len(result)} days cached")
    return result


def _get_source_max_date(unique_id: str) -> pd.Timestamp | None:
    """Return the most recent non-null timestamp available for one series."""
    if not DEMAND_PATH.exists():
        return None

    df_src = pd.read_csv(
        DEMAND_PATH,
        usecols=["ds", unique_id],
        parse_dates=["ds"],
    )
    df_src = df_src.dropna(subset=[unique_id])
    if df_src.empty:
        return None

    return pd.Timestamp(df_src["ds"].max()).normalize()


def _build_future_rows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    holiday_type_map: dict,
) -> pd.DataFrame:
    """Build placeholder rows that extend the audit cache beyond observed data."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if dates.empty:
        return pd.DataFrame()

    year_min = int(dates.year.min())
    year_max = int(dates.year.max())
    df_holidays = load_holidays_catalog(HOLIDAYS_PATH, year_min, year_max)
    df_holidays = df_holidays.copy()
    df_holidays["date"] = pd.to_datetime(df_holidays["date"]).dt.normalize()

    rows = []
    for region in REGIONS:
        for dt in dates:
            dow_int = int(dt.dayofweek)
            month = int(dt.month)
            segment = _MONTH_TO_SEGMENT[month]

            holiday_row = df_holidays[df_holidays["date"] == dt]
            hol_name = holiday_row["holiday_name"].values[0] if not holiday_row.empty else None
            hol_type = holiday_type_map.get(hol_name) if hol_name else None
            is_holiday = hol_name is not None

            row = {
                "unique_id": region,
                "date": dt,
                "year": int(dt.year),
                "month": month,
                "dow": dow_int,
                "dow_name": _WEEKDAY_NAMES[dow_int],
                "segment": segment,
                "holiday_name": hol_name,
                "holiday_type": hol_type,
                "is_declared_holiday": is_holiday,
                "is_outlier": False,
                "outlier_score": np.nan,
                "outlier_method": DISTANCE_METRIC,
                "cluster_id": np.nan,
                "event_description": None,
                "label": "holiday" if is_holiday else "normal_day",
            }
            for h in range(24):
                row[f"h_{h:02d}"] = np.nan
            rows.append(row)

    df_future = pd.DataFrame(rows)
    print(
        f"  [future] {len(dates)} days × {len(REGIONS)} regions = "
        f"{len(df_future):,} skeleton rows built"
    )
    return df_future


def main(force_rebuild: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEMAND_PATH.exists():
        print(f"ERROR: demand source not found for dataset {ACTIVE_CONFIG.key!r}: {DEMAND_PATH}")
        return
    if not HOLIDAYS_PATH.exists():
        print(f"ERROR: holidays catalog not found for dataset {ACTIVE_CONFIG.key!r}: {HOLIDAYS_PATH}")
        return
    if not REGIONS:
        print(f"ERROR: no demand series found in {DEMAND_PATH}")
        return

    holiday_type_map = _build_holiday_type_map(HOLIDAYS_PATH)

    existing_by_region: dict[str, pd.DataFrame] = {}
    if force_rebuild:
        print("\nFull rebuild requested, ignoring existing cache...")
    elif CACHE_PATH.exists():
        df_existing = pd.read_parquet(CACHE_PATH)
        df_existing["date"] = pd.to_datetime(df_existing["date"])
        for region in REGIONS:
            real = df_existing[
                (df_existing["unique_id"] == region) & df_existing["h_00"].notna()
            ].copy()
            if not real.empty:
                existing_by_region[region] = real
        print(f"\nExisting cache loaded — {len(df_existing):,} rows total")
    else:
        print("\nNo existing cache found, running a full build...")

    print(f"Building audit cache for {len(REGIONS)} regions ({ACTIVE_CONFIG.key})...")
    frames = []
    for region in REGIONS:
        source_max = _get_source_max_date(region)
        cached_max = (
            existing_by_region[region]["date"].max()
            if region in existing_by_region
            else None
        )

        if source_max is not None and cached_max is not None and source_max <= cached_max:
            print(f"  [{region}] up to date ({cached_max.date()}), reusing cache")
            frames.append(existing_by_region[region])
            continue

        df = _process_region(region, holiday_type_map)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("ERROR: no data processed. Check DEMAND_PATH and region names.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    end_of_year = pd.Timestamp("2027-12-31")
    last_cached = combined["date"].max().normalize()
    if last_cached < end_of_year:
        next_day = last_cached + pd.Timedelta(days=1)
        print(f"\nAppending skeleton rows: {next_day.date()} -> {end_of_year.date()}...")
        df_future = _build_future_rows(next_day, end_of_year, holiday_type_map)
        if not df_future.empty:
            combined = pd.concat([combined, df_future], ignore_index=True)
    else:
        print("\nCache already covers through 2027-12-31, no skeleton rows needed.")

    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_parquet(CACHE_PATH, index=False)

    print(f"\nCache saved -> {CACHE_PATH}")
    print(f"   Rows: {len(combined):,}")
    print(f"   Regions: {combined['unique_id'].nunique()}")
    print(f"   Date range: {combined['date'].min().date()} -> {combined['date'].max().date()}")
    lbl_counts = combined["label"].value_counts()
    for lbl, cnt in lbl_counts.items():
        print(f"   {lbl:>15s}: {cnt:,} days")


def cli() -> None:
    """Run the audit cache build from the script entrypoint."""
    main()


if __name__ == "__main__":
    cli()
