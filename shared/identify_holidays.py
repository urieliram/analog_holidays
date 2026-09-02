"""Exploratory holiday-detection helpers for M_identify_HOLIDAYS.ipynb.

This module contains the notebook-side logic for atypical-day detection and
plot support. It is not part of the production pipeline.
"""

import json
import math
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter as calc_easter
from scipy.spatial.distance import cdist
from scipy.stats import binomtest, wilcoxon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Data loading

# The 2006 Federal Labor Law reform moved three civic holidays to observed
# Mondays instead of fixed dates.
_LFT_REFORM_YEAR = 2006

# Maps an observed-rule value to `(month, nth_monday)`.
_NTH_MONDAY_RULES: dict[str, tuple[int, int]] = {
    'first_monday_of_february': (2, 1),
    'third_monday_of_march': (3, 3),
    'third_monday_of_november': (11, 3),
}


# Weekday names as they appear in the holiday catalogs, Monday=0 to match
# ``pd.Timestamp.dayofweek``.
_WEEKDAY_NAME_TO_INDEX: dict[str, int] = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6,
}


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return the nth weekday of the given month; n=-1 means last occurrence."""
    first = pd.Timestamp(year=year, month=month, day=1)
    if n == -1:
        last = first + pd.offsets.MonthEnd(0)
        offset = (last.dayofweek - weekday) % 7
        return (last - timedelta(days=offset)).normalize()

    if n < 1:
        raise ValueError(f"week_of_month must be >= 1 or -1, got {n}.")

    offset = (weekday - first.dayofweek) % 7
    resolved = first + timedelta(days=offset + (n - 1) * 7)
    if resolved.month != first.month:
        raise ValueError(
            f"{year}-{month:02d} has no {n}th weekday {weekday}; "
            "the rule would resolve into the following month."
        )
    return resolved.normalize()


def _nth_monday_of_month(year: int, month: int, n: int) -> pd.Timestamp:
    """Return the nth Monday of the given month."""
    return _nth_weekday_of_month(year, month, _WEEKDAY_NAME_TO_INDEX['monday'], n)


def _holiday_display_name(holiday: dict) -> str:
    """Return the preferred display name from a holiday catalog row."""
    return (
        holiday.get('name')
        or holiday.get('name_en')
        or holiday.get('name_es')
        or holiday.get('id')
        or 'Unnamed holiday'
    )


def _resolve_observed_holiday_date(year: int, holiday: dict) -> pd.Timestamp | None:
    """Resolve the observed date for a civic holiday with a movable rule."""
    observed_rule = holiday.get('observed_rule') or holiday.get('rule', '')
    nth_rule = _NTH_MONDAY_RULES.get(observed_rule)
    if nth_rule is None:
        return None

    if year >= _LFT_REFORM_YEAR:
        month_r, n_r = nth_rule
        return _nth_monday_of_month(year, month_r, n_r)

    month = holiday.get('month')
    day = holiday.get('day')
    if month is None or day is None:
        return None

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def load_results_data(source_path: Path, unique_id: str) -> pd.DataFrame:
    """Load observed rows for one series from SQLite or a wide CSV."""
    source_path = Path(source_path)

    if source_path.suffix.lower() == '.csv':
        df_csv = pd.read_csv(source_path, parse_dates=['ds'])
        df_csv = df_csv.loc[:, ~df_csv.columns.astype(str).str.startswith('Unnamed:')]

        if 'ds' not in df_csv.columns:
            raise ValueError(f'CSV file {source_path} must contain a ds column.')
        if unique_id not in df_csv.columns:
            raise ValueError(
                f'Series {unique_id!r} does not exist in {source_path}. '
                f'Available columns: {list(df_csv.columns)}'
            )

        return (
            df_csv[['ds', unique_id]]
            .rename(columns={unique_id: 'y'})
            .dropna(subset=['ds', 'y'])
            .sort_values('ds')
            .reset_index(drop=True)
        )

    with sqlite3.connect(str(source_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT r.ds, r.y
            FROM RESULTS r
            JOIN TIMESERIES t ON r.uid = t.uid
            WHERE r.model = 0
              AND t.unique_id = ?
            ORDER BY r.ds
            """,
            conn,
            params=(unique_id,),
            parse_dates=['ds'],
        )
    return df


def load_holidays_catalog(
    holidays_path: Path,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """Build concrete holiday dates for the requested year range."""
    with open(holidays_path, 'r', encoding='utf-8') as f:
        holidays_json = json.load(f)

    simple_holidays = holidays_json.get('holidays', [])
    if simple_holidays and all('date' in holiday for holiday in simple_holidays):
        rows = []
        for holiday in simple_holidays:
            dt = pd.Timestamp(holiday['date']).normalize()
            if year_min <= dt.year <= year_max:
                rows.append({
                    'date': dt,
                    'holiday_name': _holiday_display_name(holiday),
                })

        return (
            pd.DataFrame(rows, columns=['date', 'holiday_name'])
            .sort_values('date')
            .reset_index(drop=True)
        )

    years = range(year_min, year_max + 1)
    holiday_dates: dict = {}

    resolved_holiday_dates_by_id: dict[str, dict[int, pd.Timestamp]] = {}

    for holiday in holidays_json['holidays']:
        name = _holiday_display_name(holiday)
        holiday_id = holiday.get('id')
        years_filter = holiday.get('years')
        holiday_years = years
        if years_filter is not None:
            valid_years = {int(year_value) for year_value in years_filter}
            holiday_years = [year for year in years if year in valid_years]
            if not holiday_years:
                continue

        def _record(year: int, dt: pd.Timestamp, holiday_id=holiday_id, name=name) -> None:
            """Register a resolved date, keeping the by-id index in step."""
            dt = dt.normalize()
            holiday_dates[dt] = name
            if holiday_id:
                resolved_holiday_dates_by_id.setdefault(holiday_id, {})[year] = dt

        if holiday['date_type'] == 'fixed':
            for year in holiday_years:
                dt = _resolve_observed_holiday_date(year, holiday)
                if dt is None:
                    try:
                        dt = pd.Timestamp(year=year, month=holiday['month'], day=holiday['day'])
                    except ValueError:
                        continue
                _record(year, dt)

        elif holiday['date_type'] == 'movable':
            rule = holiday.get('rule', '')
            for year in holiday_years:
                dt = _resolve_observed_holiday_date(year, holiday)
                if dt is None:
                    easter_date = pd.Timestamp(calc_easter(year))
                    if 'before_easter' in rule:
                        days_before = int(rule.split('_')[0])
                        dt = easter_date - timedelta(days=days_before)
                    elif 'after_easter' in rule:
                        days_after = int(rule.split('_')[0])
                        dt = easter_date + timedelta(days=days_after)
                    elif rule == 'easter':
                        dt = easter_date
                    else:
                        continue
                _record(year, dt)

        elif holiday['date_type'] == 'relative_weekday':
            weekday_name = str(holiday.get('weekday', '')).strip().lower()
            if weekday_name not in _WEEKDAY_NAME_TO_INDEX:
                raise ValueError(
                    f"Holiday {name!r} has an unrecognized weekday {weekday_name!r}. "
                    f"Valid options: {', '.join(sorted(_WEEKDAY_NAME_TO_INDEX))}."
                )
            for year in holiday_years:
                dt = _nth_weekday_of_month(
                    year,
                    int(holiday['month']),
                    _WEEKDAY_NAME_TO_INDEX[weekday_name],
                    int(holiday['week_of_month']),
                )
                _record(year, dt)

        elif holiday['date_type'] == 'relative_to_holiday':
            relative_to = holiday.get('relative_to')
            offset_days = int(holiday.get('offset_days', 0))
            if not relative_to:
                raise ValueError(
                    f"Holiday {name!r} has date_type 'relative_to_holiday' but no 'relative_to' id."
                )
            for year in holiday_years:
                base_dt = resolved_holiday_dates_by_id.get(relative_to, {}).get(year)
                if base_dt is None:
                    # Resolution is a single pass in catalog order, so the anchor
                    # must be defined before the holidays that offset from it.
                    raise ValueError(
                        f"Holiday {name!r} is relative to {relative_to!r}, which has no "
                        f"resolved date for {year}. List the anchor holiday before it in "
                        "the catalog and make sure their 'years' filters overlap."
                    )
                dt = pd.Timestamp(base_dt) + timedelta(days=offset_days)
                _record(year, dt)

    return (
        pd.DataFrame(
            [{'date': d, 'holiday_name': n} for d, n in holiday_dates.items()],
            columns=['date', 'holiday_name'],
        )
        .sort_values('date')
        .reset_index(drop=True)
    )


# Notebook filter for holidays generated as the nth Monday after the 2006 reform.
_NTH_MONDAY_HOLIDAY_NAMES = {
    'Constitution Day',
    "Benito Juarez's Birthday",
    'Mexican Revolution Day',
}


def report_nth_monday_holidays(df_holidays: pd.DataFrame) -> pd.DataFrame:
    """Return nth-Monday holiday rows for notebook-side checks."""
    df = df_holidays.copy()
    df['date'] = pd.to_datetime(df['date'])
    mask = df['holiday_name'].isin(_NTH_MONDAY_HOLIDAY_NAMES)
    return df[mask][['holiday_name', 'date']].assign(
        weekday_name=df.loc[mask, 'date'].dt.day_name(),
        labor_law_rule=df.loc[mask, 'holiday_name'].map({
            'Constitution Day': '1st Monday of February (since 2006)',
            "Benito Juarez's Birthday": '3rd Monday of March (since 2006)',
            'Mexican Revolution Day': '3rd Monday of November (since 2006)',
        }),
    ).reset_index(drop=True)


# Wide dataframe construction

def build_wide_df(
    df_raw: pd.DataFrame,
    month_to_segment: dict,
    exclude_years: list | None = None,
) -> pd.DataFrame:
    """Pivot hourly observations to one row per day and add calendar fields."""
    df = df_raw.copy()
    if exclude_years:
        df = df[~df['ds'].dt.year.isin(exclude_years)]

    df['date'] = df['ds'].dt.date
    df['hour'] = df['ds'].dt.hour

    df_wide = df.pivot_table(
        index='date', columns='hour', values='y', aggfunc='first'
    )
    df_wide = df_wide.dropna()
    df_wide.index = pd.to_datetime(df_wide.index)

    df_wide['dow'] = df_wide.index.dayofweek
    df_wide['month'] = df_wide.index.month
    df_wide['segment'] = df_wide['month'].map(month_to_segment)
    return df_wide


def get_hour_cols(df_wide: pd.DataFrame) -> list:
    """Return the hour columns (integers 0-23) from the wide dataframe."""
    return [c for c in df_wide.columns if isinstance(c, (int, np.integer))]


# Distance computation

def _calc_distances_both(
    profiles: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean distance and `1 - Pearson r` against a centroid."""
    c = centroid.flatten()
    eucl = cdist(profiles, c.reshape(1, -1), metric='euclidean').flatten()
    pear = np.array([1 - np.corrcoef(row, c)[0, 1] for row in profiles])
    return eucl, pear


def _normalize_date_set(date_values) -> set[pd.Timestamp]:
    """Normalize a collection of dates to midnight Timestamps."""
    if date_values is None:
        return set()
    return {pd.Timestamp(dt).normalize() for dt in date_values}


def _zscore_against_reference(
    values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    """Standardize using mean and deviation estimated on a reference subset."""
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    mu = float(np.nanmean(reference_values))
    sigma = float(np.nanstd(reference_values, ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-9:
        sigma = 1e-9

    return (values - mu) / sigma


def compute_distances(
    df_wide: pd.DataFrame,
    hour_cols: list,
    segment_labels: list,
    weekday_names: list,
    distance_metric: str = 'PEARSON_EUCLIDIAN',
    reference_exclude_dates: set | None = None,
) -> pd.DataFrame:
    """Compute per-day distance to the `(segment, dow)` centroid."""
    distance_metric = distance_metric.upper()
    valid = {'EUCLIDIAN', 'PEARSON', 'PEARSON_EUCLIDIAN'}
    if distance_metric not in valid:
        raise ValueError(f'distance_metric must be one of {valid}, received: {distance_metric!r}')

    reference_exclude_dates = _normalize_date_set(reference_exclude_dates)

    rows = []
    for seg_label in segment_labels:
        for dow in range(7):
            mask = (df_wide['segment'] == seg_label) & (df_wide['dow'] == dow)
            subset = df_wide.loc[mask, hour_cols]
            if len(subset) < 3:
                continue

            reference_mask = pd.Series(
                ~subset.index.normalize().isin(reference_exclude_dates),
                index=subset.index,
            )
            reference_subset = subset.loc[reference_mask]

            if len(reference_subset) < 3:
                reference_mask = pd.Series(True, index=subset.index)
                reference_subset = subset

            centroid = reference_subset.values.mean(axis=0)

            eucl_all, pear_all = _calc_distances_both(subset.values, centroid)
            eucl_ref, pear_ref = _calc_distances_both(reference_subset.values, centroid)

            eucl_norm = _zscore_against_reference(eucl_all, eucl_ref)
            pear_norm = _zscore_against_reference(pear_all, pear_ref)

            for date, d_eucl, d_pear, d_eucl_norm, d_pear_norm, is_reference_day in zip(
                subset.index,
                eucl_all,
                pear_all,
                eucl_norm,
                pear_norm,
                reference_mask.to_numpy(),
            ):
                rows.append({
                    'date': date,
                    'segment': seg_label,
                    'dow': dow,
                    'dow_name': weekday_names[dow],
                    'dist_eucl': d_eucl,
                    'dist_pearson': d_pear,
                    'dist_eucl_norm': d_eucl_norm,
                    'dist_pearson_norm': d_pear_norm,
                    'is_reference_day': bool(is_reference_day),
                })

    df_dist = pd.DataFrame(rows)

    if distance_metric == 'EUCLIDIAN':
        df_dist['distance'] = df_dist['dist_eucl_norm']
    elif distance_metric == 'PEARSON':
        df_dist['distance'] = df_dist['dist_pearson_norm']
    else:  # PEARSON_EUCLIDIAN
        df_dist['distance'] = (df_dist['dist_eucl_norm'] + df_dist['dist_pearson_norm']) / 2

    return df_dist


# Outlier detection

def detect_outliers(
    df_dist: pd.DataFrame,
    outlier_percentile: int,
    threshold_reference_only: bool = False,
    promote_dates: set | None = None,
    promote_min_group_percentile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag outliers within each `(segment, dow)` group."""
    thresholds = (
        df_dist.groupby(['segment', 'dow'])['distance']
        .quantile(outlier_percentile / 100)
        .rename('threshold')
        .reset_index()
    )

    if threshold_reference_only and 'is_reference_day' in df_dist.columns:
        thresholds_ref = (
            df_dist[df_dist['is_reference_day']]
            .groupby(['segment', 'dow'])['distance']
            .quantile(outlier_percentile / 100)
            .rename('threshold_reference')
            .reset_index()
        )
        thresholds = thresholds.merge(thresholds_ref, on=['segment', 'dow'], how='left')
        thresholds['threshold'] = thresholds['threshold_reference'].fillna(thresholds['threshold'])
        thresholds = thresholds[['segment', 'dow', 'threshold']]

    df_dist = df_dist.merge(thresholds, on=['segment', 'dow'])
    df_dist['group_percentile'] = (
        df_dist.groupby(['segment', 'dow'])['distance']
        .rank(method='average', pct=True) * 100
    )
    df_dist['is_outlier'] = df_dist['distance'] > df_dist['threshold']
    df_dist['is_promoted_outlier'] = False

    if promote_dates is not None and promote_min_group_percentile is not None:
        promote_dates = _normalize_date_set(promote_dates)
        promote_mask = pd.to_datetime(df_dist['date']).dt.normalize().isin(promote_dates)
        df_dist['is_promoted_outlier'] = promote_mask & (
            df_dist['group_percentile'] >= float(promote_min_group_percentile)
        )
        df_dist['is_outlier'] = df_dist['is_outlier'] | df_dist['is_promoted_outlier']

    df_outliers = (
        df_dist[df_dist['is_outlier']]
        .sort_values('distance', ascending=False)
    )
    return df_dist, df_outliers


# Holiday comparison

def compare_outliers_holidays(
    df_outliers: pd.DataFrame,
    df_holidays: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Join detected outliers with the holiday catalog."""
    df_out = df_outliers[['date', 'segment', 'dow_name', 'distance']].copy()
    df_out['date'] = pd.to_datetime(df_out['date']).dt.normalize()

    df_hol = df_holidays.copy()
    df_hol['date'] = pd.to_datetime(df_hol['date']).dt.normalize()

    df_match = df_out.merge(df_hol, on='date', how='left')
    df_match['is_known_holiday'] = df_match['holiday_name'].notna()

    n_match = int(df_match['is_known_holiday'].sum())
    n_total = len(df_match)
    n_unknown = n_total - n_match

    return df_match, {'n_total': n_total, 'n_match': n_match, 'n_unknown': n_unknown}


def find_holidays_not_detected(
    df_holidays: pd.DataFrame,
    all_dates_in_data: set,
    detected_dates: set,
    weekday_names: list,
) -> pd.DataFrame:
    """Return catalog holidays present in the data but missing from outliers."""
    df_hol = df_holidays.copy()
    df_hol['date'] = pd.to_datetime(df_hol['date']).dt.normalize()

    rows = []
    for _, row in df_hol.iterrows():
        if row['date'] in all_dates_in_data and row['date'] not in detected_dates:
            entry = row.to_dict()
            entry['dow_name'] = weekday_names[row['date'].dayofweek]
            rows.append(entry)

    return pd.DataFrame(rows)


def get_date_sets(
    df_outliers_cmp: pd.DataFrame,
    df_holidays: pd.DataFrame,
    all_dates_in_data: set,
) -> dict:
    """Build the date sets used by the notebook plots."""
    outlier_dates_set = set(pd.to_datetime(df_outliers_cmp['date']).dt.normalize())
    holiday_dates_set = set(pd.to_datetime(df_holidays['date']).dt.normalize())
    match_dates_set = outlier_dates_set & holiday_dates_set
    unknown_dates_set = outlier_dates_set - holiday_dates_set
    missed_dates_set = (holiday_dates_set & all_dates_in_data) - outlier_dates_set

    return {
        'outlier_dates_set': outlier_dates_set,
        'holiday_dates_set': holiday_dates_set,
        'match_dates_set': match_dates_set,
        'unknown_dates_set': unknown_dates_set,
        'missed_dates_set': missed_dates_set,
    }


# Profile helpers

def centroid_dow_month(
    df_wide: pd.DataFrame,
    month: int,
    dow: int,
    exclude_dates: set,
    hour_cols: list,
) -> np.ndarray | None:
    """Return the average profile for one `(month, dow)` pair."""
    mask = (
        (df_wide.index.month == month)
        & (df_wide['dow'] == dow)
        & (~df_wide.index.normalize().isin(exclude_dates))
    )
    sub = df_wide.loc[mask, hour_cols]
    if len(sub) < 2:
        return None
    return sub.values.mean(axis=0)


def build_holiday_groups(
    df_holidays: pd.DataFrame,
    df_wide_index: pd.Index,
) -> dict:
    """Group in-range holiday dates by holiday name."""
    wide_norm = set(df_wide_index.normalize())
    groups: dict = {}
    for _, hrow in df_holidays.iterrows():
        h_date = pd.Timestamp(hrow['date']).normalize()
        if h_date in wide_norm:
            groups.setdefault(hrow['holiday_name'], []).append(h_date)
    return groups


def _zscore_profile(profile: np.ndarray) -> np.ndarray:
    """Normalize an hourly profile with row-wise z-score."""
    arr = np.asarray(profile, dtype=float)
    mu = float(np.nanmean(arr))
    sigma = float(np.nanstd(arr))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return arr - mu
    return (arr - mu) / sigma


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Robust correlation between two profiles, tolerant to near-zero variance."""
    za = _zscore_profile(a)
    zb = _zscore_profile(b)
    denom = float(np.linalg.norm(za) * np.linalg.norm(zb))
    if denom < 1e-12:
        return float('nan')
    return float(np.dot(za, zb) / denom)


def _closest_weekday_reference_label(
    profile: np.ndarray,
    df_wide: pd.DataFrame,
    month: int,
    exclude_dates: set,
    hour_cols: list,
) -> tuple[str | None, float]:
    """Short label for the weekday whose reference profile is the closest."""
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    profile_z = _zscore_profile(profile)
    best_label = None
    best_key = None
    best_corr = float('nan')

    for dow_idx, dow_label in enumerate(dow_labels):
        ref = centroid_dow_month(df_wide, month, dow_idx, exclude_dates, hour_cols)
        if ref is None:
            continue

        corr = _safe_corr(profile, ref)
        corr_score = corr if np.isfinite(corr) else float('-inf')
        dist_score = float(np.linalg.norm(profile_z - _zscore_profile(ref)))
        rank_key = (corr_score, -dist_score)

        if best_key is None or rank_key > best_key:
            best_key = rank_key
            best_label = dow_label
            best_corr = float(corr) if np.isfinite(corr) else float('nan')

    return best_label, best_corr


def _dominant_weekday_reference_from_sim_row(sim_row: pd.Series) -> tuple[str | None, float]:
    """Return the weekday with the largest Pearson similarity in a df_sim row."""
    best_label = None
    best_corr = float('-inf')

    for dow_label in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        corr = sim_row.get(f'r vs {dow_label}', float('nan'))
        if np.isfinite(corr) and float(corr) > best_corr:
            best_label = dow_label
            best_corr = float(corr)

    if best_label is None:
        return None, float('nan')

    return best_label, best_corr


def _format_weekday_reference(label: str | None, corr: float) -> str | None:
    """Format the weekday reference with its Pearson r when available."""
    if label is None:
        return None
    display_label = {
        'Mon': 'Mon',
        'Tue': 'Tue',
        'Wed': 'Wed',
        'Thu': 'Thu',
        'Fri': 'Fri',
        'Sat': 'Sat',
        'Sun': 'Sun',
    }.get(label, label)
    if np.isfinite(corr):
        return f'{display_label} (r={corr:.2f})'
    return display_label


def classify_holiday_weekend_type(
    df_wide: pd.DataFrame,
    df_holidays: pd.DataFrame,
    outlier_dates_set: set,
    hour_cols: list,
    alpha: float = 0.10,
    min_occurrences: int = 3,
) -> dict:
    """Classify holidays as Saturday-like, Sunday-like, Mixed, or Insufficient."""
    holiday_groups = build_holiday_groups(df_holidays, df_wide.index)
    occurrence_rows = []

    for holiday_name, dates in sorted(holiday_groups.items()):
        for h_date in sorted(dates):
            if h_date not in df_wide.index:
                continue

            profile = df_wide.loc[h_date, hour_cols].values.astype(float)
            if np.isnan(profile).any():
                continue

            sat_ref = centroid_dow_month(df_wide, h_date.month, 5, outlier_dates_set, hour_cols)
            sun_ref = centroid_dow_month(df_wide, h_date.month, 6, outlier_dates_set, hour_cols)
            if sat_ref is None or sun_ref is None:
                continue

            profile_z = _zscore_profile(profile)
            sat_z = _zscore_profile(sat_ref)
            sun_z = _zscore_profile(sun_ref)

            corr_sat = _safe_corr(profile, sat_ref)
            corr_sun = _safe_corr(profile, sun_ref)
            dist_sat = float(np.linalg.norm(profile_z - sat_z))
            dist_sun = float(np.linalg.norm(profile_z - sun_z))

            delta_corr = corr_sat - corr_sun
            delta_dist = dist_sun - dist_sat

            sat_votes = int(delta_corr > 0) + int(delta_dist > 0)
            sun_votes = int(delta_corr < 0) + int(delta_dist < 0)
            if sat_votes > sun_votes:
                occurrence_type = 'A'
            elif sun_votes > sat_votes:
                occurrence_type = 'B'
            else:
                occurrence_type = 'Mixed'

            occurrence_rows.append({
                'holiday_name': holiday_name,
                'date': h_date,
                'year': int(h_date.year),
                'month': int(h_date.month),
                'dow': int(h_date.dayofweek),
                'segment': df_wide.loc[h_date, 'segment'] if 'segment' in df_wide.columns else pd.NA,
                'corr_sat': corr_sat,
                'corr_sun': corr_sun,
                'dist_sat': dist_sat,
                'dist_sun': dist_sun,
                'delta_corr_sat_minus_sun': delta_corr,
                'delta_dist_sun_minus_sat': delta_dist,
                'occurrence_type': occurrence_type,
            })

    occurrences_df = pd.DataFrame(occurrence_rows)
    if occurrences_df.empty:
        empty_summary = pd.DataFrame(columns=[
            'holiday_name', 'n_occurrences', 'sat_like_votes', 'sun_like_votes',
            'mixed_votes', 'mean_corr_sat', 'mean_corr_sun', 'median_delta_corr',
            'wilcoxon_pvalue', 'sign_test_pvalue', 'holiday_type', 'decision',
        ])
        return {
            'occurrences_df': occurrences_df,
            'summary_df': empty_summary,
            'overall_stats': {},
        }

    summary_rows = []
    for holiday_name, holiday_group in occurrences_df.groupby('holiday_name', sort=True):
        delta = holiday_group['delta_corr_sat_minus_sun'].dropna().to_numpy(dtype=float)
        non_zero_delta = delta[np.abs(delta) > 1e-12]

        sat_like_votes = int((holiday_group['occurrence_type'] == 'A').sum())
        sun_like_votes = int((holiday_group['occurrence_type'] == 'B').sum())
        mixed_votes = int((holiday_group['occurrence_type'] == 'Mixed').sum())
        decisive_votes = sat_like_votes + sun_like_votes

        wilcoxon_pvalue = float('nan')
        if len(non_zero_delta) >= 2:
            try:
                wilcoxon_pvalue = float(
                    wilcoxon(non_zero_delta, alternative='two-sided', zero_method='wilcox').pvalue
                )
            except ValueError:
                wilcoxon_pvalue = float('nan')

        sign_test_pvalue = float('nan')
        if decisive_votes >= 1:
            sign_test_pvalue = float(binomtest(sat_like_votes, decisive_votes, p=0.5).pvalue)

        median_delta = float(np.nanmedian(delta)) if len(delta) else float('nan')
        prop_sat_like = sat_like_votes / decisive_votes if decisive_votes else float('nan')
        significant = (
            (np.isfinite(wilcoxon_pvalue) and wilcoxon_pvalue < alpha)
            or (np.isfinite(sign_test_pvalue) and sign_test_pvalue < alpha)
        )

        if len(holiday_group) < min_occurrences:
            holiday_type = 'Insufficient'
            decision = f'Fewer than {min_occurrences} occurrences'
        elif significant and median_delta > 0 and prop_sat_like >= 0.60:
            holiday_type = 'A'
            decision = 'Saturday-like'
        elif significant and median_delta < 0 and prop_sat_like <= 0.40:
            holiday_type = 'B'
            decision = 'Sunday-like'
        else:
            holiday_type = 'Mixed'
            decision = 'Insufficient evidence for A/B'

        summary_rows.append({
            'holiday_name': holiday_name,
            'n_occurrences': int(len(holiday_group)),
            'sat_like_votes': sat_like_votes,
            'sun_like_votes': sun_like_votes,
            'mixed_votes': mixed_votes,
            'mean_corr_sat': float(np.nanmean(holiday_group['corr_sat'])),
            'mean_corr_sun': float(np.nanmean(holiday_group['corr_sun'])),
            'median_delta_corr': median_delta,
            'mean_delta_corr': float(np.nanmean(holiday_group['delta_corr_sat_minus_sun'])),
            'mean_delta_dist': float(np.nanmean(holiday_group['delta_dist_sun_minus_sat'])),
            'wilcoxon_pvalue': wilcoxon_pvalue,
            'sign_test_pvalue': sign_test_pvalue,
            'holiday_type': holiday_type,
            'decision': decision,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['holiday_type', 'median_delta_corr', 'holiday_name'],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    overall_delta = occurrences_df['delta_corr_sat_minus_sun'].dropna().to_numpy(dtype=float)
    overall_non_zero = overall_delta[np.abs(overall_delta) > 1e-12]
    overall_stats = {
        'n_occurrences': int(len(occurrences_df)),
        'n_holidays': int(summary_df['holiday_name'].nunique()),
        'overall_mean_delta_corr': float(np.nanmean(overall_delta)),
        'overall_median_delta_corr': float(np.nanmedian(overall_delta)),
        'overall_sat_like_votes': int((occurrences_df['occurrence_type'] == 'A').sum()),
        'overall_sun_like_votes': int((occurrences_df['occurrence_type'] == 'B').sum()),
        'overall_mixed_votes': int((occurrences_df['occurrence_type'] == 'Mixed').sum()),
        'alpha': float(alpha),
    }

    if len(overall_non_zero) >= 2:
        try:
            overall_stats['overall_wilcoxon_pvalue'] = float(
                wilcoxon(overall_non_zero, alternative='two-sided', zero_method='wilcox').pvalue
            )
        except ValueError:
            overall_stats['overall_wilcoxon_pvalue'] = float('nan')
    else:
        overall_stats['overall_wilcoxon_pvalue'] = float('nan')

    decisive_total = overall_stats['overall_sat_like_votes'] + overall_stats['overall_sun_like_votes']
    if decisive_total >= 1:
        overall_stats['overall_sign_test_pvalue'] = float(
            binomtest(overall_stats['overall_sat_like_votes'], decisive_total, p=0.5).pvalue
        )
    else:
        overall_stats['overall_sign_test_pvalue'] = float('nan')

    return {
        'occurrences_df': occurrences_df,
        'summary_df': summary_df,
        'overall_stats': overall_stats,
    }


def run_holiday_ab_validation(
    df_wide: pd.DataFrame,
    df_holidays: pd.DataFrame,
    outlier_dates_set: set,
    hour_cols: list,
    alpha: float = 0.10,
    min_occurrences: int = 3,
) -> dict:
    """Run A/B weekend-type validation, display all results inline, and return key outputs.

    Returns a dict with keys:
        holiday_type_results  — raw output of classify_holiday_weekend_type
        df_holiday_type_occ   — per-occurrence dataframe
        df_holiday_type_summary — per-holiday summary dataframe
        overall_ab_stats      — global statistics dict
        holiday_type_map      — {holiday_name: 'A'/'B'} for decisive holidays
    """
    from IPython.display import display

    holiday_type_results = classify_holiday_weekend_type(
        df_wide=df_wide,
        df_holidays=df_holidays,
        outlier_dates_set=outlier_dates_set,
        hour_cols=hour_cols,
        alpha=alpha,
        min_occurrences=min_occurrences,
    )

    df_holiday_type_occ = holiday_type_results['occurrences_df'].copy()
    df_holiday_type_summary = holiday_type_results['summary_df'].copy()
    overall_ab_stats = holiday_type_results['overall_stats']

    df_summary_display = df_holiday_type_summary.copy()

    print('Holiday A/B validation')
    print(f"Occurrences evaluated: {overall_ab_stats.get('n_occurrences', 0)}")
    print(f"Holidays with data: {overall_ab_stats.get('n_holidays', 0)}")
    print(f"Global Wilcoxon p-value: {overall_ab_stats.get('overall_wilcoxon_pvalue', float('nan')):.4f}")
    print(f"Global sign-test p-value: {overall_ab_stats.get('overall_sign_test_pvalue', float('nan')):.4f}")
    print(f"Decision alpha: {overall_ab_stats.get('alpha', alpha):.2f}")

    holiday_type_labels = {
        'A': 'A - Saturday-like',
        'B': 'B - Sunday-like',
        'Mixed': 'Mixed / unclear',
        'Insufficient': 'Insufficient data',
    }
    df_summary_display['holiday_type_label'] = (
        df_summary_display['holiday_type']
        .map(holiday_type_labels)
        .fillna(df_summary_display['holiday_type'])
    )

    summary_columns = [
        'holiday_name',
        'holiday_type',
        'holiday_type_label',
        'n_occurrences',
        'sat_like_votes',
        'sun_like_votes',
        'mixed_votes',
        'mean_corr_sat',
        'mean_corr_sun',
        'median_delta_corr',
        'wilcoxon_pvalue',
        'sign_test_pvalue',
        'decision',
    ]
    display(
        df_summary_display[summary_columns]
        .sort_values(['holiday_type', 'holiday_name'])
        .reset_index(drop=True)
    )

    df_ab_list = (
        df_summary_display[
            df_summary_display['holiday_type'].isin(['A', 'B'])
        ][
            [
                'holiday_name',
                'holiday_type',
                'holiday_type_label',
                'n_occurrences',
                'sat_like_votes',
                'sun_like_votes',
                'median_delta_corr',
                'wilcoxon_pvalue',
                'sign_test_pvalue',
            ]
        ]
        .sort_values(['holiday_type', 'holiday_name'])
        .reset_index(drop=True)
    )

    print('\nOperational A/B list by holiday')
    display(df_ab_list)

    holiday_type_map = dict(
        zip(df_ab_list['holiday_name'], df_ab_list['holiday_type'])
    )
    print('HOLIDAY_TYPE_MAP =')
    print(holiday_type_map)

    occurrence_columns = [
        'holiday_name',
        'date',
        'year',
        'corr_sat',
        'corr_sun',
        'dist_sat',
        'dist_sun',
        'delta_corr_sat_minus_sun',
        'occurrence_type',
    ]
    display(
        df_holiday_type_occ[occurrence_columns]
        .sort_values(['holiday_name', 'date'])
        .reset_index(drop=True)
    )

    return {
        'holiday_type_results': holiday_type_results,
        'df_holiday_type_occ': df_holiday_type_occ,
        'df_holiday_type_summary': df_holiday_type_summary,
        'overall_ab_stats': overall_ab_stats,
        'holiday_type_map': holiday_type_map,
    }


# Atypical-day clustering

def cluster_atypical_profiles(
    df_wide: pd.DataFrame,
    match_dates_set: set,
    unknown_dates_set: set,
    outlier_dates_set: set,
    n_clusters: int,
    hour_cols: list,
    df_holidays: pd.DataFrame,
) -> dict:
    """Cluster atypical-day profiles and summarize weekday similarity."""
    atypical_dates = match_dates_set | unknown_dates_set

    df_atyp = df_wide[df_wide.index.normalize().isin(atypical_dates)].copy()
    df_atyp['profile_type'] = df_atyp.index.normalize().map(
        lambda d: 'Confirmed holiday' if d in match_dates_set else 'Unknown outlier'
    )

    raw_profiles = df_atyp[hour_cols].values.astype(float)
    normalized_profiles = StandardScaler().fit_transform(raw_profiles.T).T

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(normalized_profiles)
    df_atyp['cluster'] = labels

    centroids_raw = {
        k: df_atyp.loc[df_atyp['cluster'] == k, hour_cols].values.mean(axis=0)
        for k in range(n_clusters)
    }

    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_indices = [0, 1, 2, 3, 4, 5, 6]
    centroids_dow = {}
    for dow_idx, dow_label in zip(dow_indices, dow_labels):
        profiles = [
            centroid_dow_month(df_wide, m, dow_idx, outlier_dates_set, hour_cols)
            for m in range(1, 13)
        ]
        valid = [p for p in profiles if p is not None]
        centroids_dow[dow_label] = np.mean(valid, axis=0) if valid else None

    centroid_sun = centroids_dow['Sun']
    centroid_sat = centroids_dow['Sat']
    centroid_wed = centroids_dow['Wed']

    def pearson_corr(left, right):
        return np.corrcoef(left, right)[0, 1]

    sim_rows = []
    df_hol_norm = df_holidays.copy()
    df_hol_norm['date'] = pd.to_datetime(df_hol_norm['date']).dt.normalize()

    for k in range(n_clusters):
        n_k = int((labels == k).sum())
        row_data = {'Cluster': k, 'n days': n_k}
        for dow_label in dow_labels:
            reference_centroid = centroids_dow[dow_label]
            row_data[f'r vs {dow_label}'] = (
                round(pearson_corr(centroids_raw[k], reference_centroid), 3)
                if reference_centroid is not None else float('nan')
            )

        h_dates_k = (
            df_atyp.loc[
                (df_atyp['cluster'] == k) & (df_atyp['profile_type'] == 'Confirmed holiday')
            ]
            .index.normalize()
        )
        top_h = (
            df_hol_norm[df_hol_norm['date'].isin(h_dates_k)]['holiday_name']
            .value_counts().head(3).index.tolist()
        )
        row_data['top holidays'] = ', '.join(top_h) if top_h else '-'
        sim_rows.append(row_data)

    df_sim = pd.DataFrame(sim_rows).set_index('Cluster')

    return {
        'df_atyp':        df_atyp,
        'labels':         labels,
        'centroids_raw':  centroids_raw,
        'centroids_dow':  centroids_dow,
        'centroid_sun':   centroid_sun,
        'centroid_sat':   centroid_sat,
        'centroid_wed':   centroid_wed,
        'df_sim':         df_sim,
        'n_clusters':     n_clusters,
    }


def run_cluster_atypical_analysis(
    df_wide: pd.DataFrame,
    match_dates_set: set,
    unknown_dates_set: set,
    outlier_dates_set: set,
    n_clusters: int,
    hour_cols: list,
    df_holidays: pd.DataFrame,
    cluster_colors: list,
    unique_id: str,
) -> dict:
    """Cluster atypical profiles, display the similarity table, and plot.

    Wraps cluster_atypical_profiles + plot_cluster_atypical so the notebook
    cell is a single call.  Returns the full cluster_results dict.
    """
    from IPython.display import display

    cluster_results = cluster_atypical_profiles(
        df_wide,
        match_dates_set,
        unknown_dates_set,
        outlier_dates_set,
        n_clusters,
        hour_cols,
        df_holidays,
    )
    df_atyp = cluster_results['df_atyp']
    df_sim = cluster_results['df_sim']
    df_sim_display = df_sim.copy()

    weekday_reference_columns = ['r vs Mon', 'r vs Tue', 'r vs Wed', 'r vs Thu', 'r vs Fri']
    weekend_reference_columns = ['r vs Sat', 'r vs Sun']
    format_dict = {
        col: '{:.3f}'
        for col in weekday_reference_columns + weekend_reference_columns
    }

    def _row_cluster_style(row):
        bg = cluster_colors[row.name % len(cluster_colors)]
        return [f'background-color: {bg}44; color: black'] * len(row)

    print('Similarity between each cluster centroid and the weekday reference profiles')
    display(
        df_sim_display.style
        .apply(_row_cluster_style, axis=1)
        .format(format_dict)
    )

    # --- per-cluster day list ---
    df_hol_norm = df_holidays.copy()
    df_hol_norm['date'] = pd.to_datetime(df_hol_norm['date']).dt.normalize()
    date_to_holiday = dict(zip(df_hol_norm['date'], df_hol_norm['holiday_name']))

    day_rows = []
    for date_idx, row in df_atyp.iterrows():
        d = pd.Timestamp(date_idx).normalize()
        day_rows.append({
            'cluster': int(row['cluster']),
            'date': d,
            'dow': d.day_name()[:3],
            'type': row['profile_type'],
            'holiday_name': date_to_holiday.get(d, '—'),
        })
    df_days = (
        pd.DataFrame(day_rows)
        .sort_values(['cluster', 'date'])
        .reset_index(drop=True)
    )

    def _day_row_style(row):
        bg = cluster_colors[int(row['cluster']) % len(cluster_colors)]
        return [f'background-color: {bg}44; color: black'] * len(row)

    print('\nDays per cluster')
    display(
        df_days.style
        .apply(_day_row_style, axis=1)
        .format({'date': lambda d: d.strftime('%Y-%m-%d')})
        .hide(axis='index')
    )

    plot_cluster_atypical(cluster_results, df_atyp, hour_cols, cluster_colors, unique_id)

    return cluster_results


def classify_cluster_dow_type(
    df_wide: pd.DataFrame,
    df_atyp: pd.DataFrame,
    outlier_dates_set: set,
    hour_cols: list,
    alpha: float = 0.10,
) -> dict:
    """Rank each cluster against all 7 day-of-week reference profiles.

    For every day in each cluster the function computes its Pearson correlation
    with every DOW reference centroid of the same month (Mon → Sun).  At cluster
    level it ranks the 7 DOWs by mean correlation and runs a one-sided Wilcoxon
    signed-rank test between the winner and the runner-up to assess whether the
    top match is statistically unambiguous.

    Returns
    -------
    dict with keys 'occurrences_df', 'summary_df', 'dow_labels'.
    """
    dow_labels  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_indices = [  0,     1,     2,     3,     4,     5,     6   ]
    occurrence_rows = []

    for k in sorted(df_atyp['cluster'].unique()):
        for h_date in sorted(df_atyp[df_atyp['cluster'] == k].index):
            h_date = pd.Timestamp(h_date).normalize()
            if h_date not in df_wide.index:
                continue
            profile = df_wide.loc[h_date, hour_cols].values.astype(float)
            if np.isnan(profile).any():
                continue

            row: dict = {
                'cluster': k,
                'date': h_date,
                'profile_type': (
                    df_atyp.loc[h_date, 'profile_type']
                    if h_date in df_atyp.index else '—'
                ),
            }
            for dow_idx, dow_label in zip(dow_indices, dow_labels):
                ref = centroid_dow_month(
                    df_wide, h_date.month, dow_idx, outlier_dates_set, hour_cols,
                )
                row[f'corr_{dow_label}'] = (
                    _safe_corr(profile, ref) if ref is not None else float('nan')
                )

            valid = {
                d: row[f'corr_{d}']
                for d in dow_labels
                if np.isfinite(row.get(f'corr_{d}', float('nan')))
            }
            row['best_dow'] = max(valid, key=valid.get) if valid else None
            occurrence_rows.append(row)

    occurrences_df = pd.DataFrame(occurrence_rows)

    summary_rows = []
    for k, grp in occurrences_df.groupby('cluster', sort=True):
        row_s: dict = {'cluster': k, 'n_days': int(len(grp))}

        mean_corrs: dict[str, float] = {}
        for d in dow_labels:
            mc = float(np.nanmean(grp[f'corr_{d}']))
            row_s[f'mean_corr_{d}'] = mc
            mean_corrs[d] = mc

        vote_counts = grp['best_dow'].value_counts()
        for d in dow_labels:
            row_s[f'votes_{d}'] = int(vote_counts.get(d, 0))

        ranked   = sorted(dow_labels, key=lambda d: mean_corrs[d], reverse=True)
        winner   = ranked[0]
        runner   = ranked[1]
        row_s['best_dow']      = winner
        row_s['runner_up_dow'] = runner
        row_s['gap_corr']      = round(mean_corrs[winner] - mean_corrs[runner], 4)

        delta    = (grp[f'corr_{winner}'] - grp[f'corr_{runner}']).dropna().to_numpy(float)
        non_zero = delta[np.abs(delta) > 1e-12]
        wilcoxon_p = float('nan')
        if len(non_zero) >= 2:
            try:
                wilcoxon_p = float(
                    wilcoxon(non_zero, alternative='greater', zero_method='wilcox').pvalue
                )
            except ValueError:
                pass

        row_s['wilcoxon_pvalue'] = wilcoxon_p
        significant = np.isfinite(wilcoxon_p) and wilcoxon_p < alpha
        row_s['cluster_type'] = winner if significant else 'unclear'
        if significant:
            row_s['decision'] = f'{winner}-like  (Δr={row_s["gap_corr"]:.3f}, p={wilcoxon_p:.4f})'
        else:
            row_s['decision'] = (
                f'{winner} / {runner} not distinguishable'
                f'  (Δr={row_s["gap_corr"]:.3f}, p={wilcoxon_p:.4f})'
            )
        summary_rows.append(row_s)

    return {
        'occurrences_df': occurrences_df,
        'summary_df':     pd.DataFrame(summary_rows),
        'dow_labels':     dow_labels,
    }


def run_cluster_ab_validation(
    df_wide: pd.DataFrame,
    df_atyp: pd.DataFrame,
    outlier_dates_set: set,
    hour_cols: list,
    cluster_colors: list,
    alpha: float = 0.10,
    df_holidays: pd.DataFrame | None = None,
) -> dict:
    """Rank each cluster against all 7 DOW profiles and display three tables.

    Table 1 — mean Pearson correlation per DOW (heat-tinted; winner = green).
    Table 2 — vote counts (how many days had each DOW as best match).
    Table 3 — per-day detail with all 7 correlations.  If df_holidays is
              provided, 'Confirmed holiday' in profile_type is replaced by
              the actual holiday name.

    Returns the dict produced by classify_cluster_dow_type.
    """
    from IPython.display import display

    results        = classify_cluster_dow_type(df_wide, df_atyp, outlier_dates_set, hour_cols, alpha)
    summary_df     = results['summary_df']
    occurrences_df = results['occurrences_df']
    dow_labels     = results['dow_labels']
    corr_cols      = [f'mean_corr_{d}' for d in dow_labels]
    vote_cols      = [f'votes_{d}'     for d in dow_labels]

    def _row_bg(row):
        bg = cluster_colors[int(row['cluster']) % len(cluster_colors)]
        return [f'background-color: {bg}44; color: black'] * len(row)

    # ── Table 1: mean correlation per DOW ────────────────────────────────────
    sum1 = (
        summary_df[['cluster', 'n_days'] + corr_cols
                   + ['best_dow', 'runner_up_dow', 'gap_corr', 'wilcoxon_pvalue', 'decision']]
        .rename(columns={f'mean_corr_{d}': d for d in dow_labels})
    )
    fmt1 = {d: '{:.3f}' for d in dow_labels}
    fmt1.update({'gap_corr': '{:.3f}', 'wilcoxon_pvalue': '{:.4f}'})

    print(f'Cluster DOW similarity  (alpha = {alpha})')
    print('Mean Pearson r of each cluster against the 7 DOW reference profiles.\n')
    display(
        sum1.style
        .apply(_row_bg, axis=1)
        .highlight_max(subset=dow_labels, axis=1, color='#90EE90')
        .format(fmt1)
        .hide(axis='index')
    )

    # ── Table 2: votes per DOW ───────────────────────────────────────────────
    sum2 = (
        summary_df[['cluster', 'n_days'] + vote_cols + ['best_dow']]
        .rename(columns={f'votes_{d}': d for d in dow_labels})
    )
    print('\nVote counts — best-matching DOW per day')
    display(
        sum2.style
        .apply(_row_bg, axis=1)
        .highlight_max(subset=dow_labels, axis=1, color='#90EE90')
        .format({d: lambda v: str(int(v)) for d in dow_labels})
        .hide(axis='index')
    )

    # ── Table 3: per-day detail ──────────────────────────────────────────────
    corr_cols_occ = [f'corr_{d}' for d in dow_labels]
    occ3 = (
        occurrences_df[['cluster', 'date', 'profile_type', 'best_dow'] + corr_cols_occ]
        .sort_values(['cluster', 'date'])
        .reset_index(drop=True)
        .copy()
    )
    if df_holidays is not None:
        _hol = df_holidays.copy()
        _hol['date'] = pd.to_datetime(_hol['date']).dt.normalize()
        _date_to_name = dict(zip(_hol['date'], _hol['holiday_name']))
        occ3['profile_type'] = occ3.apply(
            lambda r: _date_to_name.get(r['date'], r['profile_type'])
            if r['profile_type'] == 'Confirmed holiday' else r['profile_type'],
            axis=1,
        )
    fmt3 = {'date': lambda d: d.strftime('%Y-%m-%d')}
    fmt3.update({c: '{:.3f}' for c in corr_cols_occ})

    print('\nPer-day detail')
    display(
        occ3.style
        .apply(_row_bg, axis=1)
        .highlight_max(subset=corr_cols_occ, axis=1, color='#90EE90')
        .format(fmt3)
        .hide(axis='index')
    )

    return results


def display_cluster_holiday_crosstab(
    df_atyp: pd.DataFrame,
    df_holidays: pd.DataFrame,
    cluster_colors: list,
) -> pd.DataFrame:
    """Pivot table: rows = holiday name, columns = cluster, values = day count.

    Returns the pivot DataFrame (NaN where no days exist for that combination).
    The styled table is displayed inline (Jupyter).
    """
    from IPython.display import display

    df_hol = df_holidays.copy()
    df_hol['date'] = pd.to_datetime(df_hol['date']).dt.normalize()
    date_to_holiday = dict(zip(df_hol['date'], df_hol['holiday_name']))

    df_cross = df_atyp[['cluster', 'profile_type']].copy()
    df_cross.index = pd.to_datetime(df_cross.index).normalize()
    df_cross['holiday_name'] = df_cross.index.map(date_to_holiday).fillna('— Unknown outlier')

    counts = (
        df_cross.groupby(['holiday_name', 'cluster'])
        .size()
        .rename('n')
        .reset_index()
    )

    pivot = (
        counts.pivot(index='holiday_name', columns='cluster', values='n')
        .rename(columns=lambda c: f'Cluster {c}')
        .sort_index()
    )

    def _fmt_cell(v):
        return f'{int(v)}' if pd.notna(v) and v > 0 else '—'

    def _col_style(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in df.columns:
            try:
                k = int(col.split()[-1])
                bg = cluster_colors[k % len(cluster_colors)]
                styles[col] = f'background-color: {bg}44; color: black'
            except (ValueError, IndexError):
                pass
        return styles

    print('Holiday × Cluster distribution')
    print('Rows = holiday name   |   Columns = cluster (count of days)\n')
    display(
        pivot.style
        .apply(_col_style, axis=None)
        .format(_fmt_cell, na_rep='—')
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
    )

    return pivot


# Bridge-day detection

def detect_bridges(
    df_outliers_cmp: pd.DataFrame,
    df_holidays_confirmed: pd.DataFrame,
    holiday_dates_set: set,
    window_days: int = 4,
    bridge_min_dist: int = 0,
) -> pd.DataFrame:
    """Detect unknown outliers that sit close to confirmed holidays."""
    df_unknown = df_outliers_cmp[
        ~df_outliers_cmp['date'].isin(holiday_dates_set)
    ].copy()
    df_unknown['date_norm'] = pd.to_datetime(df_unknown['date']).dt.normalize()

    bridge_cols = [
        'date', 'dow_name', 'segment', 'direction', 'delta_days',
        'holiday_ref', 'holiday_date', 'dist',
    ]
    bridge_rows = []
    for _, holiday_row in df_holidays_confirmed.iterrows():
        h_date = pd.Timestamp(holiday_row['date']).normalize()

        for delta in range(-window_days, window_days + 1):
            if abs(delta) <= bridge_min_dist:
                continue

            candidate = h_date + pd.Timedelta(days=delta)
            candidate_rows = df_unknown[df_unknown['date_norm'] == candidate]
            if candidate_rows.empty:
                continue

            row = candidate_rows.iloc[0]
            bridge_rows.append({
                'date': candidate,
                'dow_name': row['dow_name'],
                'segment': row['segment'],
                'direction': 'before' if delta < 0 else 'after',
                'delta_days': abs(delta),
                'holiday_ref': holiday_row['holiday_name'],
                'holiday_date': h_date,
                'dist': round(row['distance'], 4),
            })

    df_bridges = (
        pd.DataFrame(bridge_rows, columns=bridge_cols)
        .drop_duplicates(subset=['date'])
        .sort_values('date')
        .reset_index(drop=True)
    )

    if not df_bridges.empty:
        df_bridges['date'] = pd.to_datetime(df_bridges['date'])
        df_bridges['holiday_date'] = pd.to_datetime(df_bridges['holiday_date'])

    return df_bridges


def bridge_gap_color(delta_days: int) -> str:
    """Color based on proximity to the holiday (for bridge-day plots)."""
    gap = abs(delta_days)
    if gap == 1:
        return '#d62728'
    if gap <= 3:
        return '#ff7f0e'
    if gap <= 5:
        return '#bcbd22'
    return '#aec7e8'


# Summary

def print_summary(
    unique_id: str,
    df_wide: pd.DataFrame,
    n_total: int,
    n_match: int,
    n_unknown: int,
    df_missed: pd.DataFrame,
    df_outliers_cmp: pd.DataFrame,
    outlier_percentile: int,
    weekday_names: list,
) -> None:
    """Print the final summary of the holiday analysis."""
    print('Holiday identification summary')
    print(f'Series: {unique_id}')
    print(f'Range: {df_wide.index.min().date()} to {df_wide.index.max().date()}')
    print(f'Days analyzed: {len(df_wide)}')
    print(f'Outlier threshold: percentile {outlier_percentile}')
    print(f'Detected outliers: {n_total}')
    print(f'Match the holiday catalog: {n_match}')
    print(f'New candidates: {n_unknown}')
    print(f'Known holidays not detected: {len(df_missed)}')
    print('\nOutlier dates')
    for date_value in sorted(df_outliers_cmp['date'].dt.date.unique()):
        print(f'  {date_value} ({weekday_names[date_value.weekday()]})')


# Plotting functions


def _build_date_value_lookup(
    df_dates: pd.DataFrame | None,
    value_column: str,
) -> dict[pd.Timestamp, str]:
    """Return a normalized date -> label lookup when the column is available."""
    if (
        df_dates is None
        or 'date' not in df_dates.columns
        or value_column not in df_dates.columns
    ):
        return {}

    lookup: dict[pd.Timestamp, str] = {}
    for _, row in df_dates[['date', value_column]].iterrows():
        value = row.get(value_column, pd.NA)
        if pd.isna(value):
            continue
        lookup[pd.Timestamp(row['date']).normalize()] = str(value)
    return lookup


def _format_date_with_lookup(
    date_value: pd.Timestamp | str,
    lookup: dict[pd.Timestamp, str] | None = None,
    date_format: str = '%d/%m/%y',
) -> str:
    """Format a date and append a lookup value such as ``[F]`` when present."""
    normalized = pd.Timestamp(date_value).normalize()
    label = normalized.strftime(date_format)
    if not lookup:
        return label

    suffix = lookup.get(normalized, '')
    if suffix == '':
        return label
    return f'{label} [{suffix}]'

def plot_profiles_by_segment_dow(
    df_wide: pd.DataFrame,
    hour_cols: list,
    segment_labels: list,
    weekday_names: list,
    match_dates_set: set,
    unknown_dates_set: set,
    missed_dates_set: set,
    unique_id: str,
    df_holidays: pd.DataFrame = None,
) -> None:
    """Plot daily profiles by segment and weekday."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    _date_to_holiday = {}
    _date_to_analog_cluster = _build_date_value_lookup(df_holidays, 'analog_cluster')
    if df_holidays is not None:
        for _, hr in df_holidays.iterrows():
            dt = pd.Timestamp(hr['date']).normalize()
            _date_to_holiday[dt] = hr['holiday_name']

    n_segments = len(segment_labels)
    fig, axes = plt.subplots(n_segments, 7, figsize=(28, 5 * n_segments), sharey=True)
    if n_segments == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f'Hourly profiles by segment × weekday — {unique_id}',
        fontsize=16, y=1.002,
    )

    for i_seg, seg_label in enumerate(segment_labels):
        for dow in range(7):
            ax = axes[i_seg, dow]
            mask = (df_wide['segment'] == seg_label) & (df_wide['dow'] == dow)
            subset = df_wide.loc[mask]

            if len(subset) == 0:
                continue

            blue_dates = []
            green_dates = []

            for date_idx, row in subset.iterrows():
                date_norm = pd.Timestamp(date_idx).normalize()
                if date_norm in match_dates_set:
                    ax.plot(hour_cols, row[hour_cols].values, color='green', alpha=0.7, lw=1.5)
                    green_dates.append(date_norm)
                elif date_norm in unknown_dates_set:
                    y_vals = row[hour_cols].values
                    ax.plot(hour_cols, y_vals, color='red', alpha=0.6, lw=1.2)
                    peak_hour = int(np.argmax(y_vals))
                    ax.text(
                        peak_hour, y_vals[peak_hour],
                        _format_date_with_lookup(
                            date_norm,
                            _date_to_analog_cluster,
                            date_format='%d/%m/%y',
                        ),
                        fontsize=16, color='red', fontweight='bold',
                        ha='center', va='bottom', rotation=60,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6),
                    )
                elif date_norm in missed_dates_set:
                    ax.plot(hour_cols, row[hour_cols].values, color='blue', alpha=0.7, lw=1.5)
                    blue_dates.append(date_norm)
                else:
                    ax.plot(hour_cols, row[hour_cols].values, color='steelblue', alpha=0.35, lw=0.5)

            centroid = subset[hour_cols].mean().values
            ax.plot(hour_cols, centroid, color='black', lw=2, ls='--')

            if green_dates:
                sorted_g = [
                    _format_date_with_lookup(
                        d,
                        _date_to_analog_cluster,
                        date_format='%d/%m/%y',
                    )
                    for d in sorted(green_dates)
                ]
                mid = (len(sorted_g) + 1) // 2
                label_green = '\n'.join([', '.join(sorted_g[:mid]), ', '.join(sorted_g[mid:])]).strip('\n') if len(sorted_g) > 3 else ', '.join(sorted_g)
                ax.text(
                    0.5, 0.10, label_green,
                    transform=ax.transAxes,
                    fontsize=9, color='green', ha='center', va='bottom',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
                )

            if blue_dates:
                sorted_b = [
                    _format_date_with_lookup(
                        d,
                        _date_to_analog_cluster,
                        date_format='%d/%m/%y',
                    )
                    for d in sorted(blue_dates)
                ]
                mid = (len(sorted_b) + 1) // 2
                label_txt = '\n'.join([', '.join(sorted_b[:mid]), ', '.join(sorted_b[mid:])]).strip('\n') if len(sorted_b) > 3 else ', '.join(sorted_b)
                ax.text(
                    0.5, 0.02, label_txt,
                    transform=ax.transAxes,
                    fontsize=9, color='#1a5fa8', ha='center', va='bottom',
                    fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75),
                )

            holiday_names_here = set()
            analog_clusters_here = set()
            for d in green_dates + blue_dates:
                name = _date_to_holiday.get(d)
                if name:
                    holiday_names_here.add(name)
                analog_cluster = _date_to_analog_cluster.get(d, '')
                if analog_cluster:
                    analog_clusters_here.add(analog_cluster)

            title = f'{weekday_names[dow]}\n({seg_label}, n={len(subset)})'
            if holiday_names_here:
                title += f'\n({", ".join(sorted(holiday_names_here))})'
            if analog_clusters_here:
                title += f'\nAnalog: {"/".join(sorted(analog_clusters_here))}'

            ax.set_title(title, fontsize=12, pad=6)
            ax.set_xlabel('Hour', fontsize=10)
            ax.tick_params(labelsize=10)
            if dow == 0:
                ax.set_ylabel(seg_label, fontsize=11)

    legend_elements = [
        mlines.Line2D([0], [0], color='green', lw=2, label='Outlier = known holiday'),
        mlines.Line2D([0], [0], color='red',   lw=2, label='Outlier ≠ holiday (new candidate)'),
        mlines.Line2D([0], [0], color='blue',  lw=2, label='Known holiday not detected'),
        mlines.Line2D([0], [0], color='black', lw=2, ls='--', label='Centroid'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.show()


def plot_profiles_by_holiday(
    df_wide: pd.DataFrame,
    holiday_groups: dict,
    df_holidays: pd.DataFrame,
    hour_cols: list,
    match_dates_set: set,
    outlier_dates_set: set,
    unique_id: str,
    ncols: int = 5,
) -> None:
    """Plot one panel per holiday with all available yearly profiles."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    holiday_names_sorted = sorted(holiday_groups.keys())
    n_holidays = len(holiday_names_sorted)
    nrows = math.ceil(n_holidays / ncols)
    _date_to_analog_cluster = _build_date_value_lookup(df_holidays, 'analog_cluster')

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), sharey=True)
    axes_flat = axes.flatten()
    fig.suptitle(
        f'Hourly profiles by holiday — all years overlaid  |  {unique_id}',
        fontsize=18, y=1.002,
    )

    for idx, h_name in enumerate(holiday_names_sorted):
        ax = axes_flat[idx]
        profiles = []
        months_seen = set()
        holiday_clusters = set()

        for h_date in sorted(holiday_groups[h_name]):
            if h_date not in df_wide.index:
                continue
            profile = df_wide.loc[h_date, hour_cols].values.astype(float)
            if np.isnan(profile).any():
                continue
            profiles.append(profile)
            months_seen.add(h_date.month)
            analog_cluster = _date_to_analog_cluster.get(pd.Timestamp(h_date).normalize(), '')
            if analog_cluster:
                holiday_clusters.add(analog_cluster)

            nearest_weekday, nearest_corr = _closest_weekday_reference_label(
                profile,
                df_wide,
                h_date.month,
                outlier_dates_set,
                hour_cols,
            )

            color, alpha, lw = (
                ('green', 0.85, 2.0) if h_date in match_dates_set
                else ('#1f77b4', 0.70, 1.5)
            )
            ax.plot(hour_cols, profile, color=color, alpha=alpha, lw=lw)
            peak = int(np.argmax(profile))
            label = _format_date_with_lookup(
                h_date,
                _date_to_analog_cluster,
                date_format='%d/%m/%Y',
            )
            formatted_reference = _format_weekday_reference(nearest_weekday, nearest_corr)
            if formatted_reference is not None:
                label = f'{label} · {formatted_reference}'
            ax.text(hour_cols[peak], profile[peak], label,
                    fontsize=11, color=color, fontweight='bold', ha='center', va='bottom')

        if profiles:
            ax.plot(hour_cols, np.mean(profiles, axis=0), color='black', lw=2.5, ls='--', zorder=5)

        if months_seen:
            main_month = max(
                months_seen,
                key=lambda m: sum(1 for d in holiday_groups[h_name] if d.month == m),
            )
            for dow_ref, col_ref, lbl_ref in [(6, '#d62728', 'Sun'), (5, '#f4a0b0', 'Sat')]:
                c = centroid_dow_month(df_wide, main_month, dow_ref, outlier_dates_set, hour_cols)
                if c is not None:
                    ax.plot(hour_cols, c, color=col_ref, lw=2.2, ls=':', zorder=6)
                    ax.text(hour_cols[-1], c[-1], lbl_ref,
                            fontsize=13, color=col_ref, fontweight='bold', ha='left', va='center')

        title = f'{h_name}\n(n={len(profiles)})'
        if holiday_clusters:
            title += f'\nAnalog: {"/".join(sorted(holiday_clusters))}'

        ax.set_title(title, fontsize=13, pad=6)
        ax.set_xlabel('Hour', fontsize=11)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=10)

    for idx in range(n_holidays, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.legend(
        handles=[
            mlines.Line2D([0], [0], color='green',   lw=2,          label='Detected as outlier'),
            mlines.Line2D([0], [0], color='#1f77b4', lw=2,          label='Holiday NOT detected'),
            mlines.Line2D([0], [0], color='black',   lw=2, ls='--', label='Holiday centroid'),
            mlines.Line2D([0], [0], color='#d62728', lw=2, ls=':',  label='Sunday centroid (same month)'),
            mlines.Line2D([0], [0], color='#f4a0b0', lw=2, ls=':',  label='Saturday centroid (same month)'),
        ],
        loc='lower center', ncol=5, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_cluster_atypical(
    cluster_results: dict,
    df_atyp: pd.DataFrame,
    hour_cols: list,
    cluster_colors: list,
    unique_id: str,
) -> None:
    """Plot clustered atypical days and their weekday-similarity summary."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    n_clusters = cluster_results['n_clusters']
    centroids_raw = cluster_results['centroids_raw']
    centroid_sun = cluster_results['centroid_sun']
    centroid_sat = cluster_results['centroid_sat']
    centroid_wed = cluster_results['centroid_wed']
    df_sim = cluster_results['df_sim']

    fig, axes = plt.subplots(
        1, n_clusters + 1,
        figsize=(5.5 * (n_clusters + 1), 6),
        gridspec_kw={'width_ratios': [3] * n_clusters + [2]},
    )
    fig.suptitle(
        f'Clustering of atypical days — do they resemble a Sunday?  |  {unique_id}',
        fontsize=16, y=1.01,
    )

    for k in range(n_clusters):
        ax = axes[k]
        cluster_color = cluster_colors[k % len(cluster_colors)]
        cluster_mask = df_atyp['cluster'] == k
        cluster_rows = df_atyp[cluster_mask]

        for _, row in cluster_rows.iterrows():
            line_color = cluster_color if row['profile_type'] == 'Confirmed holiday' else '#aaaaaa'
            ax.plot(hour_cols, row[hour_cols].values, color=line_color, alpha=0.35, lw=1.2)

        ax.plot(hour_cols, centroids_raw[k], color=cluster_color, lw=3, ls='-', zorder=6, label=f'Centroid C{k}')
        ax.plot(hour_cols, centroid_sun, color='#d62728', lw=2.5, ls='--', zorder=7, label='Sun (ref)')
        ax.plot(hour_cols, centroid_wed, color='#2ca02c', lw=2.0, ls=':', zorder=7, label='Wed (ref)')
        ax.plot(hour_cols, centroid_sat, color='#f4a0b0', lw=2.5, ls=':', zorder=7, label='Sat (ref)')

        n_h = (cluster_mask & (df_atyp['profile_type'] == 'Confirmed holiday')).sum()
        n_u = (cluster_mask & (df_atyp['profile_type'] == 'Unknown outlier')).sum()
        r_sun = df_sim.loc[k, 'r vs Sun']
        r_wed = df_sim.loc[k, 'r vs Wed']
        dominant_weekday, dominant_corr = _dominant_weekday_reference_from_sim_row(df_sim.loc[k])
        dominant_reference = _format_weekday_reference(dominant_weekday, dominant_corr) or 'n/a'
        ax.set_title(
            f'Cluster {k}  (n={int(cluster_mask.sum())})\n'
            f'holidays={n_h}  unknown={n_u}\n'
            f'dominant ref={dominant_reference}\n'
            f'r vs Sun={r_sun:.3f}  r vs Wed={r_wed:.3f}',
            fontsize=12,
        )
        ax.set_xlabel('Hour', fontsize=11)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9, loc='upper left')

    import matplotlib.patches as mpatches

    ax_bar = axes[-1]
    x = np.arange(n_clusters)
    width = 0.25
    ref_specs = [
        ('r vs Sun', -width, '',   0.90),
        ('r vs Wed',  0,     '//', 0.60),
        ('r vs Sat', +width, '..', 0.35),
    ]
    for k in range(n_clusters):
        c = cluster_colors[k % len(cluster_colors)]
        for ref_key, offset, hatch, alpha in ref_specs:
            ax_bar.bar(
                x[k] + offset,
                df_sim.loc[k, ref_key],
                width,
                color=c,
                alpha=alpha,
                hatch=hatch,
                edgecolor='white',
                linewidth=0.5,
            )
    ax_bar.axhline(0.95, color='gray', lw=1, ls='--')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'C{k}' for k in range(n_clusters)], fontsize=12)
    for tick_label, k in zip(ax_bar.get_xticklabels(), range(n_clusters)):
        tick_label.set_color(cluster_colors[k % len(cluster_colors)])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Pearson r', fontsize=12)
    ax_bar.set_title('Similarity\nvs Sun / Wed / Sat', fontsize=13)
    hatch_legend = [
        mpatches.Patch(facecolor='#888888', alpha=0.90, hatch='',   edgecolor='white', label='r vs Sun'),
        mpatches.Patch(facecolor='#888888', alpha=0.60, hatch='//', edgecolor='white', label='r vs Wed'),
        mpatches.Patch(facecolor='#888888', alpha=0.35, hatch='..', edgecolor='white', label='r vs Sat'),
        mlines.Line2D([0], [0], color='gray', lw=1, ls='--', label='r = 0.95'),
    ]
    ax_bar.legend(handles=hatch_legend, fontsize=10)
    ax_bar.tick_params(labelsize=11)

    plt.tight_layout()
    plt.show()


def plot_distance_distribution(
    df_dist: pd.DataFrame,
    segment_labels: list,
    outlier_percentile: int,
    unique_id: str,
) -> None:
    """Plot distance histograms by segment with the outlier threshold."""
    import matplotlib.pyplot as plt

    n_segs = len(segment_labels)
    ncols = min(4, n_segs)
    nrows = math.ceil(n_segs / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    fig.suptitle(f'Pearson distance distribution by segment — {unique_id}', fontsize=15)

    for idx, seg_label in enumerate(segment_labels):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_dist[df_dist['segment'] == seg_label]['distance']

        if len(sub) == 0:
            ax.set_visible(False)
            continue

        thr = sub.quantile(outlier_percentile / 100)
        ax.hist(sub, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(thr, color='red', lw=2, ls='--', label=f'p{outlier_percentile}={thr:.3f}')
        ax.set_title(seg_label, fontsize=12)
        ax.set_xlabel('Distance (1 − r)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)

    for idx in range(n_segs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_bridges_timeline(
    df_bridges: pd.DataFrame,
    df_holidays_confirmed: pd.DataFrame,
    df_missed: pd.DataFrame,
    unique_id: str,
) -> None:
    """Plot detected bridge days against confirmed holiday dates."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    if df_bridges.empty:
        print('No bridge days detected - nothing to visualize.')
        return

    years_plot = sorted(df_bridges['date'].dt.year.unique())

    fig, axes = plt.subplots(
        len(years_plot), 1,
        figsize=(18, 2.8 * len(years_plot)),
        sharex=False,
    )
    if len(years_plot) == 1:
        axes = [axes]

    fig.suptitle(
        f'Unknown outliers - proximity to confirmed holidays  |  {unique_id}',
        fontsize=13, y=1.005,
    )

    for ax, yr in zip(axes, years_plot):
        ax.set_xlim(0, 366)
        ax.set_ylim(-0.65, 0.65)
        ax.axhline(0, color='#cccccc', lw=0.8, zorder=0)
        ax.set_title(str(yr), fontsize=10, loc='left', pad=3)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        ax.set_xlabel('Day of year', fontsize=8)

        yr_holidays = df_holidays_confirmed[df_holidays_confirmed['date'].dt.year == yr]
        for _, holiday_row in yr_holidays.iterrows():
            doy = holiday_row['date'].timetuple().tm_yday
            ax.axvline(doy, color='#2ca02c', lw=1.5, alpha=0.6, zorder=1)
            ax.text(doy + 0.5, 0.54, holiday_row['holiday_name'],
                    rotation=75, fontsize=6, color='#2ca02c', ha='left', va='bottom', zorder=3)

        yr_silent = (
            df_missed[pd.to_datetime(df_missed['date']).dt.year == yr]
            if not df_missed.empty else pd.DataFrame()
        )
        for _, holiday_row in yr_silent.iterrows():
            doy = pd.Timestamp(holiday_row['date']).timetuple().tm_yday
            ax.axvline(doy, color='#aaaaaa', lw=1.0, alpha=0.4, ls=':', zorder=1)
            ax.text(doy + 0.5, -0.58, holiday_row['holiday_name'],
                    rotation=75, fontsize=5.5, color='#aaaaaa', ha='left', va='bottom', zorder=2)

        yr_rows = df_bridges[df_bridges['date'].dt.year == yr]
        for _, bridge_row in yr_rows.iterrows():
            doy = bridge_row['date'].timetuple().tm_yday
            color = bridge_gap_color(bridge_row['delta_days'])
            is_bridge = bridge_row['delta_days'] <= 3
            marker = 'D' if is_bridge else 'o'
            ax.scatter(doy, 0, color=color, s=100, zorder=4,
                       marker=marker, edgecolors='black', linewidths=0.6)
            y_txt = 0.28 if bridge_row['direction'] == 'after' else -0.28
            ax.text(doy, y_txt, f"{bridge_row['date'].strftime('%d/%m')}\n{bridge_row['direction']}",
                    fontsize=6.5, ha='center', va='center', zorder=5)
            if is_bridge:
                h_doy = pd.Timestamp(bridge_row['holiday_date']).timetuple().tm_yday
                ax.annotate('',
                            xy=(h_doy, 0.02), xytext=(doy, 0.02),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=1.2, connectionstyle='arc3,rad=0.35'),
                            zorder=3)

    legend_elements = [
        mpatches.Patch(color='#2ca02c', label='Confirmed holiday (outlier)'),
        mpatches.Patch(color='#aaaaaa', label='Silent holiday (excluded as anchor)'),
        mpatches.Patch(color='#d62728', label='Direct bridge (1d)'),
        mpatches.Patch(color='#ff7f0e', label='Extended bridge (2-3d)'),
        mpatches.Patch(color='#bcbd22', label='Long bridge (4-5d)'),
        mpatches.Patch(color='#aec7e8', label='Unrelated (>5d)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=9, label='Possible bridge'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=9, label='Unrelated'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.015))
    plt.tight_layout()
    plt.show()


def run_cluster_38h_analysis(
    df_wide: pd.DataFrame,
    match_dates_set: set,
    df_holidays_display: pd.DataFrame,
    hour_cols: list,
    cluster_colors: list,
    unique_id: str,
    n_clusters: int = 3,
    previously_w_hours: int = 14,
) -> dict:
    """Cluster confirmed holidays by their 38-h event profile.

    For each holiday D, builds a feature vector of length
    ``previously_w_hours + 24`` by concatenating:
      - the last ``previously_w_hours`` hours of the eve day (D-1)
      - all 24 hours of the holiday day (D)

    KMeans is applied to the standardised vectors.  Cluster labels are
    DOW-agnostic letters (C, D, E, ...).

    Parameters
    ----------
    df_wide:
        Wide-format DataFrame with DatetimeIndex (one row per day) and
        columns h0…h23 plus ``segment``.
    match_dates_set:
        Set of normalised Timestamps for confirmed holiday dates.
    df_holidays_display:
        DataFrame with at least ``date`` and ``holiday_name`` columns.
    hour_cols:
        List of 24 hour-column names in order, e.g. ``['h0', ..., 'h23']``.
    cluster_colors:
        List of hex colour strings; one per cluster (reused cyclically).
    unique_id:
        Series label used in plot titles.
    n_clusters:
        Number of KMeans clusters (default 3).
    previously_w_hours:
        Hours taken from the eve day (default 14, i.e. h10…h23).

    Returns
    -------
    dict with keys:
        ``df_phol``       – per-event DataFrame with cluster labels
        ``df_days_phol``  – per-day table sorted by cluster/date
        ``kmeans_phol``   – fitted KMeans object
        ``centroids_phol``– (n_clusters, 38) centroid array
        ``feat_cols``     – ordered list of the 38 feature column names
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    _CLUSTER_LABELS = list('CDEFGHIJ')

    eve_hour_cols = hour_cols[-previously_w_hours:]
    hol_hour_cols = hour_cols
    feat_cols = (
        [f'eve_{c}' for c in eve_hour_cols]
        + [f'hol_{c}' for c in hol_hour_cols]
    )

    _date_to_name = dict(zip(
        pd.to_datetime(df_holidays_display['date']).dt.normalize(),
        df_holidays_display['holiday_name'],
    ))

    _wide_idx = set(df_wide.index.normalize())
    rows = []
    for d in sorted(match_dates_set):
        eve = d - pd.Timedelta(days=1)
        if eve not in _wide_idx:
            continue
        row_eve = df_wide.loc[df_wide.index.normalize() == eve].squeeze()
        row_hol = df_wide.loc[df_wide.index.normalize() == d].squeeze()
        eve_vals = row_eve[eve_hour_cols].values.astype(float)
        hol_vals = row_hol[hol_hour_cols].values.astype(float)
        if np.isnan(eve_vals).any() or np.isnan(hol_vals).any():
            continue
        rows.append({
            'date': d,
            'holiday_name': _date_to_name.get(d, str(d.date())),
            **dict(zip([f'eve_{c}' for c in eve_hour_cols], eve_vals)),
            **dict(zip([f'hol_{c}' for c in hol_hour_cols], hol_vals)),
        })

    df_phol = pd.DataFrame(rows).set_index('date')
    print(f'Events with complete {previously_w_hours + 24}-h profile: {len(df_phol)}')

    X = df_phol[feat_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    kmeans_phol = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit(X_scaled)
    df_phol['cluster'] = kmeans_phol.labels_
    df_phol['cluster_type'] = [_CLUSTER_LABELS[k] for k in kmeans_phol.labels_]

    centroids_phol = np.array([
        df_phol.loc[df_phol['cluster'] == k, feat_cols].values.mean(axis=0)
        for k in range(n_clusters)
    ])

    day_rows = []
    for d, row in df_phol.iterrows():
        day_rows.append({'type': row['cluster_type'], 'date': d,
                         'dow': d.day_name()[:3], 'holiday_name': row['holiday_name']})
    df_days_phol = (
        pd.DataFrame(day_rows)
        .sort_values(['type', 'date'])
        .reset_index(drop=True)
    )

    def _day_bg(row):
        k = _CLUSTER_LABELS.index(row['type'])
        return [f'background-color: {cluster_colors[k % len(cluster_colors)]}44; color: black'] * len(row)

    print('\nDays per cluster')
    display(df_days_phol.style
            .apply(_day_bg, axis=1)
            .format({'date': lambda d: d.strftime('%Y-%m-%d')})
            .hide(axis='index'))

    x_axis = list(range(-previously_w_hours, 24))
    fig, axes = plt.subplots(1, n_clusters, figsize=(5.5 * n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    fig.suptitle(
        f'38-h event profiles  (eve last {previously_w_hours} h + holiday 24 h)  |  {unique_id}',
        fontsize=13, y=1.02)
    for k, ax in enumerate(axes):
        label = _CLUSTER_LABELS[k]
        color = cluster_colors[k % len(cluster_colors)]
        mask = df_phol['cluster'] == k
        data = df_phol.loc[mask, feat_cols].values
        for profile in data:
            ax.plot(x_axis, profile, color=color, alpha=0.30, lw=0.9)
        ax.plot(x_axis, centroids_phol[k], color=color, lw=3,
                label=f'Centroid {label}', zorder=5)
        ax.axvline(0, color='grey', ls='--', lw=0.9, label='midnight')
        ax.set_title(f'Type {label}  (n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Hour relative to holiday start (h=0)', fontsize=10)
        ax.set_xticks(range(-previously_w_hours, 24, 4))
        if k == 0:
            ax.set_ylabel(f'Demand [{unique_id}]', fontsize=10)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    display_cluster_38h_crosstab(df_phol, cluster_colors, _CLUSTER_LABELS[:n_clusters])

    return {
        'df_phol': df_phol,
        'df_days_phol': df_days_phol,
        'kmeans_phol': kmeans_phol,
        'centroids_phol': centroids_phol,
        'feat_cols': feat_cols,
    }


def display_cluster_38h_crosstab(
    df_phol: pd.DataFrame,
    cluster_colors: list,
    cluster_labels: list | None = None,
) -> pd.DataFrame:
    """Holiday × Cluster pivot table for the 38-h event-profile clustering.

    Parameters
    ----------
    df_phol:
        Output DataFrame from ``run_cluster_38h_analysis``; must contain
        ``holiday_name`` and ``cluster_type`` columns.
    cluster_colors:
        List of hex colour strings; indexed by position of the letter label
        in ``cluster_labels``.
    cluster_labels:
        Ordered list of cluster-type letters actually present (e.g. ['C','D','E']).
        Defaults to the sorted unique values in ``df_phol['cluster_type']``.

    Returns
    -------
    pd.DataFrame
        The pivot table (rows = holiday name, columns = cluster type letter).
    """
    from IPython.display import display as ipy_display

    if cluster_labels is None:
        cluster_labels = sorted(df_phol['cluster_type'].unique())

    counts = (
        df_phol.groupby(['holiday_name', 'cluster_type'])
        .size()
        .rename('n')
        .reset_index()
    )

    pivot = (
        counts.pivot(index='holiday_name', columns='cluster_type', values='n')
        .reindex(columns=cluster_labels)
        .sort_index()
    )

    def _fmt_cell(v):
        return f'{int(v)}' if pd.notna(v) and v > 0 else '—'

    def _col_style(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        for col in df.columns:
            try:
                k = cluster_labels.index(col)
                bg = cluster_colors[k % len(cluster_colors)]
                styles[col] = f'background-color: {bg}44; color: black'
            except (ValueError, IndexError):
                pass
        return styles

    print('Holiday × Cluster  (38-h profiles)')
    print('Rows = holiday name   |   Columns = cluster type (count of events)\n')
    ipy_display(
        pivot.style
        .apply(_col_style, axis=None)
        .format(_fmt_cell, na_rep='—')
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])
    )

    return pivot


def _build_analog_cluster_lookup(
    df_holiday_selector_features: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the per-day analog cluster lookup used for 38-h profiles."""
    return (
        df_holiday_selector_features[['date', 'holiday_name', 'analog_cluster']]
        .dropna(subset=['analog_cluster'])
        .assign(date=lambda frame: pd.to_datetime(frame['date']).dt.normalize())
        .drop_duplicates(subset=['date', 'holiday_name'])
    )


def _prepare_analog_cluster_38h_frames(
    df_phol: pd.DataFrame,
    analog_cluster_lookup: pd.DataFrame,
    analog_cluster_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join 38-h profiles with analog-cluster labels and build the day table."""
    df_phol_reset = df_phol.reset_index().copy()
    if 'date' not in df_phol_reset.columns:
        first_col = df_phol_reset.columns[0]
        df_phol_reset = df_phol_reset.rename(columns={first_col: 'date'})

    df_phol_fgh = (
        df_phol_reset
        .assign(date=lambda frame: pd.to_datetime(frame['date']).dt.normalize())
        .merge(analog_cluster_lookup, on=['date', 'holiday_name'], how='left')
    )
    df_phol_fgh = (
        df_phol_fgh[df_phol_fgh['analog_cluster'].isin(analog_cluster_labels)]
        .sort_values(['analog_cluster', 'date'])
        .set_index('date')
    )

    day_rows = [
        {
            'type': row['analog_cluster'],
            'date': date_value,
            'dow': date_value.day_name()[:3],
            'holiday_name': row['holiday_name'],
            'cluster_38h': row.get('cluster_type', pd.NA),
        }
        for date_value, row in df_phol_fgh.iterrows()
    ]
    df_days_fgh = pd.DataFrame(
        day_rows,
        columns=['type', 'date', 'dow', 'holiday_name', 'cluster_38h'],
    )
    df_days_fgh = (
        df_days_fgh.sort_values(['type', 'date'])
        .reset_index(drop=True)
    )

    return df_phol_fgh, df_days_fgh


def _display_analog_cluster_38h_days(
    df_days_fgh: pd.DataFrame,
    analog_cluster_labels: list[str],
    cluster_colors: list,
) -> None:
    """Display the per-day table for analog-cluster 38-h profiles."""
    from IPython.display import display

    def _day_bg_fgh(row):
        idx = analog_cluster_labels.index(row['type'])
        bg = cluster_colors[idx % len(cluster_colors)]
        return [f'background-color: {bg}44; color: black'] * len(row)

    print('\nDays per analog cluster')
    display(
        df_days_fgh.style
        .apply(_day_bg_fgh, axis=1)
        .format({'date': lambda d: d.strftime('%Y-%m-%d')})
        .hide(axis='index')
    )


def _build_analog_cluster_38h_suptitle(
    analog_cluster_labels: list[str],
    unique_id: str,
    selection_criterion: str | None = None,
) -> str:
    label_text = '/'.join(str(label) for label in analog_cluster_labels) or 'n/a'
    title = f'38-h event profiles by analog cluster ({label_text})  |  {unique_id}'
    if selection_criterion is not None and str(selection_criterion).strip():
        title += f'\nSelection criterion: {selection_criterion}'
    return title


def _plot_analog_cluster_38h_profiles(
    df_phol_fgh: pd.DataFrame,
    analog_cluster_labels: list[str],
    cluster_colors: list,
    unique_id: str,
    feat_cols: list,
    previously_w_hours: int,
    selection_criterion: str | None = None,
) -> np.ndarray:
    """Plot 38-h profiles grouped by analog-cluster label and return centroids."""
    import matplotlib.pyplot as plt

    x_axis = list(range(-previously_w_hours, 24))
    fig, axes = plt.subplots(
        1,
        len(analog_cluster_labels),
        figsize=(5.5 * len(analog_cluster_labels), 5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    fig.suptitle(
        _build_analog_cluster_38h_suptitle(
            analog_cluster_labels,
            unique_id,
            selection_criterion=selection_criterion,
        ),
        fontsize=13,
        y=1.02,
    )

    centroids_fgh = []
    for idx, (label, ax) in enumerate(zip(analog_cluster_labels, axes)):
        color = cluster_colors[idx % len(cluster_colors)]
        mask = df_phol_fgh['analog_cluster'] == label
        data = df_phol_fgh.loc[mask, feat_cols].values
        if len(data):
            centroid = data.mean(axis=0)
            centroids_fgh.append(centroid)
            for profile in data:
                ax.plot(x_axis, profile, color=color, alpha=0.30, lw=0.9)
            ax.plot(x_axis, centroid, color=color, lw=3,
                    label=f'Mean {label}', zorder=5)
        else:
            centroids_fgh.append(np.full(len(feat_cols), np.nan))
            ax.text(0.5, 0.5, 'No events', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11)

        ax.axvline(0, color='grey', ls='--', lw=0.9, label='midnight')
        ax.set_title(f'Analog {label}  (n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Hour relative to holiday start (h=0)', fontsize=10)
        ax.set_xticks(range(-previously_w_hours, 24, 4))
        if idx == 0:
            ax.set_ylabel(f'Demand [{unique_id}]', fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    return np.vstack(centroids_fgh)


def run_analog_cluster_38h_analysis(
    df_phol: pd.DataFrame,
    df_holiday_selector_features: pd.DataFrame,
    cluster_colors: list,
    unique_id: str,
    feat_cols: list,
    previously_w_hours: int = 14,
    cluster_labels: tuple[str, ...] = ('F', 'G', 'H'),
    selection_criterion: str | None = None,
) -> dict:
    """Group 38-h holiday profiles by stable analog-space labels."""
    required_selector_cols = {'date', 'holiday_name', 'analog_cluster'}
    missing_selector_cols = required_selector_cols - set(df_holiday_selector_features.columns)
    if missing_selector_cols:
        raise ValueError(
            'df_holiday_selector_features is missing required columns: '
            f'{sorted(missing_selector_cols)}'
        )

    missing_feat_cols = [col for col in feat_cols if col not in df_phol.columns]
    if missing_feat_cols:
        raise ValueError(f'df_phol is missing feature columns: {missing_feat_cols}')

    analog_cluster_labels = list(cluster_labels)
    analog_cluster_lookup = _build_analog_cluster_lookup(df_holiday_selector_features)
    df_phol_fgh, df_days_fgh = _prepare_analog_cluster_38h_frames(
        df_phol,
        analog_cluster_lookup,
        analog_cluster_labels,
    )

    if df_phol_fgh.empty:
        analog_label_text = '/'.join(str(label) for label in analog_cluster_labels) or 'n/a'
        print(f'No 38-h profiles were matched to analog clusters {analog_label_text}.')
        centroids_fgh = np.empty((0, len(feat_cols)))
    else:
        _display_analog_cluster_38h_days(
            df_days_fgh,
            analog_cluster_labels,
            cluster_colors,
        )
        centroids_fgh = _plot_analog_cluster_38h_profiles(
            df_phol_fgh,
            analog_cluster_labels,
            cluster_colors,
            unique_id,
            feat_cols,
            previously_w_hours,
            selection_criterion=selection_criterion,
        )

    print(f'Analog-cluster profiles available: {len(df_phol_fgh)}')
    return {
        'df_phol_fgh': df_phol_fgh,
        'df_days_fgh': df_days_fgh,
        'centroids_fgh': centroids_fgh,
        'feat_cols': feat_cols,
    }


_SELECTOR_DOW_NAME_MAP = {
    'Mon': 'Monday',
    'Tue': 'Tuesday',
    'Wed': 'Wednesday',
    'Thu': 'Thursday',
    'Fri': 'Friday',
    'Sat': 'Saturday',
    'Sun': 'Sunday',
}

_SELECTOR_DAY_CLASS_MAP = {
    1: 'Weekday',
    2: 'Saturday',
    3: 'Sunday',
}

_POST_HOLIDAY_RECOVERY_LABEL = 'Post-holiday recovery'

_SELECTOR_SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
}

_SELECTOR_PROFILE_COLUMNS = [
    'best_matching_weekday',
    'daily_profile_cluster',
    'daily_profile_cluster_id',
    'daily_profile_archetype',
    'event_profile_cluster',
    'event_profile_cluster_id',
]

_SELECTOR_FEATURE_COLUMNS = [
    'unique_id',
    'holiday_name',
    'anchor_holiday_name',
    'date',
    'holiday_day_type',
    'weekday_name',
    'day_class_code',
    'day_class_name',
    'season',
    'date_rule',
    'is_fixed_date',
    'is_observed_monday_rule',
    *_SELECTOR_PROFILE_COLUMNS,
]


def _resolve_selector_group_cols(
    group_cols: tuple[str, ...],
    *column_sources,
) -> tuple[str, ...]:
    """Keep selector grouping series-aware when ``unique_id`` is available."""
    resolved_group_cols = tuple(group_cols)
    if 'unique_id' in resolved_group_cols:
        return resolved_group_cols

    if column_sources and all('unique_id' in set(source) for source in column_sources):
        return ('unique_id', *resolved_group_cols)

    return resolved_group_cols


def _load_selector_holiday_metadata(holidays_path: Path | str | None) -> dict[str, dict]:
    """Return a holiday-name metadata map from the JSON catalog."""
    if holidays_path is None:
        return {}

    with open(Path(holidays_path), 'r', encoding='utf-8') as file_obj:
        payload = json.load(file_obj)

    return {
        _holiday_display_name(holiday): holiday
        for holiday in payload.get('holidays', [])
    }


def _label_special_day_types(
    special_dates: list[pd.Timestamp],
    date_to_name: dict[pd.Timestamp, str],
) -> dict[pd.Timestamp, str]:
    """Classify special dates as H1, H2, or H3 using the project taxonomy."""
    labels: dict[pd.Timestamp, str] = {}
    special_set = set(special_dates)

    for date_value in special_dates:
        prev_date = date_value - pd.Timedelta(days=1)
        next_date = date_value + pd.Timedelta(days=1)
        if next_date in special_set and prev_date not in special_set:
            labels[date_value] = 'H1'

    for date_value in special_dates:
        if date_value in labels:
            continue

        prev_date = date_value - pd.Timedelta(days=1)
        prev_name = date_to_name.get(prev_date)

        if (
            date_value.month == 1
            and date_value.day == 1
            and prev_name == "New Year's Eve"
        ):
            labels[date_value] = 'H3'
            continue

        if prev_date in special_set and labels.get(prev_date) != 'H1':
            labels[date_value] = 'H3'

    for date_value in special_dates:
        labels.setdefault(date_value, 'H2')

    return labels


def _build_selector_base_rows(
    df_holidays: pd.DataFrame,
    available_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    """Create the base selector table, including derived H4 recovery days."""
    df_base = df_holidays[['date', 'holiday_name']].copy()
    df_base['date'] = pd.to_datetime(df_base['date']).dt.normalize()
    df_base = (
        df_base[df_base['date'].isin(available_dates)]
        .drop_duplicates(subset=['date'])
        .sort_values('date')
        .reset_index(drop=True)
    )

    if df_base.empty:
        return pd.DataFrame(
            columns=['holiday_name', 'anchor_holiday_name', 'date', 'holiday_day_type']
        )

    special_dates = df_base['date'].tolist()
    date_to_name = dict(zip(df_base['date'], df_base['holiday_name']))
    day_type_map = _label_special_day_types(special_dates, date_to_name)
    special_set = set(special_dates)

    rows = [
        {
            'holiday_name': row['holiday_name'],
            'anchor_holiday_name': row['holiday_name'],
            'date': row['date'],
            'holiday_day_type': day_type_map[row['date']],
        }
        for _, row in df_base.iterrows()
    ]

    idx = 0
    while idx < len(special_dates):
        run_start = run_end = special_dates[idx]
        jdx = idx + 1
        while jdx < len(special_dates) and special_dates[jdx] == run_end + pd.Timedelta(days=1):
            run_end = special_dates[jdx]
            jdx += 1

        run_length = (run_end - run_start).days + 1
        if run_length >= 2:
            candidate = run_end + pd.Timedelta(days=1)
            if candidate not in special_set and candidate in available_dates:
                rows.append({
                    'holiday_name': _POST_HOLIDAY_RECOVERY_LABEL,
                    'anchor_holiday_name': date_to_name.get(run_end, _POST_HOLIDAY_RECOVERY_LABEL),
                    'date': candidate,
                    'holiday_day_type': 'H4',
                })

        idx = jdx

    return pd.DataFrame(rows).sort_values(['date', 'holiday_name']).reset_index(drop=True)


def _selector_date_rule(
    holiday_name: str,
    holiday_date: pd.Timestamp,
    holiday_metadata: dict[str, dict],
) -> tuple[str, bool, bool]:
    """Return the observance-rule fields for the selector table."""
    if holiday_name == _POST_HOLIDAY_RECOVERY_LABEL:
        return 'derived_recovery_day', False, False

    metadata = holiday_metadata.get(holiday_name, {})
    observed_rule = metadata.get('observed_rule')
    date_type = metadata.get('date_type')

    if observed_rule and holiday_date.year >= _LFT_REFORM_YEAR:
        return 'observed_monday_rule', False, True

    if observed_rule and holiday_date.year < _LFT_REFORM_YEAR:
        return 'fixed_date', True, False

    if date_type == 'fixed':
        return 'fixed_date', True, False

    if date_type == 'movable':
        return 'movable_date', False, False

    if date_type == 'relative_weekday':
        return 'relative_weekday_rule', False, False

    if date_type == 'relative_to_holiday':
        return 'relative_to_holiday_rule', False, False

    return 'unknown', False, False


def _selector_cluster_archetype(label: str | None) -> str | None:
    """Expand a short DOW label such as Sat into a readable archetype."""
    if label is None or pd.isna(label):
        return None
    if label == 'unclear':
        return 'unclear'

    weekday_name = _SELECTOR_DOW_NAME_MAP.get(str(label), str(label))
    return f'{weekday_name}-like'


def _selector_day_class_code(day_of_week: int) -> int:
    """Return 1 for weekdays, 2 for Saturday, and 3 for Sunday."""
    if day_of_week == 5:
        return 2
    if day_of_week == 6:
        return 3
    return 1


def _empty_selector_feature_frame() -> pd.DataFrame:
    """Return an empty selector feature table with the exported schema."""
    return pd.DataFrame(columns=_SELECTOR_FEATURE_COLUMNS)


def _finalize_selector_feature_frame(
    selector_df: pd.DataFrame,
    holiday_metadata: dict[str, dict],
) -> pd.DataFrame:
    """Add calendar fields and enforce the selector export schema."""
    if selector_df.empty:
        return _empty_selector_feature_frame()

    selector_df = selector_df.copy()
    selector_df['date'] = pd.to_datetime(selector_df['date']).dt.normalize()

    if 'unique_id' not in selector_df.columns:
        selector_df['unique_id'] = pd.NA

    for col_name in _SELECTOR_PROFILE_COLUMNS:
        if col_name not in selector_df.columns:
            selector_df[col_name] = pd.NA

    selector_df['weekday_name'] = selector_df['date'].dt.day_name()
    selector_df['day_class_code'] = selector_df['date'].dt.dayofweek.map(_selector_day_class_code)
    selector_df['day_class_name'] = selector_df['day_class_code'].map(_SELECTOR_DAY_CLASS_MAP)
    selector_df['season'] = selector_df['date'].dt.month.map(_SELECTOR_SEASON_MAP)

    rule_fields = selector_df.apply(
        lambda row: _selector_date_rule(
            row['holiday_name'],
            row['date'],
            holiday_metadata,
        ),
        axis=1,
        result_type='expand',
    )
    rule_fields.columns = ['date_rule', 'is_fixed_date', 'is_observed_monday_rule']
    selector_df = pd.concat([selector_df, rule_fields], axis=1)

    return selector_df[_SELECTOR_FEATURE_COLUMNS].sort_values(['date', 'holiday_name']).reset_index(drop=True)


def build_holiday_selector_features(
    df_wide: pd.DataFrame,
    df_holidays: pd.DataFrame,
    cluster_ab_results: dict,
    df_phol: pd.DataFrame,
    holidays_path: Path | str | None = None,
    unique_id: str | None = None,
) -> pd.DataFrame:
    """Build a selector-ready holiday feature table for the analog workflow.

    The output combines:
      - H1/H2/H3/H4 day typing
      - calendar season
      - weekday-vs-weekend code (1=Weekday, 2=Saturday, 3=Sunday)
      - holiday observance rule (`fixed_date`, `observed_monday_rule`, `movable_date`)
      - per-day best matching weekday from the 8d validation table
      - daily-profile cluster labels from 8c (A, B, ...)
      - 38-h event-profile cluster labels from 8c-bis (C, D, E, ...)

    Parameters
    ----------
    df_wide:
        Wide-format daily DataFrame whose index defines the dates available
        in the notebook.
    df_holidays:
        Holiday catalog DataFrame with at least `date` and `holiday_name`.
    cluster_ab_results:
        Output dict from `run_cluster_ab_validation`; must contain
        `occurrences_df` and `summary_df`.
    df_phol:
        Output DataFrame from `run_cluster_38h_analysis`.
    holidays_path:
        Optional path to `holidays_recognized.json` to recover the original
        rule metadata behind each holiday.

    Returns
    -------
    pd.DataFrame
        Selector-ready feature table, one row per holiday/special day.
    """
    available_dates = set(pd.to_datetime(df_wide.index).normalize())
    selector_df = _build_selector_base_rows(df_holidays, available_dates)
    if unique_id is not None and not selector_df.empty:
        selector_df['unique_id'] = str(unique_id)

    if selector_df.empty:
        return _empty_selector_feature_frame()

    holiday_metadata = _load_selector_holiday_metadata(holidays_path)

    occurrences_df = cluster_ab_results.get('occurrences_df', pd.DataFrame()).copy()
    summary_df = cluster_ab_results.get('summary_df', pd.DataFrame()).copy()

    if not occurrences_df.empty:
        occurrences_df['date'] = pd.to_datetime(occurrences_df['date']).dt.normalize()
        cluster_ids = sorted(pd.Series(occurrences_df['cluster']).dropna().astype(int).unique())
        cluster_letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        daily_cluster_map = {
            cluster_id: cluster_letters[idx]
            for idx, cluster_id in enumerate(cluster_ids)
        }
        summary_cluster_type_map = {}
        if not summary_df.empty:
            summary_cluster_type_map = dict(
                zip(summary_df['cluster'], summary_df['cluster_type'])
            )

        daily_features = (
            occurrences_df[['date', 'cluster', 'best_dow']]
            .drop_duplicates(subset=['date'])
            .rename(columns={'cluster': 'daily_profile_cluster_id'})
        )
        daily_features['daily_profile_cluster_id'] = daily_features['daily_profile_cluster_id'].astype('Int64')
        daily_features['daily_profile_cluster'] = daily_features['daily_profile_cluster_id'].map(daily_cluster_map)
        daily_features['best_matching_weekday'] = daily_features['best_dow'].map(
            _SELECTOR_DOW_NAME_MAP
        ).fillna(daily_features['best_dow'])
        daily_features['daily_profile_archetype'] = daily_features['daily_profile_cluster_id'].map(
            summary_cluster_type_map
        ).map(_selector_cluster_archetype)
        daily_features = daily_features[['date', *_SELECTOR_PROFILE_COLUMNS[:4]]]
    else:
        daily_features = pd.DataFrame(
            columns=['date', *_SELECTOR_PROFILE_COLUMNS[:4]]
        )

    if not df_phol.empty:
        event_features = (
            df_phol.reset_index()[['date', 'cluster', 'cluster_type']]
            .copy()
            .rename(columns={
                'cluster': 'event_profile_cluster_id',
                'cluster_type': 'event_profile_cluster',
            })
        )
        event_features['date'] = pd.to_datetime(event_features['date']).dt.normalize()
        event_features['event_profile_cluster_id'] = event_features['event_profile_cluster_id'].astype('Int64')
        event_features = event_features.drop_duplicates(subset=['date'])
    else:
        event_features = pd.DataFrame(
            columns=['date', 'event_profile_cluster', 'event_profile_cluster_id']
        )

    selector_df = selector_df.merge(daily_features, on='date', how='left')
    selector_df = selector_df.merge(event_features, on='date', how='left')

    return _finalize_selector_feature_frame(selector_df, holiday_metadata)

def _selector_modal_value(values: pd.Series):
    """Return the dominant non-null value of a selector column."""
    valid = values.dropna()
    if valid.empty:
        return pd.NA

    modes = valid.mode(dropna=True)
    if len(modes) == 0:
        return pd.NA

    return modes.iloc[0]


_ANALOG_CLUSTER_LABELS = list('FGHIJKLMNOPQRSTUVWXYZ')


def _selector_group_modal_frame(
    df_selector: pd.DataFrame,
    group_cols: tuple[str, ...],
    value_cols: list[str],
) -> pd.DataFrame:
    """Return the modal value of each selector column per grouping key."""
    grouped = df_selector.groupby(list(group_cols), dropna=False, sort=True)
    rows = []

    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {col_name: group_key[idx] for idx, col_name in enumerate(group_cols)}
        for col_name in value_cols:
            row[col_name] = _selector_modal_value(group_df[col_name])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[*group_cols, *value_cols])

    return pd.DataFrame(rows)


def _build_selector_prior_row(
    group_key: tuple,
    group_cols: tuple[str, ...],
    group_df: pd.DataFrame,
    inferred_cols: list[str],
    anchor_fallback_lookup: dict[object, dict],
) -> dict:
    """Build a single prior row with anchor-holiday fallback when needed."""
    row = {col_name: group_key[idx] for idx, col_name in enumerate(group_cols)}
    row['history_rows'] = int(len(group_df))
    row['history_years'] = int(group_df['date'].dt.year.nunique()) if 'date' in group_df else pd.NA
    row['prior_resolution_scope'] = 'family_day_type'

    for col_name in inferred_cols:
        value = _selector_modal_value(group_df[col_name])
        if pd.isna(value):
            anchor_name = row.get('anchor_holiday_name')
            anchor_lookup_key = (
                (row.get('unique_id'), anchor_name)
                if 'unique_id' in row else anchor_name
            )
            value = anchor_fallback_lookup.get(anchor_lookup_key, {}).get(col_name, pd.NA)
            if pd.notna(value):
                row['prior_resolution_scope'] = 'anchor_holiday_name'

        row[f'inferred_{col_name}'] = value

    unresolved = any(pd.isna(row[f'inferred_{col_name}']) for col_name in inferred_cols)
    if unresolved:
        row['prior_resolution_scope'] = 'partially_unresolved'

    return row


def build_holiday_selector_priors(
    df_selector: pd.DataFrame,
    group_cols: tuple[str, ...] = ('anchor_holiday_name', 'holiday_day_type'),
) -> pd.DataFrame:
    """Infer ex-ante selector labels from historical holiday families.

    For a future candidate whose profile-based labels are not yet observed,
    this helper assigns a prior label by taking the dominant historical value
    within the same holiday family. By default, the family is defined by the
    pair `(anchor_holiday_name, holiday_day_type)`.

    Parameters
    ----------
    df_selector:
        Selector table produced by ``build_holiday_selector_features``.
    group_cols:
        Columns that define the holiday family used to compute the prior.

    Returns
    -------
    pd.DataFrame
        One row per holiday family with inferred labels and support counts.
    """
    resolved_group_cols = _resolve_selector_group_cols(group_cols, df_selector.columns)

    required_cols = set(resolved_group_cols)
    missing_cols = required_cols - set(df_selector.columns)
    if missing_cols:
        raise ValueError(f'Missing required selector columns: {sorted(missing_cols)}')

    inferred_cols = [
        'best_matching_weekday',
        'daily_profile_cluster',
        'daily_profile_cluster_id',
        'daily_profile_archetype',
        'event_profile_cluster',
        'event_profile_cluster_id',
    ]

    anchor_fallback_lookup: dict[str, dict] = {}
    if 'anchor_holiday_name' in df_selector.columns:
        anchor_group_cols = (
            ('unique_id', 'anchor_holiday_name')
            if 'unique_id' in df_selector.columns else ('anchor_holiday_name',)
        )
        anchor_modal_df = _selector_group_modal_frame(
            df_selector,
            anchor_group_cols,
            inferred_cols,
        )
        anchor_fallback_lookup = {
            (
                (row['unique_id'], row['anchor_holiday_name'])
                if 'unique_id' in anchor_group_cols else row['anchor_holiday_name']
            ): {
                col_name: row[col_name]
                for col_name in inferred_cols
            }
            for _, row in anchor_modal_df.iterrows()
        }

    summary_rows = []
    grouped = df_selector.groupby(list(resolved_group_cols), dropna=False, sort=True)

    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        summary_rows.append(
            _build_selector_prior_row(
                group_key=group_key,
                group_cols=resolved_group_cols,
                group_df=group_df,
                inferred_cols=inferred_cols,
                anchor_fallback_lookup=anchor_fallback_lookup,
            )
        )

    column_order = list(resolved_group_cols) + [
        'history_rows',
        'history_years',
        'inferred_best_matching_weekday',
        'inferred_daily_profile_cluster',
        'inferred_daily_profile_cluster_id',
        'inferred_daily_profile_archetype',
        'inferred_event_profile_cluster',
        'inferred_event_profile_cluster_id',
    ]

    return pd.DataFrame(summary_rows)[column_order].sort_values(list(resolved_group_cols)).reset_index(drop=True)


def build_future_holiday_selector_features(
    df_holidays: pd.DataFrame,
    df_priors: pd.DataFrame,
    available_dates,
    holidays_path: Path | str | None = None,
    group_cols: tuple[str, ...] = ('anchor_holiday_name', 'holiday_day_type'),
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    unique_id: str | None = None,
) -> pd.DataFrame:
    """Build ex-ante selector rows for future holidays using historical priors.

    This helper is intended for forecast horizons already present in the hourly
    source table but excluded from the historical clustering window. It creates
    the future H1/H2/H3/H4 rows from the holiday calendar, then fills the
    profile-related fields from ``df_priors`` so the export stays leakage-free.
    """
    resolved_group_cols = _resolve_selector_group_cols(group_cols, df_priors.columns)

    required_cols = set(resolved_group_cols)
    missing_cols = required_cols - set(df_priors.columns)
    if missing_cols:
        raise ValueError(f'Missing required prior columns: {sorted(missing_cols)}')

    inferred_map = {
        'best_matching_weekday': 'inferred_best_matching_weekday',
        'daily_profile_cluster': 'inferred_daily_profile_cluster',
        'daily_profile_cluster_id': 'inferred_daily_profile_cluster_id',
        'daily_profile_archetype': 'inferred_daily_profile_archetype',
        'event_profile_cluster': 'inferred_event_profile_cluster',
        'event_profile_cluster_id': 'inferred_event_profile_cluster_id',
    }
    missing_inferred = set(inferred_map.values()) - set(df_priors.columns)
    if missing_inferred:
        raise ValueError(f'Missing required inferred prior columns: {sorted(missing_inferred)}')

    selector_df = _build_selector_base_rows(df_holidays, _normalize_date_set(available_dates))
    if selector_df.empty:
        return _empty_selector_feature_frame()

    if 'unique_id' in resolved_group_cols:
        if unique_id is None:
            raise ValueError('unique_id is required when selector priors are series-specific.')
        selector_df['unique_id'] = str(unique_id)

    if start_date is not None:
        selector_df = selector_df[
            selector_df['date'] >= pd.Timestamp(start_date).normalize()
        ].copy()
    if end_date is not None:
        selector_df = selector_df[
            selector_df['date'] <= pd.Timestamp(end_date).normalize()
        ].copy()
    if selector_df.empty:
        return _empty_selector_feature_frame()

    prior_cols = list(resolved_group_cols) + list(inferred_map.values())
    if 'prior_resolution_scope' in df_priors.columns:
        prior_cols.append('prior_resolution_scope')

    selector_df = selector_df.merge(
        df_priors[prior_cols].drop_duplicates(subset=list(resolved_group_cols)),
        on=list(resolved_group_cols),
        how='left',
    )
    for target_col, source_col in inferred_map.items():
        selector_df[target_col] = selector_df[source_col]

    holiday_metadata = _load_selector_holiday_metadata(holidays_path)
    return _finalize_selector_feature_frame(selector_df, holiday_metadata)


def _resolve_analog_criterion_columns(criterion: str) -> tuple[str, str]:
    """Return selector/prior columns used by the analog-cluster criterion."""
    criterion_spec = _resolve_analog_criterion_spec(criterion)
    return criterion_spec.get('selector_col'), criterion_spec.get('prior_col')


_ANALOG_CRITERION_ALIASES = {
    'seasonal_winter_sprint_fall': 'seasonal_winter_spring_fall',
}

ANALOG_CLUSTER_CRITERIA_CATALOG = {
    'shape_pearson_CDE_map_FGH': (
        'Maps the current 38h shape-based event_profile_cluster letters '
        'C/D/E/... to stable analog labels F/G/H/...'
    ),
    'seasonal_heat_cold': (
        'Groups holiday dates into heat vs cold seasons, mapping '
        'Spring/Summer to heat and Autumn/Winter to cold.'
    ),
    'seasonal_winter_sprint_fall': (
        'Groups holiday dates by year season and maps Winter/Spring/Fall/Summer '
        'to stable analog labels.'
    ),
    'best_matching_weekday': (
        'Groups holiday dates by best_matching_weekday, the closest weekday-profile '
        'label assigned to each date.'
    ),
    'observance_tier': (
        'Groups holiday dates by how inhábil they actually are (observance tier): '
        'working (barely a holiday, e.g. Labor Day), partial (industry/commerce '
        'partially active, e.g. Holy Saturday) and full (fully observed, e.g. '
        'Christmas Day). Tiers derived from the observed_strength diagnostic.'
    ),
    'observance_tier_depth': (
        'Four-tier refinement of observance_tier that splits the fully-observed group '
        'by demand-drop depth: working, partial, full_civic (shallow ~15% drop, e.g. '
        'Independence/Constitution) and full_deep (~30% drop, e.g. Christmas/New Year). '
        'Separates patriotic civic holidays from the deep Dec-Jan winter holidays.'
    ),
    'holiday_identity': (
        'Pure Similar-Days hard filter: every distinct holiday (by anchor name) '
        'becomes its own analog cluster, so the downstream selector only matches '
        'a target against other instances of the SAME holiday. Tests the calendar-'
        'identity filter against the shape-based analog search.'
    ),
}

_SEASONAL_ANALOG_VALUE_ALIASES = {
    'winter': 'winter',
    'spring': 'spring',
    'sprint': 'spring',
    'fall': 'fall',
    'autumn': 'fall',
    'summer': 'summer',
}

_HEAT_COLD_SEASON_MAP = {
    'spring': 'heat',
    'summer': 'heat',
    'fall': 'cold',
    'winter': 'cold',
}

_SHAPE_PEARSON_ANALOG_ORDER = tuple('CDEFGHIJKLMNOPQRSTUVWXYZ')
_SEASONAL_ANALOG_ORDER = ('winter', 'spring', 'fall', 'summer')
_SEASONAL_HEAT_COLD_ORDER = ('heat', 'cold')

# Observance tier per anchor holiday. Derived from the observed_strength diagnostic
# (shared/observed_strength.py): median observed_strength across regions/years, i.e.
# actual_demand_drop / typical_demand_drop. Thresholds: <0.55 -> working,
# 0.55-0.80 -> partial, >=0.80 -> full. The strength shown in the comment is the
# per-anchor median that placed it in its tier.
_OBSERVANCE_TIER_BY_ANCHOR = {
    'labor day': 'working',              # 0.47 -- least inhábil
    'mexican revolution day': 'working',  # 0.54
    'holy saturday': 'partial',          # 0.57
    'maundy thursday': 'partial',        # 0.63
    'good friday': 'partial',            # 0.70
    'christmas eve': 'partial',          # 0.71
    "new year's eve": 'partial',         # 0.77
    'christmas day': 'full',             # 0.85
    'constitution day': 'full',          # 0.88
    "new year's day": 'full',            # 0.94
    "benito juarez's birthday": 'full',  # 0.97
    'independence day': 'full',          # 0.99 -- fully observed
}
_OBSERVANCE_TIER_ORDER = ('working', 'partial', 'full')

# Four-tier refinement: split the fully-observed anchors by demand-drop DEPTH (typical
# drop = how far below a normal weekday the holiday lands, from the observed_strength
# diagnostic). Civic patriotic holidays drop ~15% (Independence 14.3%, Juarez 15.4%,
# Constitution 16.7%); the Dec-Jan winter holidays drop ~30% (Christmas Day 30.8%,
# New Year's Day 30.2%). Working/partial are unchanged from observance_tier.
_OBSERVANCE_TIER_DEPTH_BY_ANCHOR = {
    'labor day': 'working',
    'mexican revolution day': 'working',
    'holy saturday': 'partial',
    'maundy thursday': 'partial',
    'good friday': 'partial',
    'christmas eve': 'partial',
    "new year's eve": 'partial',
    'constitution day': 'full_civic',     # ~16.7% drop
    "benito juarez's birthday": 'full_civic',  # ~15.4%
    'independence day': 'full_civic',     # ~14.3%
    'christmas day': 'full_deep',         # ~30.8%
    "new year's day": 'full_deep',        # ~30.2%
}
_OBSERVANCE_TIER_DEPTH_ORDER = ('working', 'partial', 'full_civic', 'full_deep')


def _normalize_analog_criterion_name(criterion: str) -> str:
    criterion_text = str(criterion).strip()
    return _ANALOG_CRITERION_ALIASES.get(criterion_text, criterion_text)


def _derive_seasonal_analog_criterion_value(row: pd.Series) -> object:
    season_value = row.get('season', pd.NA)
    if pd.notna(season_value):
        season_text = str(season_value).strip().lower()
        normalized = _SEASONAL_ANALOG_VALUE_ALIASES.get(season_text)
        if normalized is not None:
            return normalized

    date_value = pd.to_datetime(row.get('date', pd.NaT), errors='coerce')
    if pd.notna(date_value):
        month_value = int(date_value.month)
        if month_value in (12, 1, 2):
            return 'winter'
        if month_value in (3, 4, 5):
            return 'spring'
        if month_value in (6, 7, 8):
            return 'summer'
        return 'fall'

    return pd.NA


def _derive_seasonal_heat_cold_analog_criterion_value(row: pd.Series) -> object:
    seasonal_value = _derive_seasonal_analog_criterion_value(row)
    if pd.isna(seasonal_value):
        return pd.NA
    return _HEAT_COLD_SEASON_MAP.get(str(seasonal_value).strip().lower(), pd.NA)


def _derive_observance_tier_analog_criterion_value(row: pd.Series) -> object:
    """Map a holiday's anchor name to its observance tier (working/partial/full)."""
    anchor_value = row.get('anchor_holiday_name', pd.NA)
    if pd.isna(anchor_value):
        anchor_value = row.get('holiday_name', pd.NA)
    if pd.isna(anchor_value):
        return pd.NA
    anchor_text = str(anchor_value).strip().lower()
    return _OBSERVANCE_TIER_BY_ANCHOR.get(anchor_text, pd.NA)


def _derive_observance_tier_depth_analog_criterion_value(row: pd.Series) -> object:
    """Map a holiday's anchor name to its depth-aware observance tier."""
    anchor_value = row.get('anchor_holiday_name', pd.NA)
    if pd.isna(anchor_value):
        anchor_value = row.get('holiday_name', pd.NA)
    if pd.isna(anchor_value):
        return pd.NA
    anchor_text = str(anchor_value).strip().lower()
    return _OBSERVANCE_TIER_DEPTH_BY_ANCHOR.get(anchor_text, pd.NA)


def _derive_holiday_identity_analog_criterion_value(row: pd.Series) -> object:
    """Use the holiday's own anchor name as its analog-cluster group.

    This is the pure ``Similar Days`` hard filter: every distinct holiday
    (Independence Day, Memorial Day, MLK Day, ...) becomes its own analog
    cluster, so the downstream selector only ever matches a target against
    other instances of the *same* holiday. Day-after (H4) rows inherit their
    anchor, so e.g. Christmas Day H2 and H4 share one cluster.
    """
    anchor_value = row.get('anchor_holiday_name', pd.NA)
    if pd.isna(anchor_value):
        anchor_value = row.get('holiday_name', pd.NA)
    if pd.isna(anchor_value):
        return pd.NA
    return str(anchor_value).strip()


def _resolve_analog_criterion_spec(criterion: str) -> dict:
    normalized_criterion = _normalize_analog_criterion_name(criterion)
    criterion_map = {
        'event_profile_cluster': {
            'selector_col': 'event_profile_cluster',
            'prior_col': 'inferred_event_profile_cluster',
        },
        'daily_profile_cluster': {
            'selector_col': 'daily_profile_cluster',
            'prior_col': 'inferred_daily_profile_cluster',
        },
        'best_matching_weekday': {
            'selector_col': 'best_matching_weekday',
            'prior_col': 'inferred_best_matching_weekday',
        },
        'daily_profile_archetype': {
            'selector_col': 'daily_profile_archetype',
            'prior_col': 'inferred_daily_profile_archetype',
        },
        'shape_pearson_CDE_map_FGH': {
            'selector_col': 'event_profile_cluster',
            'prior_col': 'inferred_event_profile_cluster',
            'ordered_values': _SHAPE_PEARSON_ANALOG_ORDER,
        },
        'seasonal_heat_cold': {
            'selector_col': None,
            'prior_col': None,
            'ordered_values': _SEASONAL_HEAT_COLD_ORDER,
            'value_getter': _derive_seasonal_heat_cold_analog_criterion_value,
        },
        'seasonal_winter_spring_fall': {
            'selector_col': None,
            'prior_col': None,
            'ordered_values': _SEASONAL_ANALOG_ORDER,
            'value_getter': _derive_seasonal_analog_criterion_value,
        },
        'observance_tier': {
            'selector_col': None,
            'prior_col': None,
            'ordered_values': _OBSERVANCE_TIER_ORDER,
            'value_getter': _derive_observance_tier_analog_criterion_value,
        },
        'observance_tier_depth': {
            'selector_col': None,
            'prior_col': None,
            'ordered_values': _OBSERVANCE_TIER_DEPTH_ORDER,
            'value_getter': _derive_observance_tier_depth_analog_criterion_value,
        },
        'holiday_identity': {
            'selector_col': None,
            'prior_col': None,
            # No fixed order: groups are alphabetised by anchor holiday name.
            'value_getter': _derive_holiday_identity_analog_criterion_value,
        },
    }

    try:
        return criterion_map[normalized_criterion].copy()
    except KeyError as exc:
        valid = ', '.join(sorted(set(criterion_map) | set(_ANALOG_CRITERION_ALIASES)))
        raise ValueError(f'Unsupported analog criterion {criterion!r}. Options: {valid}') from exc


def _resolve_analog_criterion_values(
    df_values: pd.DataFrame,
    criterion: str,
) -> pd.Series:
    criterion_spec = _resolve_analog_criterion_spec(criterion)
    value_getter = criterion_spec.get('value_getter')
    if value_getter is not None:
        return df_values.apply(value_getter, axis=1)

    selector_col = criterion_spec.get('selector_col')
    prior_col = criterion_spec.get('prior_col')

    resolved_values = pd.Series(pd.NA, index=df_values.index, dtype='object')
    if selector_col is not None and selector_col in df_values.columns:
        resolved_values = df_values[selector_col].copy()
    if prior_col is not None and prior_col in df_values.columns:
        resolved_values = resolved_values.where(resolved_values.notna(), df_values[prior_col])
    return resolved_values


def _order_analog_criterion_frame(
    df_values: pd.DataFrame,
    criterion: str,
) -> pd.DataFrame:
    ordered_values_df = df_values.copy()
    if ordered_values_df.empty or 'analog_criterion_value' not in ordered_values_df.columns:
        return ordered_values_df

    criterion_spec = _resolve_analog_criterion_spec(criterion)
    ordered_values = tuple(criterion_spec.get('ordered_values') or ())
    if not ordered_values:
        return ordered_values_df.sort_values(
            'analog_criterion_value',
            key=lambda values: values.astype(str),
        ).reset_index(drop=True)

    order_map = {value: idx for idx, value in enumerate(ordered_values)}
    ordered_values_df['_analog_order'] = ordered_values_df['analog_criterion_value'].map(order_map)

    missing_mask = ordered_values_df['_analog_order'].isna()
    if missing_mask.any():
        extra_values = (
            ordered_values_df.loc[missing_mask, 'analog_criterion_value']
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        extra_order = {value: len(order_map) + idx for idx, value in enumerate(extra_values)}
        ordered_values_df.loc[missing_mask, '_analog_order'] = (
            ordered_values_df.loc[missing_mask, 'analog_criterion_value']
            .astype(str)
            .map(extra_order)
        )

    return (
        ordered_values_df
        .sort_values('_analog_order')
        .drop(columns=['_analog_order'])
        .reset_index(drop=True)
    )


def _build_analog_cluster_catalog(
    df_values: pd.DataFrame,
    criterion: str,
    cluster_labels: tuple[str, ...],
) -> pd.DataFrame:
    catalog = _order_analog_criterion_frame(df_values, criterion)
    catalog = catalog.drop(columns=['analog_cluster', 'analog_criterion'], errors='ignore').copy()

    label_pool = list(cluster_labels)
    if len(catalog) > len(label_pool):
        label_pool.extend(_ANALOG_CLUSTER_LABELS[len(label_pool):len(catalog)])
    if len(catalog) > len(label_pool):
        raise ValueError('Not enough analog-cluster labels available for the resolved criterion groups.')

    catalog.insert(0, 'analog_cluster', label_pool[: len(catalog)])
    catalog.insert(1, 'analog_criterion', criterion)
    return catalog


def assign_holiday_selector_analog_clusters(
    df_selector: pd.DataFrame,
    df_priors: pd.DataFrame,
    criterion: str = 'event_profile_cluster',
    group_cols: tuple[str, ...] = ('anchor_holiday_name', 'holiday_day_type'),
    cluster_labels: tuple[str, ...] = ('F', 'G', 'H'),
) -> dict:
    """Assign stable analog-cluster labels to historical selector rows.

    The internal criterion can change over time (`event_profile_cluster`,
    `daily_profile_cluster`, etc.), but the output analog-space labels remain
    stable and human-facing (`F`, `G`, `H`, ...).

    Parameters
    ----------
    df_selector:
        Historical selector table.
    df_priors:
        Output of ``build_holiday_selector_priors``.
    criterion:
        Column family used to define the analog-space grouping.
    group_cols:
        Key used to merge priors back into the selector table.
    cluster_labels:
        Stable output labels. If there are more unique criterion groups than
        provided labels, labels continue automatically as `I`, `J`, ...

    Returns
    -------
    dict with keys:
        ``df_selector_clusters``   – selector rows with analog-cluster columns
        ``analog_cluster_catalog`` – mapping from internal criterion value to
                                     stable analog label
    """
    resolved_group_cols = _resolve_selector_group_cols(
        group_cols,
        df_selector.columns,
        df_priors.columns,
    )
    selector_col, prior_col = _resolve_analog_criterion_columns(criterion)

    missing_selector = set(resolved_group_cols) - set(df_selector.columns)
    missing_priors = set(resolved_group_cols) - set(df_priors.columns)
    if missing_selector:
        raise ValueError(f'Missing selector columns: {sorted(missing_selector)}')
    if missing_priors:
        raise ValueError(f'Missing prior columns: {sorted(missing_priors)}')

    criterion_spec = _resolve_analog_criterion_spec(criterion)
    selector_col = criterion_spec.get('selector_col')
    prior_col = criterion_spec.get('prior_col')
    if prior_col is not None and prior_col not in df_priors.columns:
        raise ValueError(
            f'Missing prior criterion column {prior_col!r} for analog criterion {criterion!r}.'
        )

    df_clusters = df_selector.copy()
    for col_name in [prior_col, 'analog_criterion', 'analog_criterion_value', 'analog_cluster']:
        if col_name is not None and col_name in df_clusters.columns:
            df_clusters = df_clusters.drop(columns=[col_name])

    if prior_col is not None:
        merge_cols = list(resolved_group_cols) + [prior_col]
        df_clusters = df_clusters.merge(
            df_priors[merge_cols].drop_duplicates(subset=list(resolved_group_cols)),
            on=list(resolved_group_cols),
            how='left',
        )

    resolved_values = _resolve_analog_criterion_values(df_clusters, criterion)
    df_clusters['analog_criterion'] = criterion
    df_clusters['analog_criterion_value'] = resolved_values

    counts = (
        df_clusters.dropna(subset=['analog_criterion_value'])
        .groupby('analog_criterion_value', dropna=False)
        .agg(
            n_rows=('date', 'size'),
            n_anchor_holidays=('anchor_holiday_name', 'nunique'),
        )
        .reset_index()
    )

    catalog = _build_analog_cluster_catalog(counts, criterion, cluster_labels)
    analog_map = dict(zip(catalog['analog_criterion_value'], catalog['analog_cluster']))
    df_clusters['analog_cluster'] = df_clusters['analog_criterion_value'].map(analog_map)

    return {
        'df_selector_clusters': df_clusters,
        'analog_cluster_catalog': catalog,
        'analog_criterion_selector_col': selector_col,
        'analog_criterion_prior_col': prior_col,
    }


def load_selector_cluster_lookup(
    selector_path: Path | str,
    cluster_column: str = 'analog_cluster',
    unique_id: str | None = None,
) -> dict[pd.Timestamp, object]:
    """Load a per-date selector cluster lookup from an exported selector CSV."""
    selector_path = Path(selector_path)
    if not selector_path.exists():
        raise FileNotFoundError(f'Selector feature file not found: {selector_path}')

    df_selector = pd.read_csv(selector_path, parse_dates=['date'])
    if 'date' not in df_selector.columns:
        raise ValueError(f'Selector CSV {selector_path} must contain a date column.')
    if cluster_column not in df_selector.columns:
        raise ValueError(
            f'Selector CSV {selector_path} must contain the cluster column {cluster_column!r}.'
        )

    if 'unique_id' in df_selector.columns:
        available_unique_ids = pd.Series(df_selector['unique_id']).dropna().astype(str).unique().tolist()
        if unique_id is not None:
            df_selector = df_selector[df_selector['unique_id'].astype(str) == str(unique_id)].copy()
            if df_selector.empty:
                raise ValueError(
                    f'Selector CSV {selector_path} has no rows for unique_id={unique_id!r}.'
                )
        elif len(available_unique_ids) > 1:
            preview = ', '.join(sorted(available_unique_ids)[:5])
            raise ValueError(
                'Selector CSV contains multiple unique_id values. '
                f'Pass unique_id to select one series: {preview}'
            )

    df_clusters = df_selector[['date', cluster_column]].copy()
    df_clusters['date'] = pd.to_datetime(df_clusters['date']).dt.normalize()
    df_clusters = df_clusters.dropna(subset=[cluster_column])
    if df_clusters.empty:
        return {}

    conflicting_dates = (
        df_clusters.groupby('date')[cluster_column]
        .nunique(dropna=True)
        .loc[lambda series: series > 1]
    )
    if not conflicting_dates.empty:
        preview = ', '.join(
            pd.Timestamp(date_value).strftime('%Y-%m-%d')
            for date_value in conflicting_dates.index[:5]
        )
        raise ValueError(
            'Selector CSV has conflicting cluster labels for the same date '
            f'in column {cluster_column!r}: {preview}'
        )

    return (
        df_clusters.sort_values('date')
        .drop_duplicates(subset=['date'], keep='first')
        .set_index('date')[cluster_column]
        .to_dict()
    )


def identify_future_holiday_analog_cluster(
    candidate: pd.Series | dict,
    df_priors: pd.DataFrame,
    criterion: str = 'event_profile_cluster',
    group_cols: tuple[str, ...] = ('anchor_holiday_name', 'holiday_day_type'),
    analog_cluster_catalog: pd.DataFrame | None = None,
    cluster_labels: tuple[str, ...] = ('F', 'G', 'H'),
) -> pd.Series:
    """Assign an ex-ante analog cluster to a future holiday candidate."""
    candidate_series = pd.Series(candidate).copy()
    if 'anchor_holiday_name' not in candidate_series and 'holiday_name' in candidate_series:
        candidate_series['anchor_holiday_name'] = candidate_series['holiday_name']

    resolved_group_cols = _resolve_selector_group_cols(
        group_cols,
        candidate_series.index,
        df_priors.columns,
    )

    if 'unique_id' in resolved_group_cols and ('unique_id' not in candidate_series or pd.isna(candidate_series['unique_id'])):
        prior_unique_ids = pd.Series(df_priors['unique_id']).dropna().astype(str).unique().tolist()
        if len(prior_unique_ids) == 1:
            candidate_series['unique_id'] = prior_unique_ids[0]

    missing = [col_name for col_name in resolved_group_cols if col_name not in candidate_series or pd.isna(candidate_series[col_name])]
    if missing:
        raise ValueError(f'Candidate is missing required analog-key fields: {missing}')

    criterion_spec = _resolve_analog_criterion_spec(criterion)
    prior_col = criterion_spec.get('prior_col')
    if prior_col is not None and prior_col not in df_priors.columns:
        raise ValueError(
            f'Missing prior criterion column {prior_col!r} for analog criterion {criterion!r}.'
        )

    prior_match = df_priors.copy()
    for col_name in resolved_group_cols:
        prior_match = prior_match[prior_match[col_name] == candidate_series[col_name]]

    if prior_match.empty:
        raise ValueError(
            'No selector prior was found for the candidate. '
            f'Expected a match on {list(resolved_group_cols)}.'
        )

    prior_row = prior_match.sort_values('history_rows', ascending=False).iloc[0]
    for col_name in prior_row.index:
        if col_name.startswith('inferred_') and col_name not in candidate_series.index:
            candidate_series[col_name] = prior_row[col_name]

    candidate_series['analog_criterion'] = criterion
    candidate_frame = pd.DataFrame([candidate_series])
    candidate_series['analog_criterion_value'] = _resolve_analog_criterion_values(
        candidate_frame,
        criterion,
    ).iloc[0]

    if analog_cluster_catalog is None:
        if prior_col is not None and prior_col in df_priors.columns:
            resolved_values = df_priors[prior_col].dropna().drop_duplicates().tolist()
        else:
            resolved_values = list(criterion_spec.get('ordered_values') or ())
        if not resolved_values and pd.notna(candidate_series['analog_criterion_value']):
            resolved_values = [candidate_series['analog_criterion_value']]
        analog_cluster_catalog = _build_analog_cluster_catalog(
            pd.DataFrame({'analog_criterion_value': resolved_values}),
            criterion,
            cluster_labels,
        )[[
            'analog_cluster',
            'analog_criterion',
            'analog_criterion_value',
        ]]

    analog_map = dict(zip(
        analog_cluster_catalog['analog_criterion_value'],
        analog_cluster_catalog['analog_cluster'],
    ))
    candidate_series['analog_cluster'] = analog_map.get(candidate_series['analog_criterion_value'], pd.NA)
    candidate_series['prior_resolution_scope'] = prior_row.get('prior_resolution_scope', pd.NA)

    return candidate_series


def get_historical_analog_pool(
    candidate: pd.Series | dict,
    df_selector: pd.DataFrame,
    df_priors: pd.DataFrame,
    criterion: str = 'event_profile_cluster',
    group_cols: tuple[str, ...] = ('anchor_holiday_name', 'holiday_day_type'),
    analog_cluster_catalog: pd.DataFrame | None = None,
    history_end_date: pd.Timestamp | str | None = None,
    cluster_labels: tuple[str, ...] = ('F', 'G', 'H'),
) -> dict:
    """Return the historical selector rows compatible with a future candidate."""
    cluster_results = assign_holiday_selector_analog_clusters(
        df_selector=df_selector,
        df_priors=df_priors,
        criterion=criterion,
        group_cols=group_cols,
        cluster_labels=cluster_labels,
    )
    df_selector_clusters = cluster_results['df_selector_clusters']
    analog_cluster_catalog = (
        cluster_results['analog_cluster_catalog']
        if analog_cluster_catalog is None else analog_cluster_catalog
    )

    candidate_info = identify_future_holiday_analog_cluster(
        candidate=candidate,
        df_priors=df_priors,
        criterion=criterion,
        group_cols=group_cols,
        analog_cluster_catalog=analog_cluster_catalog,
        cluster_labels=cluster_labels,
    )

    pool_df = df_selector_clusters.copy()
    if pd.notna(candidate_info.get('analog_cluster', pd.NA)):
        pool_df = pool_df[pool_df['analog_cluster'] == candidate_info['analog_cluster']].copy()
    else:
        pool_df = pool_df.iloc[0:0].copy()

    if history_end_date is not None and 'date' in pool_df.columns:
        cutoff = pd.Timestamp(history_end_date).normalize()
        pool_df = pool_df[pd.to_datetime(pool_df['date']).dt.normalize() < cutoff].copy()

    if 'date' in pool_df.columns:
        pool_df = pool_df.sort_values('date').reset_index(drop=True)

    return {
        'candidate': candidate_info,
        'pool': pool_df,
        'analog_cluster_catalog': analog_cluster_catalog,
    }
