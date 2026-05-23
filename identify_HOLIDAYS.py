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


def _nth_monday_of_month(year: int, month: int, n: int) -> pd.Timestamp:
    """Return the nth Monday of the given month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (7 - first.dayofweek) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


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
                    'holiday_name': holiday['name'],
                })

        return (
            pd.DataFrame(rows, columns=['date', 'holiday_name'])
            .sort_values('date')
            .reset_index(drop=True)
        )

    years = range(year_min, year_max + 1)
    holiday_dates: dict = {}

    for holiday in holidays_json['holidays']:
        name = holiday['name']
        years_filter = holiday.get('years')
        holiday_years = years
        if years_filter is not None:
            valid_years = {int(year_value) for year_value in years_filter}
            holiday_years = [year for year in years if year in valid_years]
            if not holiday_years:
                continue

        if holiday['date_type'] == 'fixed':
            for year in holiday_years:
                dt = _resolve_observed_holiday_date(year, holiday)
                if dt is None:
                    try:
                        dt = pd.Timestamp(year=year, month=holiday['month'], day=holiday['day'])
                    except ValueError:
                        continue
                holiday_dates[dt.normalize()] = name

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
                holiday_dates[dt.normalize()] = name

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
                        date_norm.strftime('%d/%m/%y'),
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
                sorted_g = [d.strftime('%d/%m/%y') for d in sorted(green_dates)]
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
                sorted_b = [d.strftime('%d/%m/%y') for d in sorted(blue_dates)]
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
            for d in green_dates + blue_dates:
                name = _date_to_holiday.get(d)
                if name:
                    holiday_names_here.add(name)

            title = f'{weekday_names[dow]}\n({seg_label}, n={len(subset)})'
            if holiday_names_here:
                title += f'\n({", ".join(sorted(holiday_names_here))})'

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

        for h_date in sorted(holiday_groups[h_name]):
            if h_date not in df_wide.index:
                continue
            profile = df_wide.loc[h_date, hour_cols].values.astype(float)
            if np.isnan(profile).any():
                continue
            profiles.append(profile)
            months_seen.add(h_date.month)

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
            label = h_date.strftime('%d/%m/%Y')
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

        ax.set_title(f'{h_name}\n(n={len(profiles)})', fontsize=13, pad=6)
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

    ax_bar = axes[-1]
    x = np.arange(n_clusters)
    width = 0.25
    ax_bar.bar(x - width, df_sim['r vs Sun'].values, width, color='#d62728', alpha=0.8, label='r vs Sun')
    ax_bar.bar(x, df_sim['r vs Wed'].values, width, color='#2ca02c', alpha=0.8, label='r vs Wed')
    ax_bar.bar(x + width, df_sim['r vs Sat'].values, width, color='#f4a0b0', alpha=0.8, label='r vs Sat')
    ax_bar.axhline(0.95, color='gray', lw=1, ls='--', label='r=0.95')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'C{k}' for k in range(n_clusters)], fontsize=12)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Pearson r', fontsize=12)
    ax_bar.set_title('Similarity\nvs Sun / Wed / Sat', fontsize=13)
    ax_bar.legend(fontsize=11)
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
