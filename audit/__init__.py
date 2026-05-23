"""
analog_holidays/audit
==================
Interactive human-in-the-loop validation tool for holiday and
special-day classification in SEN demand time series.

Usage
-----
1. Precompute cache (run once):
       python -m analog_holidays.audit.build_cache

2. Launch Streamlit app:
       streamlit run analog_holidays/audit/app.py
"""
from .data_loader import (
    HOUR_COLS,
    LABELS_PATH,
    CACHE_PATH,
    REGIONS,
    VALID_LABELS,
    load_audit_df,
    get_region_dates,
    get_day_row,
)
from .state_manager import (
    update_label,
    save_labels,
    load_audit_log,
    compute_flag_cols,
    export_wide,
    export_nixtla_long,
    export_exogenous_db,
    export_hourly_wide,
)

__all__ = [
    "HOUR_COLS",
    "LABELS_PATH",
    "CACHE_PATH",
    "REGIONS",
    "VALID_LABELS",
    "load_audit_df",
    "get_region_dates",
    "get_day_row",
    "update_label",
    "save_labels",
    "load_audit_log",
    "compute_flag_cols",
    "export_wide",
    "export_nixtla_long",
    "export_exogenous_db",
    "export_hourly_wide",
]
