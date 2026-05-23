"""Holiday audit tools for the analog_holidays repository.

The `audit/` package contains the Streamlit review app, cache builders,
state-management helpers, and exported audit artifacts. The companion
notebook `N_holiday_features_manager.ipynb` lives at the repository root.

Usage
-----
1. From the repository root, precompute the cache:
    python audit/build_cache.py

2. Launch the Streamlit app:
    streamlit run audit/app.py

3. For the notebook-driven workflow, open:
    N_holiday_features_manager.ipynb
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
