# analog_holidays

Tools for identifying, auditing, and exporting holiday and special-day labels
for demand time series.

## Project layout

- `M_identify_HOLIDAYS.ipynb` - exploratory notebook for holiday detection.
- `N_holiday_features_manager.ipynb` - root-level notebook for cache updates, custom holiday injection, and launching the audit app.
- `P_analog_holidays.ipynb` - analog-holiday modeling experiments.
- `analog/` - core AnalogKNN model, special-day analog variant, and holiday orchestration helpers.
- `audit/` - Streamlit audit app, cache build pipeline, state management, and generated audit artifacts.
- `shared/` - package-shared dataset configuration and holiday-detection helpers.
- `web/` - static React SPA plus JSON export utilities for sharing audit outputs.

## Common entry points

From the repository root:

```bash
python audit/build_cache.py
streamlit run audit/app.py
python web/export_to_json.py
```

If your Windows environment does not expose `python`, use `py -3` instead.