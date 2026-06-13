# Experiments

This repository is an **experimentation platform** for finding the best analog-based method to
forecast Mexican electricity demand **on holidays**. The pipeline is a *component mixer*: you swap
clusters, distance/selection criteria, regression methods, and parameters, then test combinations
across the SEN regions (`SEN_demand_*`) and across H1/H2/H3/H4 day types.

For research to compound, every run must be **repeatable** and **comparable**. That is what this
folder is for: each experiment records the exact conditions that produced a result, the error
metrics for every series and target date tested, and the figures generated. A future reader (or
agent) must be able to reproduce any number quoted here.

## What is a "component" you can mix

| Axis | Knob | Values seen so far |
|------|------|--------------------|
| Analog-space cluster | `USE_CLUSTER` / `MATCH_TARGET_CLUSTER`, `analog_cluster_criterion` | off, `shape_pearson_CDE_map_FGH`, `seasonal_heat_cold`, `seasonal_winter_sprint_fall`, `best_matching_weekday` |
| Analog selection | `TYPEDIST`, `K`, `MIN_SPECIAL_POINTS`, `MIN_EVENT_GAP`, `MAX_EVENTS`, `RECENT_WEEKEND_ANALOGS` | `pearson`/`euclidian`/`dtw`; integer knobs |
| Regression | `TYPEREG`, `SCALE_METHOD`, `N_COMPONENTS`, `REGRESSOR_PARAMS` | `PCR`/`PLS`/`RidgeReg`/`LassoReg`; `None`/`standard`/`minmax` |
| Window | `SEASON_LENGTH`, `FORECAST_START_OFFSET_HOURS` | `38 = 14 + 24` (keep the constraint) |
| Scope | `unique_id` set, `TARGET_DATES`, holiday day types | 8 SEN regions; 2025–2026 holidays |
| Tuning | Optuna `n_trials`, `timeout`, search spaces, `OPTUNA_MIN_K` | per-run |

## Directory layout

Every run is registered as one folder named `experiment_<YYYY_MM_DD_HH_MM>` (optionally with a
`_<slug>`), so folders are chronologically sortable and never collide. This is created
**automatically** — see "Automatic registration" below.

```
experiments/
  README.md                              # this file — the convention
  TEMPLATE/
    manifest.yaml                        # human reference for the manifest fields
    notes.md                             # analyst log template
  experiment_<YYYY_MM_DD_HH_MM>[_<slug>]/  # one folder per run
    manifest.yaml                        # exact conditions + provenance (reproducibility contract)
    metrics.csv                          # per (unique_id × target_date) error metrics
    summary.csv                          # aggregated metrics per series (+ an ALL row)
    plots/                               # generated figures, PNG (committed)
    notes.md                             # hypothesis, observations, conclusions, next steps
```

Example: `experiment_2026_06_13_09_06_seasonal_heat_cold_vs_shape`. The timestamp uses the same
`%Y_%m_%d_%H_%M` convention as the rest of the package; if two runs land in the same minute a
`_02`, `_03`, … suffix is appended so a past run is never overwritten.

## Automatic registration

Do not assemble these folders by hand. The helper
[`shared/experiment_logging.py`](../shared/experiment_logging.py) does it:

```python
from analog_holidays.shared.experiment_logging import save_experiment_run

save_experiment_run(
    config=EXPERIMENT_CONFIG,          # the conditions dict (every mixed knob)
    batch_results=batch_result_2025_all,  # {unique_id: AnalogHolidayBatchResult}
    figures=PDF_FIGURES,               # {name: matplotlib Figure} -> plots/<name>.png
    selector_features_path=SELECTOR_FEATURES_PATH,  # attaches holiday_day_type per row
    slug="seasonal_heat_cold_vs_shape",
)
```

It writes `manifest.yaml` (with git commit/branch and your `config`), concatenates every series'
`run_analog_holidays_batch` output into `metrics.csv`, builds `summary.csv`, and saves the figures
as PNG. The notebook
[P_analog_holidays_38h_ahead cluster.ipynb](../P_analog_holidays_38h_ahead%20cluster.ipynb) has a
final **"Register this run as a reproducible experiment"** cell that calls it with the run's
variables — run that cell at the end of each corrida.

## `manifest.yaml` — the reproducibility contract

The manifest must capture **everything needed to re-run the experiment and get the same numbers**.
Copy [TEMPLATE/manifest.yaml](TEMPLATE/manifest.yaml). Required blocks: `id`/`hypothesis`,
`provenance` (git commit + branch, date, author, source file checksum/path, env + seed),
`scope` (series, target dates, day types), and the full `components` block (every knob above). If a
value is tuned by Optuna rather than fixed, record the search space and the **winning value per
series** in `metrics.csv`.

## `metrics.csv` — schema

One row per `(unique_id, target_date)`. The core columns mirror the dataframe returned by
`run_analog_holidays_batch` (see [analog/analog_holidays.py](../analog/analog_holidays.py)); the
first two columns are added when consolidating across series:

| Column | Source | Notes |
|--------|--------|-------|
| `experiment_id` | manifest `id` | ties the row to its conditions |
| `unique_id` | batch input | the SEN series (batch runs are per-series) |
| `target_date`, `holiday_label` | batch row | |
| `holiday_day_type` | selector features | H1/H2/H3/H4 — add via the selector lookup |
| `analog_cluster`, `cluster_filter_label`, `filter_by_cluster` | batch row | which cluster / criterion filtered |
| `k`, `selected_analogs` | batch row | requested vs actually used (`analogs <= k`) |
| `typedist`, `typereg`, `scale_method`, `n_components`, `regressor_params` | batch row / config | winning config per series if Optuna-tuned |
| `forecast_start_offset_hours`, `forecast_hours` | batch row | 14 / 38 |
| `mae_window`, `mape_window_pct` | batch row | over the full 38-h window |
| `mae_24h`, `mape_24h_pct` | batch row | over the 24 holiday hours |
| `mape_38_pct`, `mape_14_pct`, `mape_24_pct`, `bias_38`, `bias_14`, `bias_24`, `mpe_24_pct` | panel metrics | recommended — the per-band metrics shown on the plots |
| `t_sel_sec`, `t_reg_sec` | batch row | selection / regression timing |
| `fail`, `error` | batch row | non-empty `error` ⇒ excluded from aggregates |

Keep it **tidy and machine-readable** (no merged cells, one header row, ISO dates). This is the
artifact future experiments diff against.

## `summary.csv` — aggregated comparison

Aggregate `metrics.csv` for at-a-glance comparison and cross-experiment tables. At minimum one row
per `(unique_id)` and one row per `(analog_cluster_criterion / config)` with: `n_targets`,
`fail_rate`, and the mean **and** median of `mape_24_pct` (or your primary metric), plus `bias_24`.
Report median alongside mean — holiday samples are small and skewed.

## Plots

Save figures under `plots/` as **PNG** — PNG/SVG/CSV are committed; `*.pdf`, `*.pkl`, `*.parquet`
are git-ignored, so do not rely on them for the record. Use
`plot_batch_inference_grid` and `plot_batch_pair_sequences_grid` and name files by content, e.g.
`plots/<unique_id>_inference_grid.png`, `plots/<unique_id>_pair_sequences.png`.

## How to run a repeatable experiment

1. Set your conditions (clusters, selection, regression, scope) in the notebook config cell, and
   optionally `EXPERIMENT_SLUG = "..."` to label the folder.
2. Run the workflow for every series in scope (the batch + plot cells).
3. Run the final **"Register this run as a reproducible experiment"** cell — it calls
   `save_experiment_run` and writes the whole `experiment_<timestamp>/` folder
   (`manifest.yaml`, `metrics.csv`, `summary.csv`, `plots/`, `notes.md`).
4. Edit the generated `notes.md`: state what you observed, whether the hypothesis held, and —
   importantly — **what to try next**, cross-referencing the experiment you compared against.

`manifest.yaml` is generated for you; [TEMPLATE/manifest.yaml](TEMPLATE/manifest.yaml) documents
every field for reference and for runs driven from a plain script instead of the notebook.

## Scientific discipline

- Change **one axis at a time** when you want a clean causal read; reserve full grid sweeps for
  exploration and label them as such.
- Always keep a **baseline** experiment (current production defaults) to compare against, and quote
  the baseline id in every comparison.
- Beware tiny samples: a handful of holidays per series per year. Prefer medians, report `n`, and
  do not over-interpret a single date.
- Never silently overwrite an experiment folder; create a new id. Past results are the record.
