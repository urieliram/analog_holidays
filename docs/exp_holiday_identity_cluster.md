# Experiment — `holiday_identity` clustering (pure Similar-Days hard filter, ERCOT)

**Idea.** Test the claim in [`similar_days_vs_analog.md`](similar_days_vs_analog.md) §4 that the two
approaches are complementary: use the **calendar identity** of the holiday as a *hard filter* on the
analog pool, and let the downstream shape-based search (neighbour selection + dimensionality-reduced
regression + rescaling) work *inside* that pool. Concretely: one analog cluster per distinct holiday
(Independence Day, Memorial Day, MLK Day, …), so a target is only ever matched against **other
instances of the same holiday**.

## What was added

1. **New analog criterion `holiday_identity`** in [`shared/identify_holidays.py`](../shared/identify_holidays.py):
   a `value_getter` that returns the row's `anchor_holiday_name`, so every distinct holiday becomes
   its own analog-space label (F, G, H, …). Registered in `ANALOG_CLUSTER_CRITERIA_CATALOG` and
   `_resolve_analog_criterion_spec`. Day-after (H4) rows inherit their anchor, so e.g. Christmas Day
   H2 and H4 share one cluster.
2. **Notebook config** [`M_identify_HOLIDAYS_ERCOT.ipynb`](../M_identify_HOLIDAYS_ERCOT.ipynb) cell 3:
   `CLUSTERING_CRITERIUM = 'holiday_identity'` (+ catalog comment).
3. **Regenerated** [`holidays/holiday_selector_features_ercot.csv`](../holidays/holiday_selector_features_ercot.csv)
   with the new criterion → **15 clusters (F–T)**, one per ERCOT holiday. The previous
   `best_matching_weekday` export is backed up alongside as
   `holiday_selector_features_ercot__best_matching_weekday.csv`.

The downstream `P_analog_holidays_38h_ahead_ERCOT.ipynb` consumes this with no code change
(`CLUSTER_COLUMN = 'analog_cluster'`, `MATCH_TARGET_CLUSTER = True`).

## Plumbing check (validated)

With `match_target_cluster=True` the eligible analog pool is restricted to the same holiday:

| Target (2025) | Selected analogs (same-holiday filter) |
|---|---|
| Christmas Day 12-25 | 2023/2024/2016/2020/2017 **-12-25** |
| Independence Day 07-04 | 2019/2020/2018/2024/2016 **-07-04** |
| Memorial Day 05-26 | 2020-05-25, 2023-05-29, 2019-05-27, … (the floating last-Monday-of-May, same holiday) |

Without the filter the pool is shape-matched across *all* special days, mixing unrelated holidays
(e.g. Christmas analogs land on Jul-4, Feb-20, Aug-27, …).

## Result — full tuned head-to-head

Both criteria run through the **identical** `P_…ERCOT` pipeline with per-`(region, target)` Optuna
(`n_trials=25`, `timeout=300s`, `max_eval_dates=19`), rolling cutoff, **9 regions × 19 targets = 171
cells**. Metric = `mape_24_pct` (raw, the 24 holiday hours). Runs:
`experiment_2026_06_25_13_07_holiday_identity_cluster_ercot` vs
`experiment_2026_06_25_14_18_best_matching_weekday_baseline_ercot`.

| Criterion | median mape24 | mean mape24 | failed cells |
|---|---|---|---|
| **`holiday_identity` (same-holiday)** | **6.22%** | **8.17%** | **0** |
| `best_matching_weekday` (baseline) | 6.95% | 9.02% | 4 |

**`holiday_identity` wins on both median (−0.73 pp) and mean (−0.86 pp), better in 88/171 cells, and
fails 0 cells vs 4.** The same-holiday pool never ran dry (2–7 analogs per cell).

### Per-holiday — the interpretable split (identity − baseline, pp)

| Identity helps a lot | Δ | Identity hurts (deep winter) | Δ |
|---|---|---|---|
| Martin Luther King Jr. | **−5.22** | New Year's Day | +1.77 |
| Thanksgiving Day | −4.44 | Christmas Day | +1.69 |
| Labor Day | −4.26 | Christmas Eve | +1.68 |
| San Jacinto Day | −2.32 | Day after Thanksgiving | +0.63 |
| Lyndon B. Johnson | −1.12 | Presidents' Day | +0.35 |

This is the doc's **complementarity thesis, confirmed**: the hard identity filter wins big on civic /
idiosyncratic holidays whose profile the weekday-shape clustering mismatches (MLK, Thanksgiving,
Labor), and **loses on the deep Dec–Jan holidays** (Christmas, New Year), where the broader
shape-/weekday-matched pool supplies richer, well-aligned analogs and the same-holiday-only restriction
(~5–9 neighbours) is too thin. Per-region, identity wins or ties 7/9 regions; it only clearly loses on
WEST (+1.48) and SCENT (+0.52).

**Takeaway / next step.** A *hybrid* selector — `holiday_identity` for civic/idiosyncratic holidays,
shape/`observance_tier` for the deep winter cluster — should beat either pure criterion. That is the
natural follow-up experiment.

### Caveats

- Raw (pre bias-adjustment) error; bias-adjusted medians track raw (identity 6.63%). Single dataset
  (ERCOT). Both runs share git commit `6d8e3f4`, dirty tree.
- With a same-holiday-only pool, the earliest year of each holiday has **no prior analogs** under the
  rolling cutoff; here the 2025–2026 horizon has ~8 prior years each, so 0 cells failed.
- `OPTUNA_MIN_K_BY_CLUSTER`/`MAX_K_BY_CLUSTER` are keyed on the old F/G/H labels; under
  `holiday_identity` labels run F–T and fall back to the global `OPTUNA_MIN_K`. Set per-label floors
  to tune further.

## To run the full experiment

1. Open `P_analog_holidays_38h_ahead_ERCOT.ipynb` (it already points at the regenerated selector CSV).
2. Restore real tuning in cell 4: `OPTUNA_N_TRIALS = 25`, `OPTUNA_TIMEOUT_SEC = 300`,
   `OPTUNA_MAX_EVAL_DATES = 19` (currently smoke-test values 1/120/3).
3. Run all → it tunes + forecasts 9 regions × 19 targets and registers an experiment under
   `experiments/`. Compare its median/mean `mape_24` against the `best_matching_weekday` baseline.

## To revert

Set `CLUSTERING_CRITERIUM` back to `best_matching_weekday` in the M_ notebook and re-run section 11,
or copy `holiday_selector_features_ercot__best_matching_weekday.csv` back over the active CSV.
