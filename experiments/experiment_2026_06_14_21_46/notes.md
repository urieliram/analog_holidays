# Notes -- experiment_2026_06_14_21_46

## Hypothesis
Grouping the analog pool by **observance tier** -- how inhábil a holiday actually is
(`working` / `partial` / `full`, the new `observance_tier` criterion) -- instead of by
demand-curve shape lets each target borrow only from holidays that are "off" to the same
degree. Expected effect: cut the systematic **under-prediction (+bias)** on soft holidays
(the "substituted / puente" Mode B found in the observed_strength diagnostic), where the
old pools mixed full holidays into a soft target and forecast it too depressed.

## Setup deltas vs baseline
- Comparison baseline: **experiment_2026_06_13_13_27** (`shape_pearson_CDE_map_FGH`,
  ALL median mape_24 = 4.50%).
- Changed axis: `analog_cluster_criterion` -> **observance_tier** (F=working {Labor Day,
  Revolution Day}, G=partial {Maundy Thu, Good Fri, Holy Sat, Christmas Eve, NY Eve},
  H=full {Constitution, Juárez, Independence, Christmas Day, NY Day}). Tiers derived from
  `shared/observed_strength.py` (thresholds <0.55 / 0.55-0.80 / >=0.80).
- **Confound to keep in mind:** Optuna re-tuned per run, so the winners also moved
  (this run: k=2, LassoReg/None; baseline: k=10, RidgeReg/standard). This is an honest
  end-to-end criterion comparison, NOT a single-axis isolation. See next experiments.
- Scope identical: 8 SEN regions × 19 target dates (2025-2026), 152 rows, 0 fails.

## Observations
- **ALL median mape_24: 3.76% vs 4.50% baseline (-0.73 pp, ~-16% rel).** Mean 5.27 vs 5.76.
  ALL median bias_24 flips small +1.0 (was -1.0); no large systematic bias remains.
- **By day type (median mape_24, NEW vs BASE):** H1 5.59 -> **3.57 (-2.02)**, H3 5.08 ->
  4.11 (-0.97), H2 4.12 -> 3.76 (-0.36). The H1 eves (mostly the `partial` cluster) gain
  the most -- exactly the soft-holiday Mode B the criterion targets.
- **By cluster (signed mpe_24 median):** F(working) -1.15%, G(partial) **+0.42%**,
  H(full) +0.35%. The weakly-observed +bias (was +1.73% in the diagnostic) is largely gone
  -- partial holidays no longer forecast too low.
- **Biggest per-date gains:** Good Friday 2025 8.50 -> 3.56 (-4.94), New Year's Eve 2025
  7.55 -> 3.62 (-3.93), New Year's Day 2026 7.16 -> 4.43 (-2.73), Christmas Day 2025
  6.57 -> 4.33 (-2.24), Maundy Thursday 2025 5.95 -> 3.87 (-2.09). 13 of 19 dates improved.
- **Regressions (6 dates):** Holy Saturday 2026 4.52 -> 6.06 (+1.54), Independence Day 2025
  5.15 -> 6.39 (+1.24), Benito Juárez 2025 2.56 -> 4.00 (+1.44). Independence is a `full`
  holiday that paired worse here; Holy Saturday (H3/partial) is the lone soft regression.

## Conclusion
Hypothesis **supported at the aggregate and day-type level**: observance_tier beats the
shape criterion on median mape_24 and removes the soft-holiday under-prediction bias,
with the gain concentrated on H1/H3 partial days as predicted. Caveat: part of the lift
may come from Optuna's k/regressor change (k=2 Lasso is also fragile on small pools), so
the criterion's *isolated* effect is not yet proven. A few `full`-tier dates (Independence)
got worse and deserve a look.

## Recommended next experiments
1. **Isolate the criterion.** Re-run both `shape_pearson_CDE_map_FGH` and `observance_tier`
   with k and regressor FIXED (k=10, RidgeReg/standard -- the baseline winner), Optuna off
   or restricted. If observance_tier still wins, the criterion is the cause, not the tuning.
2. **Guard the small-pool / k=2 risk.** Raise `OPTUNA_MIN_K` (e.g. 4) or widen cluster F
   (working has only 2 anchors / 88 rows); confirm the gain survives a less fragile k.
3. **Fix the `full`-tier regressions.** Diagnose Independence Day 2025 & Holy Saturday 2026
   pair plots -- is `full` too coarse (Independence behaves unlike Christmas/New Year)?
   Try splitting `full` into civic vs Dec-Jan, or a 4-tier variant.
