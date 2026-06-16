# Notes -- experiment_2026_06_15_08_36

## Hypothesis
Two refinements over the 3-tier observance run (experiment_2026_06_14_21_46): (a) split the
fully-observed tier by demand-drop DEPTH -- `full_civic` (Independence/Constitution/Juárez,
~15% drop) vs `full_deep` (Christmas/New Year, ~30% drop) -- to stop civic holidays
borrowing from deep Dec-Jan winter analogs; and (b) raise `OPTUNA_MIN_K` 2 -> 4 to kill the
fragile k=2 the 3-tier run landed on. Expected: fix the Independence/Juárez regressions
without losing the soft-holiday gains.

## Setup deltas vs baseline
- Criterion `observance_tier` (3 tiers) -> **`observance_tier_depth`** (4 tiers F/G/H/I:
  working / partial / full_civic / full_deep). Rows per cluster F=88 G=272 H=136 I=152.
- `OPTUNA_MIN_K` 2 -> 4 (chosen k now spans 4-10, mode 5). **Two axes changed at once** --
  this is a combined refinement, not an isolation. Compare against both
  experiment_2026_06_14_21_46 (T3) and experiment_2026_06_13_13_27 (BASE, shape).
- Scope identical: 8 SEN regions × 19 dates, 152 rows, 0 fails.

## Observations
- **ALL median mape_24: 3.766% -- TIED with T3 (3.762%)**, still well below BASE 4.50%.
  Mean slightly worse (5.44 vs T3 5.27); ALL median bias_24 +4.02 (T3 +1.01) -- under-
  prediction crept up. So the combined change did NOT move the aggregate median.
- **Targeted regressions fixed (the goal):** Independence Day 2025 6.39(T3) -> **5.13**
  (back to BASE 5.15); Benito Juárez 2026 3.87 -> **2.83**; Juárez 2025 4.00 -> 3.82.
  Splitting civic from deep worked exactly where intended.
- **But new soft-day regressions appeared vs T3**, traceable to MIN_K=4 (their tier is
  unchanged): Christmas Eve 2025 3.30 -> 4.77 (+1.47), Labor Day 2025 2.61 -> 3.84 (+1.23),
  Maundy Thu 2025 3.87 -> 4.99 (+1.12), Good Friday 2025 3.56 -> 4.59 (+1.03), Constitution
  2025 2.46 -> 3.38 (+0.92). These soft dates preferred the k=2 that MIN_K now forbids.
- **By cluster (mape_24 / signed mpe_24):** F work 3.53/-1.74, G partial 4.22/+1.05,
  H civic **3.48/+1.62**, I deep 4.37/**-3.77**. The deep Dec-Jan cluster now over-predicts
  (-3.77%) -- isolating it removed the civic dates that were offsetting it.

## Conclusion
**Mixed / net-neutral at the aggregate.** The 4-tier split achieved its narrow objective
(Independence + Juárez fixed, civic cluster H is now the best at 3.48%), but raising
MIN_K to 4 simultaneously hurt several partial/working dates that genuinely did best at
k=2, so the median is unchanged vs the 3-tier run. The two axes cancel. The depth split is
a keeper; MIN_K=4 looks net-negative for the soft tiers.

## Recommended next experiments
1. **Un-confound:** re-run `observance_tier_depth` with MIN_K back to 2 (vs this run) to
   confirm the depth split alone beats T3 -- expected if the civic fix survives without the
   MIN_K penalty.
2. **Per-tier k floor:** let MIN_K depend on tier -- low (2) for partial/working where
   small homogeneous pools favour few neighbours, higher (4-5) for full_deep where
   over-prediction (-3.77%) suggests too few/over-fit analogs.
3. **full_deep over-prediction:** the deep cluster now leans -3.77% mpe; test a small
   positive bias correction or more components (N_COMPONENTS) just for cluster I.
