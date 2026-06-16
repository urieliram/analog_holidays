# Notes -- experiment_2026_06_15_11_12

## Hypothesis
Does the `observance_tier` win survive on a bigger validation set, and how badly does the
COVID-contaminated history (2020-2021, with 2022 missing entirely) distort the earliest
post-pandemic targets? Expanded targets from 19 -> 42 by adding all complete 2023 (11) and
2024 (12) holidays.

## Setup deltas vs baseline
- Criterion `observance_tier` (3-tier, production), uniform MIN_K=2 -- same as the banked
  best (experiment_2026_06_14_21_46). ONLY the target set changed: +2023 +2024.
- 8 regions × 42 dates = 336 rows, 0 fails. Rolling per-target cutoff, so each target only
  sees earlier history (2023 targets => analogs almost entirely 2020-2021 COVID years).

## Observations
- **Sanity check passes exactly:** the 2025-2026 subset median mape_24 = 3.762, byte-equal
  to the 19-date run (3.762, n=152). Pipeline is deterministic; expansion is consistent.
- **By target year (median mape_24):** 2023 **4.99%** (mean 7.15, mpe -2.09), 2024 3.94%,
  2025 3.52%, 2026 4.42% (n=7, Jan-May only). 
- **Pandemic contamination confirmed and localized to 2023** (+1.0-1.5 pp vs clean years),
  exactly as predicted: 2023's only same-cluster analogs are 2020-2021. The single worst
  date is **Christmas Day 2023 = 14.11%** -- its only full-tier analogs are the deep-lockdown
  Christmases of 2020-2021, a terrible match; mpe is negative (over-prediction), i.e. it
  forecast 2023 as depressed as a COVID Christmas.
- **Post-pandemic-only (2024-2026) median = 3.859%**, vs the shape baseline ~4.50%. The
  observance_tier gain (~0.6-0.7 pp) HOLDS at 35 clean dates -- it was not 19-date noise.

## Conclusion
Two firm results. (1) **The observance_tier improvement is robust**, not a small-sample
artifact: ~3.86% on the post-pandemic set vs 4.50% shape baseline. (2) **The 2020-2021
contamination is real, quantified, and confined to 2023** because the 2022 gap leaves 2023
starved of clean analogs; it fades by 2024+ as clean history accumulates. ALL-42 median
(4.065) is dragged up purely by 2023 -- it should not be read as a regression.

## Recommended next experiments
1. **Mask the pandemic analogs:** exclude 2020 (and test 2021) holidays from the analog pool
   and re-run the 42-date set. Expect 2023 (esp. Christmas 2023 14% -> ?) to improve sharply
   while 2024-2026 barely move -- isolating the contamination cost.
2. **Report production metrics on the clean window only** (2024+): use 3.86% as the honest
   headline; footnote 2023 as analog-starved.
3. **Per-tier MIN_K, civic-only:** the depth/k thread is still open -- if pursued, test
   4-tier + {civic:4, deep:2} on this larger set so the comparison isn't 19-date noise.
