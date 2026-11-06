# Notes -- experiment_2026_06_15_11_51

## Hypothesis
With the test set held fixed at 2025-2026, does **more analog history help** -- specifically,
do the COVID-contaminated 2020-2021 years help or hurt the clean 2025-2026 forecasts? First
confirmed empirically that the pipeline already uses ALL available history: for Christmas
2025 (SIN), 6 of 10 analogs are 2020-2021. This run trims history to 2023+ (drops pandemic)
to measure their contribution.

## Setup deltas vs baseline
- New `HISTORY_START` knob (cell 3): trims the working source to ds >= HISTORY_START before
  fitting; recorded as `config.history_start`. Here set to **2023-01-01** (excludes 2020-2021).
- Single axis vs experiment_2026_06_14_21_46 (FULL history, 2020-2024): only the analog
  history start. Same criterion (observance_tier), same 19 targets (2025-2026), MIN_K=2.
- 152 rows, 0 fails. NOTE: 2022 is missing from the data, so "2023+" = 2023-2024 (2 years)
  as analogs for 2025 targets.

## Observations
- **FULL history (2020-2024): median mape_24 = 3.762%.** **CLEAN history (2023-2024 only):
  4.414%.** Trimming the pandemic years makes it **+0.65 pp WORSE**. Mean 5.27 -> 6.13.
- Per-date: **13 of 19 dates are BETTER with full history**, only 6 better when trimmed.
- Biggest losses from cutting history (i.e. pandemic analogs were HELPING): Christmas Day
  2025 4.33 -> **8.80 (+4.47)**, Holy Saturday 2025 5.44 -> 8.58, New Year's Eve 2025
  3.62 -> 6.63, Good Friday 2026 4.41 -> 7.38. Deep/partial holidays with few clean analogs
  lean hard on the extra (even contaminated) history.
- A minority improved when trimmed (pandemic analogs hurt them): Holy Saturday 2026
  6.06 -> 4.27, Constitution 2026 4.48 -> 2.87, Maundy Thursday 2026 3.59 -> 2.22.

## Conclusion
**More history helps -- hypothesis confirmed.** Even the COVID-contaminated 2020-2021 years
are net-positive for 2025-2026 forecasts: cutting them costs +0.65 pp median (13/19 dates
worse). With only 2 clean years (2023-2024) each cluster has too few analogs (~2 per anchor),
and the analog ranking + regression extract more signal from a larger pool than the
contamination costs. This also reconciles the earlier 2023-target failure: pandemic data is
only toxic when it is the ONLY history available (2023 targets); as a minority of a rich pool
(2025-2026 targets) it helps. Keep using full history in production.

## Recommended next experiments
1. **Confirm monotonicity (optional):** a 2024-only history point (HISTORY_START=2024-01-01)
   should be worse still than 4.41% -- closing the "more history is better" curve.
2. **Targeted pandemic down-weighting, not removal:** the few dates that improved when
   trimmed (Holy Sat 2026, Constitution 2026) suggest a soft down-weight of 2020-2021
   analogs (e.g. recency/quality weight) could capture the best of both -- keep the pool size
   but discount lockdown profiles. Test a recency weight vs hard cutoff.
3. **Revert `HISTORY_START` to None in production** (done) -- full history is the winner.
