# Notes -- per-cluster k-ceiling (WINNER: cap H=6)

Pair: this `kcap_H6` and `experiment_2026_06_16_09_31_kcap_H4`, vs no-cap control
`experiment_2026_06_16_00_29_event_gap_gap24_control`. Driver: `experiments/run_kcap_experiment.py`.
New mechanism `OPTUNA_MAX_K_BY_CLUSTER` (mirror of `OPTUNA_MIN_K_BY_CLUSTER`) + `optuna_max_k`
param in `tune_analog_holidays_optuna`.

## Hypothesis
At gap=24 the deep residual tracks k: deep median by bucket k<=3 -> 3.99%, k4-6 -> 3.31%,
k7+ -> 12.22%. So capping k on the full-observance cluster H (where the k>=7 blow-ups live)
should clip the catastrophic tail without forcing everything to k=2.

## Setup deltas vs baseline
- One axis: per-cluster k ceiling. cap H=6 (this run) and H=4. All else production (observance_tier,
  MIN_K=2, gap=24, full history, adaptive regressor search), 19 targets.
- No-op verified: `optuna_max_k=None` reproduces control exactly (SIN Christmas k=6, 3.3787).

## Observations (mape_24)
| variant | all med | all mean | deep med | deep mean | soft med | soft mean |
|---|---|---|---|---|---|---|
| control (no cap) | 3.810 | 5.469 | 3.605 | 5.600 | 3.882 | 5.409 |
| **cap H=6** | 3.739 | 5.336 | 3.516 | 5.477 | 3.889 | 5.271 |
| cap H=4 | 3.643 | 5.221 | 3.593 | 5.209 | 3.653 | 5.227 |

- **cap H=6 touches only 9 of 64 H-cells** (those with control k=8 -> 6): mean delta **-2.21pp**
  (deep -1.97, civic -2.33). Big wins (a civic cell 17.51->7.16, deep 20.90->17.94), one regression
  (+3.53). **No collateral**: F/G clusters and most H cells unchanged; soft segment flat/slightly better.
- The all-mean improvement (-0.13pp) is ENTIRELY the 9 clipped catastrophic cells -- a tail-risk
  fix, visible in the mean, not the median (median move 3.810->3.739 is within the ~0.4pp noise floor).
- **Civic-H cells over-select k and blow up too** (NES Juarez/Constitution k=8) and benefit from the
  cap -> cluster H (full observance) is the right scope; no depth split (4-tier) needed.
- cap H=4 squeezes more on aggregate (all mean 5.221) but is riskier: forcing k=4 on cells that
  wanted k5-6 causes large regressions (a deep cell 9.36->17.95, +8.58). Not worth the variance.

## Conclusion
**A k-ceiling of ~6 on cluster H is a low-risk tail guard.** It clips the k>=7 over-selection
blow-ups across deep AND civic full-observance holidays with no collateral damage (mean -2.21pp on
the 9 affected cells, ~-0.13pp overall; median within noise). cap H=4 is too aggressive. This is the
mechanistic fix for the deep residual that findings 6 (regressor) and 7 (gap) pointed to: it is about
analog QUALITY (don't dilute the sharp deep drop with less-similar neighbors), enforced via k.

## Recommended next experiments
1. **Adopt `OPTUNA_MAX_K_BY_CLUSTER = {'H': 6}` in production** (notebook In[3] + wire `optuna_max_k`
   into the In[7] tune call, mirroring `OPTUNA_MIN_K_BY_CLUSTER`). Pure tail-risk reduction, no downside.
2. **Expand the validation set** before trusting the median (the gain is a mean/tail effect on n=152;
   the recurring noise-floor caveat).
3. **Recency/quality analog weighting** (open thread 1) -- the remaining, larger lever on the deep
   residual now that k over-selection is capped.
