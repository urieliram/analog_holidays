# Notes -- MIN_EVENT_GAP axis (ANCHOR: gap=24 / production)

Pair: this `gap24_control` vs sibling `experiment_2026_06_16_00_44_event_gap_gap12`.
Driver: `experiments/run_gap_experiment.py`.

## Hypothesis
Lowering `MIN_EVENT_GAP` 24 -> 12 lets candidate analog windows overlap every 12h (~doubles the
analog candidate pool). Does a larger pool help the analog-starved deep / Mode-A cells?

## Setup deltas vs baseline
- One axis: MIN_EVENT_GAP (24 vs 12). All else at production: `observance_tier`, MIN_K=2, full
  history, adaptive regressor search PCR/PLS/Ridge/Lasso, 19 targets (2025-2026).
- gap=24 reproduces production: ALL median mape_24 **3.810%** (vs 3.762% in `14_21_46`; the 0.05pp
  is `initial_k` seed noise), deep subset 3.605% (identical to the deep-sweep control).

## Observations (mape_24)
| segment | n | gap=24 median | gap=12 median | delta |
|---|---|---|---|---|
| ALL | 152 | **3.810%** | 3.987% | +0.177 pp |
| **deep** | 48 | **3.605%** | 4.519% | **+0.915 pp** |
| soft | 104 | 3.882% | 3.862% | -0.020 pp |

- **gap=12 is WORSE, concentrated on deep holidays**: Christmas Day +3.27pp (2.96->6.23), New
  Year's Eve +2.63pp, Independence +1.20pp. Soft days unchanged (-0.02pp).
- Mechanism: gap=12 admits overlapping (shifted-duplicate) windows, not independent analogs ->
  Optuna raises k (PEN Christmas 6->11, SIN Independence 6->11) -> the sharp deep-drop signal is
  diluted by less-similar neighbors. Deep holidays want FEW highly-similar analogs.
- Win/loss gap12 vs gap24: 58 better / 60 worse / 34 ties; mean_delta +0.307pp -> net worse.

## Conclusion
**Hypothesis refuted (and inverted).** More analog candidates do not help; on deep holidays they
hurt. gap=24 wins. Reinforces deep-sweep finding 6 from another angle: the deep residual is about
analog **quality/similarity**, not quantity or regressor. **Action taken: reverted the uncommitted
`MIN_EVENT_GAP=12` to 24 in the notebook** (HEAD was already 24; debt resolved with evidence).

## Recommended next experiments
1. **Better pool, not bigger**: depth-matched analogs for deep targets + recency/quality weighting
   (down-weight COVID/lockdown profiles) -- the consistent redirect from findings 6 and 7.
2. **Cap k on deep clusters**: deep cells degrade as k grows; test a per-cluster k ceiling (mirror
   of `OPTUNA_MIN_K_BY_CLUSTER`) on the deep tier.
3. **Expand validation set** before trusting sub-0.4pp moves (recurring noise-floor caveat).
