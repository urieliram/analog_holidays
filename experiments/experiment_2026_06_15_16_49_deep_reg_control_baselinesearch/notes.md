# Notes -- deep regression sweep (ANCHOR: control / baseline search)

This folder is the **anchor (control)** of a 6-run regression sweep scoped to the deep /
Mode-A holidays. Sibling runs share everything except `typereg_choices`:
`..._deep_reg_PCR / _PLS / _RidgeReg / _LassoReg / _RF_enriched`.

## Hypothesis
The worst MAPE on holidays are the deep / Mode-A days (Christmas, New Year, Independence).
Does the **regression axis** hold an untapped gain there -- specializing the regressor
(forcing one) or enriching the family (adding non-linear RF) -- vs the production adaptive
Optuna search (PCR/PLS/Ridge/Lasso)?

## Setup deltas vs baseline
- Scope: 6 deep dates (2025-01-01, 2025-09-16, 2025-12-24/25/31, 2026-01-01) x 8 SEN regions = 48 rows.
- Fixed at production: `observance_tier`, MIN_K=2, **MIN_EVENT_GAP=24**, full history, 38h window.
- Only axis swept: `OPTUNA_TYPEREG_CHOICES`. Driver: `experiments/run_deep_regression_sweep.py`.
- Fidelity check: this control on Christmas/SIN = RidgeReg k=6 mape_24 **3.3787%**, identical to
  the robust winner `experiment_2026_06_14_21_46` -> driver reproduces the pipeline exactly.

## Observations (deep subset, mape_24)
| variant | median | mean | vs control |
|---|---|---|---|
| **control (adaptive search)** | **3.605%** | 5.600% | anchor |
| RF_enriched | 3.467% | 5.512% | -0.14 pp (RF never selected: picks Ridge 38 / PLS 10) |
| PLS forced | 3.637% | 5.740% | +0.03 pp (tie) |
| PCR forced | 4.343% | 5.694% | +0.74 pp |
| LassoReg forced | 4.469% | 6.979% | +0.87 pp |
| RidgeReg forced | 4.684% | 7.326% | +1.08 pp |

- Control regressor picks: RidgeReg 29, LassoReg 12, PLS 4, PCR 3 (~60% Ridge, 25% Lasso).
- **Forcing a single regressor is equal-or-worse.** Forcing Ridge/Lasso inflates the mean via
  catastrophic cells: OCC Christmas 2.40 -> **25.19%**, SIN Christmas 3.38 -> **15.45%**, NES
  3.06 -> 10.63. Same regressor, different tuned k/scale: constraining the search space
  destabilizes the *joint* k/scale tuning on volatile regions.
- **RF adds nothing**: never chosen (k=2-8 analogs is too few for trees). RF_enriched's -0.14 pp
  is Optuna-trajectory jitter (13 better / 11 worse / 24 ties vs control).
- **Noise floor**: a benign change in the Optuna `initial_k` seed alone shifts the deep median
  0.42 pp (this control 3.605 vs logged baseline 4.028, same gap=24 + same search). Anything
  inside ~0.4 pp here is noise.

## Conclusion
**Mode A is NOT a regressor-choice problem.** Optuna's per-date adaptive search already converges
to the right mix; forcing one regressor is equal-or-worse (and forcing Ridge is dangerous), and
RF is unusable with so few analogs. Keep the production adaptive search unchanged. The deep-holiday
residual lives in the **analog signal / magnitude**, not the regression method.

## Recommended next experiments
1. **Analog pool for deep days**: which historical days each deep target draws (depth-matched vs
   civic-contaminated under 3-tier H), and recency/quality weighting of the pool (ties to open
   thread 1). This is where the depth signal must be fixed.
2. **Magnitude / bias correction scoped to deep**: the deep cells over/under-shoot in magnitude;
   a depth-scoped bias model, separate from the regressor, is the untested lever.
3. **Config hygiene**: revert the uncommitted `MIN_EVENT_GAP=12` to 24 (production), or re-baseline
   the whole bitácora at 12 -- do not leave the two mixed (exp. 11_51 is the only one at 12).
