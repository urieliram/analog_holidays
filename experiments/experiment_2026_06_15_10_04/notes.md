# Notes -- experiment_2026_06_15_10_04

## Hypothesis
On the winning 3-tier `observance_tier` criterion, apply a **per-tier neighbour floor**
(`OPTUNA_MIN_K_BY_CLUSTER = {F:2, G:2, H:4}`): keep soft tiers at k=2 but let the
fully-observed tier H tune with k>=4, where prior runs suggested the civic gain lived.
Expected: beat the uniform-MIN_K=2 3-tier run (3.762%).

## Setup deltas vs baseline
- Criterion `observance_tier` (3 tiers, production winner). New mechanism: per-target
  `optuna_min_k` keyed on the target's analog_cluster (cell 4 map + cell 8 loop; recorded
  as `tuning.optuna_min_k_by_cluster` in the manifest).
- Single axis vs experiment_2026_06_14_21_46 (3-tier, uniform MIN_K=2): only H's floor 2->4.
- Scope identical: 8×19 = 152 rows, 0 fails. Implementation verified correct: F & G results
  are byte-identical to the uniform run; only H's chosen-k distribution shifted (min 4).

## Observations
- **ALL median mape_24: 3.997% -- WORSE than 3.762% uniform.** Per-date vs uniform:
  2 wins / 5 losses / 12 ties.
- **F(work) and G(partial) unchanged** (2.89 / 4.00) -- confirms the per-tier wiring works.
- **H(full) got WORSE: 3.86 -> 4.40 (+0.54).** Forcing k>=4 on the *combined* full tier
  helped the civic dates (Independence 2025 6.39 -> 5.73, Juárez 2025 4.00 -> 3.56) but hurt
  the deep Dec-Jan dates and others (Juárez 2026 3.87 -> 5.32, New Year 2026 4.43 -> 5.19),
  netting negative.

## Conclusion
**Per-tier MIN_K does NOT help on the 3-tier criterion -- and it tells us why the earlier
runs behaved as they did.** The 3-tier H=full lumps civic holidays (which DO want k>=4) with
the deep Dec-Jan holidays (which are best at k=2; cluster I scored 3.71 @k2 vs 4.37 @k4 in
the 4-tier runs). A single k floor can't serve both, so it nets out worse. This **revises
the previous note**: the depth split and per-tier-k are COMPLEMENTARY, not alternatives --
neither helps alone, but split (to separate civic from deep) + per-tier k (civic->4,
deep->2) is the only theoretically-motivated combo left untested.

BUT keep perspective: all observance variants now sit in a tight 3.76-4.00% median band,
~0.5-0.7 pp better than the shape baseline (4.50%), and per-date win/loss between them is a
coin flip on n=19 dates. **The robust, banked win is `observance_tier` vs shape; the tier/k
micro-tuning is at the noise floor.** Diminishing returns -- validate on more dates before
trusting any further split.

## Recommended next experiments
1. **The one motivated combo:** 4-tier `observance_tier_depth` + per-tier MIN_K
   `{F:2, G:2, H_civic:4, I_deep:2}`. If it doesn't clear ~3.6%, stop subdividing.
2. **Expand the test set before more tuning:** add 2020-2024 holidays as targets (rolling
   cutoff already supports it) so medians stop being 19-date noise.
3. **Production:** settle on 3-tier `observance_tier`, uniform MIN_K=2
   (experiment_2026_06_14_21_46) -- simplest and tied-best. Already restored as the active
   selector criterion.
