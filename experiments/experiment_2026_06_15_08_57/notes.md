# Notes -- experiment_2026_06_15_08_57

## Hypothesis
Isolation run: does the 4-tier depth split (`observance_tier_depth`) beat the 3-tier
`observance_tier` ON ITS OWN, with `OPTUNA_MIN_K` held at 2 (same as the 3-tier run)? This
un-confounds experiment_2026_06_15_08_36, where the split and MIN_K=4 changed together.

## Setup deltas vs baseline
- Same criterion as experiment_2026_06_15_08_36 (`observance_tier_depth`, F/G/H/I) but
  **`OPTUNA_MIN_K` back to 2**. Single axis vs the 3-tier run (experiment_2026_06_14_21_46):
  ONLY the full -> full_civic/full_deep split differs.
- Scope identical: 8 SEN regions × 19 dates, 152 rows, 0 fails. k spans 2-9 (mode 2).

## Observations
- **ALL median mape_24: 3.796% -- marginally WORSE than 3-tier 3.762%** (and 4tier+MINK4
  3.766, BASE 4.50). Per-date vs 3-tier: **10 wins / 9 losses** -- a coin flip, i.e. no
  real signal. The depth split adds no aggregate value over 3 tiers.
- **The Independence/Juárez "fix" was MIN_K, not the split.** Independence Day 2025:
  6.39 (3-tier) -> 6.62 (this, 4-tier same MIN_K=2) -- actually WORSE; but 5.13 under
  4tier+MINK4. So MIN_K=4 fixed Independence, the civic/deep split did not. Juárez 2026
  3.87 -> 3.77 here (negligible).
- **Civic cluster needs more neighbours.** Cluster H(civic) mape_24 4.03 here (MIN_K=2) vs
  3.48 under MIN_K=4 -- the fully-observed civic holidays prefer k>=4. Cluster I(deep)
  over-predicts (mpe -2.32). Soft tiers F(work) 3.13 and G(partial) 3.95 do well at low k.
- New worst date: Holy Saturday 2026 7.32 (a partial/H3 date) -- noisy single case.

## Conclusion
**The depth split does NOT help.** In isolation it ties/marginally loses to the simpler
3-tier `observance_tier`, and the earlier Independence/Juárez improvement is now proven to
come from raising MIN_K, not from separating civic vs deep holidays. The real lever is
**neighbour count per tier**: fully-observed (civic/deep) holidays want k>=4, soft
(partial/working) holidays want k=2. Recommendation: **keep `observance_tier` (3 tiers) as
the production criterion** and pursue per-tier k, not more tiers.

## Recommended next experiments
1. **Per-tier MIN_K (the real win):** on the 3-tier `observance_tier`, set MIN_K=2 for
   working/partial and MIN_K=4-5 for full. Expect to capture the civic gain without the
   soft-day penalty -- should beat all four runs so far.
2. **Revert to 3-tier in production** (M_identify CLUSTERING_CRITERIUM -> observance_tier,
   re-stamp the selector) unless per-tier k makes the 4th tier pay off.
3. **full_deep over-prediction (-2.3 to -3.8% mpe across runs):** test +N_COMPONENTS or a
   small positive bias correction scoped to the deep Dec-Jan cluster.
