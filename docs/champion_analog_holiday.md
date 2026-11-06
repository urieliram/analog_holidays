# Champion Analog-Holiday Forecasting Model — Technical Specification

**Scope.** Operational day-ahead forecasting of the 24 hourly demand values of Mexican public
holidays, for the 8 SEN regions, using an **analog (nearest-special-day) + dimensionality-reduced
regression** pipeline with rolling per-target tuning. This document fixes the *champion* configuration
for reproducibility, records every parameter and its rationale, the per-cluster and per-region
considerations, the performance, and the limitations (including where a same-holiday seasonal naive
beats it — see [`seasonal_naive_benchmark.md`](seasonal_naive_benchmark.md)).

| Field | Value |
|---|---|
| Champion run | `experiments/experiment_2026_06_16_11_35_production_cap_H6_plotted/` |
| Git commit / branch | `10ff161` / `analog_holiday_H1_H2_H3` |
| Headline (median `mape_24`) | **3.74%** all · **2.88%** tractable-5 regions · 6.26% hard-3 |
| Mean `mape_24` | 5.34% all |
| Scope | 8 regions × **19 holiday instances** (12 official holidays, 2025–2026) = 152 cells, 0 failures |

---

## 1. Pipeline architecture

Three stages; the analog forecaster only ever *reads* the cluster label produced upstream.

```
shared/observed_strength.py     ── scores each holiday's observed demand drop (per region)
            │
M_identify_HOLIDAYS.ipynb       ── applies the observance_tier criterion, writes analog_cluster
            │                       into holidays/holiday_selector_features.csv
            ▼
holidays/holiday_selector_features.csv   (per row: date, unique_id, anchor_holiday_name,
            │                             holiday_day_type ∈ {H1..H4}, analog_cluster ∈ {F,G,H})
            ▼
P_analog_holidays_38h_ahead cluster.ipynb   ── per (region, target): Optuna rolling tuning →
            │  (core: analog/analog_holidays.py)   analog selection → regression → bias adjustment
            ▼
experiments/<run>/   metrics.csv · summary.csv · plots/ · manifest.yaml   (save_experiment_run)
```

- **Forecast window:** 38 h = **14 h pre-holiday + 24 h holiday** (`SEASON_LENGTH=38`,
  `FORECAST_START_OFFSET_HOURS=14`). The reported error `mape_24` is over the 24 holiday hours.
- **Evaluation:** rolling / expanding — for each target date the model is tuned and fit using only
  history strictly before that date (`training_cutoff: rolling`).
- **Per-region, per-target:** every `(region, holiday)` gets its own Optuna search and fit.

---

## 2. Champion configuration (complete)

All values are the production knobs in `P_…cluster.ipynb` (cells In[3]/In[7]) and the core defaults
in `analog/analog_holidays.py`. Rationale cites the experiment findings in
[`experiments/BITACORA.md`](../experiments/BITACORA.md).

### 2.1 Window & scope
| Parameter | Value | Notes |
|---|---|---|
| `SEASON_LENGTH` | 38 | 14 pre + 24 holiday hours (keep the constraint) |
| `FORECAST_START_OFFSET_HOURS` | 14 | forecast starts 14 h before holiday midnight |
| `SPECIAL_LABELS` | `('holiday',)` | which labelled days are analog candidates |
| Targets | 19 instances of 12 official holidays | rolling cutoff per target |

> **Scope (exact).** All **8** SEN regions. The **12 official Mexican public holidays** (New Year,
> Constitution, Juárez, Maundy Thu / Good Fri / Holy Sat, Labor Day, Independence, Revolution,
> Christmas Eve, Christmas Day, New Year's Eve), as **19 instances**: all 12 in 2025 + the 7 that fall
> in Jan–May 2026 (the extent of the 2026 data). Only the **main day** (H1/H2/H3) is a target; the
> **H4 "day-after" days** (2-Jan, 26-Dec, Holy-Sat+1) are in the analog pool but **not evaluated**.
> Holidays present in both years have 16 cells (8 reg × 2 yr); 2025-only holidays (Independence,
> Revolution, Christmas Eve/Day, New Year's Eve) have **8 cells** — relevant when reading per-holiday
> medians (e.g. the Independence benchmark result rests on a single year × 8 regions).

### 2.2 Analog selection
| Parameter | Value | Rationale |
|---|---|---|
| `TYPEDIST` (search) | `['pearson','euclidian']` | distance for ranking analogs; tuned per target |
| `K` | tuned (see tuning) | number of analog neighbours kept |
| `MIN_SPECIAL_POINTS` | 24 | require the 24 holiday hours inside each candidate window |
| `MIN_EVENT_GAP` | **24** | min hours between candidate events. **Finding 7:** gap=12 (~2× pool via overlapping windows) is **worse**, esp. on deep holidays (k inflates, dilutes the sharp drop). |
| `MAX_EVENTS` | `None` | use all historical candidates |
| `RECENT_WEEKEND_ANALOGS` | 0 | off |
| `MATCH_TARGET_CLUSTER` | `True` | analogs restricted to the target's `analog_cluster` |

### 2.3 Clustering criterion — `observance_tier`
The analog pool is partitioned by **how strongly the day is actually observed** (the demand-drop
depth), scored by `shared/observed_strength.py` (actual drop ÷ typical drop, per holiday × region).
A static anchor→tier map with thresholds on **median observed strength**:

| Tier | Cluster | Threshold (observed strength) | Examples |
|---|---|---|---|
| working | **F** | < 0.55 | Labor Day, Revolution (often near-workday / substituted) |
| partial | **G** | 0.55 – 0.80 | Maundy Thu, Good Fri, Holy Sat, Christmas Eve, New Year's Eve |
| full | **H** | ≥ 0.80 | Christmas Day, New Year's Day, Independence, Constitution, Juárez |

- **Finding 2 (the robust win):** vs the earlier `shape_pearson` criterion, `observance_tier` took the
  all-targets median `mape_24` from **4.50% → 3.76%**, removing the systematic under-prediction bias on
  soft holidays (cluster G partial: +1.73% → +0.42%).
- **Finding 3:** a 4-tier depth split (`observance_tier_depth`) and per-tier `MIN_K` add **no** value in
  isolation — both are at the noise floor. 3-tier is kept for simplicity.

### 2.4 Regression
| Parameter | Value | Rationale |
|---|---|---|
| `TYPEREG` (search) | `['PCR','PLS','RidgeReg','LassoReg']` | tuned per target. **Finding 6:** forcing any single regressor is equal-or-worse; the adaptive search already converges (~60% Ridge, ~25% Lasso); non-linear RF is unusable (too few analogs). |
| `SCALE_METHOD` (search) | `[None,'standard','minmax']` | tuned per target |
| `N_COMPONENTS` | tuned (≤ k, for PCR/PLS) | latent dimensions |
| `REGRESSOR_PARAMS` | `{}` | — |
| `LEVELS` | `[50, 80, 95]` | prediction-interval levels |

### 2.5 Per-target Optuna tuning
For each `(region, target)` the following are searched jointly to minimise rolling MAPE on the
target's *historical* special days:

| Knob | Value |
|---|---|
| Searched | `typereg`, `typedist`, `scale_method`, `k`, `n_components` |
| `OPTUNA_N_TRIALS` | 25 |
| `OPTUNA_TIMEOUT_SEC` | 300 |
| `OPTUNA_MAX_EVAL_DATES` | 19 (rolling folds) |
| `OPTUNA_RANDOM_SEED` | 42 |
| **k lower bound** | `OPTUNA_MIN_K_BY_CLUSTER = {F:2, G:2, H:2}` (uniform 2) |
| **k upper bound** | dynamic (realisable pool, ≤24) **∧ `OPTUNA_MAX_K_BY_CLUSTER = {'H': 6}`** |

- **Finding 8 (the k ceiling):** at gap=24, deep/full-observance (cluster H) cells blow up when Optuna
  picks **k ≥ 7** (deep median by bucket: k4-6 → 3.31%, **k7+ → 12.22%**). Capping H at **k ≤ 6** clips
  the 9/64 catastrophic cells (−2.21 pp on them, ~−0.13 pp overall) with **no collateral** to F/G or to
  the median (mean/tail effect only). A pure tail-risk guard.
- **Finding 5 (history):** use **all** history (`HISTORY_START=None`). Trimming the COVID years
  (2020–2021) is worse (+0.65 pp): they are net-positive analogs for clean 2025–2026 targets.

### 2.6 Bias adjustment (intraday factor model)
After the point forecast, an **hourly bias-factor model** rescales the 38-h profile using up to
`HOURLY_FACTOR_ANALOGS = 4` of the most similar neighbours (`fit_hourly_bias_factor_model`),
producing the `*_bias_adjusted` metrics. Head/tail split = 14 h / 24 h. Reported `mape_24` is the
**raw** (pre-adjustment) holiday error; `mape_holiday24_bias_adjusted_pct` is the adjusted variant.

---

## 3. Per-cluster considerations
| Cluster | Meaning | `MIN_K` | `MAX_K` | Note |
|---|---|---|---|---|
| F (working) | weakly observed / substituted | 2 | none | soft; over-treating it was the old +bias (fixed by the criterion) |
| G (partial) | partially observed | 2 | none | Christmas Eve / New Year's Eve live here |
| H (full) | fully observed (civic **and** deep) | 2 | **6** | deep (Christmas/NYD/Independence) + civic (Constitution/Juárez) both over-select k → cap |

The 3-tier criterion deliberately **does not** split civic vs deep inside H (Finding 3 showed the split
adds nothing on the median); the `MAX_K=6` guard handles the deep tail instead.

---

## 4. Per-region considerations (champion run)
| Region | median | mean | `bias_24` (MW) | character |
|---|---|---|---|---|
| CEL | **2.10** | 2.39 | +38 | easy, tight analogs |
| OCC | **2.46** | 3.71 | −61 | easy |
| SIN | **3.02** | 3.81 | +277 | large region, over-predicts |
| NOR | 3.53 | 4.50 | −49 | seasonal but predictable |
| ORI | 3.80 | 4.76 | +162 | over-predicts |
| PEN | 6.22 | 8.39 | −15 | **genuinely hard** (volatile holidays) |
| NES | 6.26 | 7.67 | +9 | hard (≈ naive) |
| NTE | 7.12 | 7.46 | −82 | **method underperforms naive (fixable)** |

- **The dominant error pattern is REGIONAL, not holiday-type** (3× spread). Diagnosed
  (`experiments/diagnose_regional.py`): not a thin pool (0 starved cells); driven by **intrinsic
  demand noise** (corr 0.53 with normal-day variability) and **year-to-year observance variability**
  (corr 0.55). NES/NTE/PEN are weather-sensitive (Monterrey-industrial / north / Yucatán-AC); their
  variability likely tracks temperature, which the analog-only model cannot see.
- **Bias sign flips by region** (SIN/ORI/CEL over-predict; NTE/OCC/NOR under-predict) → a **per-region
  bias correction** is a clear, in-pipeline lever.
- **NTE caveat (Finding 10):** a same-holiday seasonal mean beats the champion there (4.13% vs 7.12%),
  so NTE's gap is a *method* failure, not irreducible noise — route NTE to the naive baseline.

---

## 5. Performance summary

- **Median `mape_24`:** 3.74% (all) · 2.88% (tractable-5) · 6.26% (hard-3). Mean 5.34% (all).
- **vs same-holiday naive** (full benchmark: [`seasonal_naive_benchmark.md`](seasonal_naive_benchmark.md)):
  the champion beats `persist1`/`mean2`/`mean3`/`mean4` overall and on 8/12 holidays (Christmas Eve/Day,
  New Year's Eve/Day, Maundy Thu, Good Fri, Labor Day, Revolution), with large edges on the shape-rich
  days (New Year's Eve +4.15 pp, Christmas Eve +3.00 pp, Christmas Day +2.68 pp).
- Figures: `experiments/experiment_2026_06_16_11_35_production_cap_H6_plotted/plots/`
  (`batch_inference_<region>.png` = forecast vs actual; `batch_pair_sequences_<region>.png` = analogs).

---

## 6. Limitations & recommended hybrid

1. **Stable civic holidays are over-engineered.** A plain same-holiday seasonal mean beats the champion
   on **Independence (−3.32 pp)**, Constitution, Juárez and Holy Saturday — fixed-date, self-similar
   holidays with no nearby analogs to borrow from.
2. **A per-holiday hybrid** (route Independence/Constitution/Juárez/Holy-Sat — and region NTE — to a
   same-holiday seasonal mean; champion elsewhere) lowers the headline to **≈3.37%** in-sample
   (validation-selected upper bound; must be confirmed on held-out years).
3. **No exogenous covariates.** The hard regions' residual is largely weather-driven; a temperature
   covariate is the only lever expected to materially reduce NES/NTE/PEN error — an architectural change
   outside the current analog-only pipeline.
4. **Small samples / noise floor.** ~0.4 pp is the empirical noise floor (a benign Optuna seed change
   moves the deep median that much). Treat sub-0.4 pp differences as noise; report medians with n.

---

## 7. Reproducibility

- **Code commit:** `10ff161` (`analog_holiday_H1_H2_H3`). **Seed:** `OPTUNA_RANDOM_SEED=42`.
- **Data:** `holidays/holiday_demand_mx.csv` — hourly, 2020-01-02 → 2027-12-31, **2022 missing**
  (targets 2025–2026 are unaffected; their analogs span 2020–2024). Cluster labels:
  `holidays/holiday_selector_features.csv` (criterion `observance_tier`).
- **Run (headless):**
  ```bash
  MPLBACKEND=Agg python3 experiments/run_full_plotted.py     # cap-H6 production + plots
  ```
  or, end-to-end with all diagnostics, execute `P_analog_holidays_38h_ahead cluster.ipynb`
  (`MPLBACKEND=Agg`; full PyQt pytest segfaults). The final "Register this run" cell calls
  `save_experiment_run` → a timestamped `experiments/<run>/` folder.
- **Where the champion knobs live in the notebook:** cell In[3] (`OPTUNA_*`, `MIN_EVENT_GAP=24`,
  `OPTUNA_MAX_K_BY_CLUSTER={'H':6}`), cell In[7] (rolling loop resolves per-cluster k bounds and passes
  `optuna_min_k`/`optuna_max_k` to `tune_analog_holidays_optuna`), cell In[16] (`save_experiment_run`).
- **Core:** `tune_analog_holidays_optuna` (search + folds), `run_analog_holidays` (fit + forecast),
  `fit_hourly_bias_factor_model` (bias) in `analog/analog_holidays.py`.

---

## 8. Findings & take-aways (the experiment arc)

| # | Finding | Take-away |
|---|---|---|
| 2 | `observance_tier` clustering | 4.50% → 3.76%; the robust, validated win. Cluster by *how observed* the day is. |
| 3 | depth split / per-tier-k | no value alone; at the noise floor. Keep it simple. |
| 5 | more history | use **all** history; COVID years are net-positive for clean targets. |
| 6 | regressor axis | exhausted; adaptive PCR/PLS/Ridge/Lasso already optimal; don't force one; RF unusable. |
| 7 | `MIN_EVENT_GAP` | **24, not 12**; more analog candidates *hurt* deep holidays (k inflates). |
| 8 | k ceiling | `MAX_K={'H':6}`; clips the k≥7 deep blow-ups — a free tail-risk guard. |
| 9 | regional diagnosis | error is regional (3×), driven by intrinsic noise + observance variability, not pool. |
| 10 | honest measurement | tractable-5 median **2.88%**; PEN irreducible, **NTE method-fixable** (worse than naive). |
| benchmark | same-holiday naive | beats the analog model on stable civic holidays → **hybrid** recommended. |

**One-line thesis.** The champion is an *observance-clustered, per-target-tuned, k-capped* analog
regressor that excels on shape-rich and few-instance holidays; its residual error is dominated by a few
weather-driven regions (irreducible) and by stable civic holidays that a same-holiday seasonal mean
forecasts better — motivating a naive/analog **hybrid** and, ultimately, a weather covariate.
