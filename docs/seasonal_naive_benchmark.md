# Seasonal-Naive Benchmark — Champion Analog Model vs Same-Holiday Baselines

**Purpose.** Quantify how much the champion analog-holiday model (see
[`champion_analog_holiday.md`](champion_analog_holiday.md)) beats *naive baselines built only from
each holiday's own history*. This isolates the value added by analog selection + regression over
simply re-using past instances of the same holiday, and exposes the holidays/regions where the
analog machinery is not justified.

Reproducible driver: [`experiments/seasonal_naive_benchmark.py`](../experiments/seasonal_naive_benchmark.py)
→ writes `experiments/seasonal_naive_results.csv`.
Champion metrics: `experiments/experiment_2026_06_16_11_35_production_cap_H6_plotted/metrics.csv`.

---

## 1. Baselines

For each `(region, holiday target)` the target's actual 24-h holiday profile is compared against
forecasts assembled **exclusively from prior instances of the *same* holiday** (e.g. Independence
Day forecast from past Independence Days), matched by `anchor_holiday_name` + `holiday_day_type`:

| Baseline | Definition |
|---|---|
| `persist1` | the **most recent prior** instance (last year's same holiday) |
| `mean2` | mean profile of the **last 2** instances |
| `mean3` | mean profile of the **last 3** instances |
| `mean4` | mean profile of the **last 4** instances |

- **Window / metric:** 24 holiday hours (00:00–23:00 of the holiday), MAPE — identical to the
  champion's `mape_24_pct`, so the numbers are directly comparable.
- **Data:** hourly demand 2020–2027, **2022 entirely missing**; so for a 2025 target the last 4
  instances are 2024, 2023, 2021, 2020.
- **Scope:** all **8** SEN regions × **19 instances of the 12 official holidays** (all 12 in 2025 +
  the 7 in Jan–May 2026) = 152 cells. Main day only (H4 "day-after" excluded). Per-holiday cell counts
  differ: both-year holidays = 16 cells, **2025-only holidays = 8 cells** (Independence, Revolution,
  Christmas Eve/Day, New Year's Eve).

---

## 2. Results

### 2.1 Overall and by region difficulty (median `mape_24`, %)

| group | n | champion | persist1 | mean2 | mean3 | mean4 | champ<persist1 | champ<mean2 |
|---|---|---|---|---|---|---|---|---|
| **ALL** | 152 | **3.74** | 5.00 | 4.97 | 5.77 | 6.68 | 66% | 57% |
| TRACTABLE (5 reg) | 95 | **2.88** | 4.08 | 3.91 | 4.94 | 5.71 | 72% | 66% |
| HARD (NES/NTE/PEN) | 57 | 6.26 | 6.89 | **6.70** | 8.33 | 9.06 | 56% | 42% |

- Overall the champion beats every naive baseline, but **only modestly over `persist1`/`mean2`**
  (66% / 57% of cells).
- **Averaging more years gets worse** (`mean3` 5.77, `mean4` 6.68 ≫ `persist1` 5.00): demand grows
  year-on-year and the 2022 gap forces `mean4` back to 2020–2021 (COVID, lower level), biasing it.
- On **HARD regions the champion is roughly tied with `mean2`** (6.26 vs 6.70; wins only 42% of cells).

### 2.2 By holiday (median across the 8 regions, %) — champion edge vs best naive

| holiday | champion | best naive | edge (pp) | winner |
|---|---|---|---|---|
| New Year's Eve | **2.68** | 6.84 | **+4.15** | champion |
| Christmas Eve | **4.28** | 7.28 | **+3.00** | champion |
| Christmas Day | **2.96** | 5.64 | **+2.68** | champion |
| Maundy Thursday | **3.70** | 5.07 | +1.37 | champion |
| Good Friday | **4.11** | 5.32 | +1.21 | champion |
| Labor Day | **3.34** | 3.91 | +0.58 | champion |
| New Year's Day | **2.77** | 3.26 | +0.49 | champion |
| Mexican Revolution Day | **1.83** | 2.28 | +0.45 | champion |
| Benito Juárez's Birthday | 4.31 | **3.93** (mean2) | −0.38 | naive |
| Holy Saturday | 5.88 | **5.40** (persist1) | −0.48 | naive |
| Constitution Day | 3.68 | **2.78** (mean2) | −0.91 | naive |
| **Independence Day** | 6.36 | **3.04** (mean4) | **−3.32** | **naive** |

### 2.3 By region (median across holidays, %)

| region | champion | persist1 | mean2 | best naive |
|---|---|---|---|---|
| CEL | **2.10** | 2.84 | 3.15 | champion |
| OCC | **2.46** | 3.39 | 3.50 | champion |
| SIN | **3.02** | 4.45 | 5.21 | champion |
| NOR | **3.53** | 10.25 | 11.29 | champion (huge) |
| ORI | **3.80** | 6.24 | 4.09 | champion |
| PEN | **6.22** | 14.74 | 8.28 | champion (huge) |
| NES | 6.26 | **6.03** | 6.70 | ~tied (persist1) |
| **NTE** | 7.12 | 4.97 | **4.13** (mean2) | **naive (method hurts)** |

---

## 3. Findings & insights

1. **The analog model wins where cross-holiday borrowing and shape help; it loses where a holiday
   is stable and self-similar.** Champion wins on 8/12 holidays — all the *shape-rich / few-instance*
   days: Christmas Eve/Day, New Year's Eve/Day, the Easter-week trio, Labor Day, Revolution. It loses
   on 4/12 — the **stable civic, fixed-date** holidays: **Independence (by 3.3 pp), Constitution,
   Juárez, Holy Saturday**, where a plain mean of the holiday's own past instances is better.

2. **Independence Day is the headline anomaly.** Fixed date, consistent observance, no nearby holidays
   to borrow from → `mean4` = **3.04%** vs champion 6.36%. The analog machinery *over-engineers* it.

3. **`persist1` vs `mean2`:** averaging the last 2 instances is the best single naive overall
   (4.97% vs 5.00%) and the strongest competitor on civic holidays; `mean3/4` only help the most
   stable, trend-flat holiday (Independence) and hurt elsewhere (demand growth + 2022 gap).

4. **NTE confirms a fixable method failure, not irreducible noise:** `mean2` (4.13%) beats champion
   (7.12%) — NTE's holidays *are* predictable from their own history; the analog selection/regression
   actively degrades them. (NES is ~tied; PEN is genuinely hard — naive `persist1` 14.74%.)

5. **Region-difficulty reframing holds:** champion tractable-5 median = **2.88%** vs raw 3.74%; the
   global number is inflated by NES/NTE/PEN.

---

## 4. Implication — a per-holiday **hybrid** beats the pure analog model

Selecting, per holiday, the method with the better validation median (naive on Independence→`mean4`,
Constitution→`mean2`, Juárez→`mean2`, Holy Saturday→`persist1`; champion elsewhere):

| | champion | **hybrid** |
|---|---|---|
| ALL median | 3.74% | **3.37%** |
| ALL mean | 5.34% | **5.22%** |
| TRACTABLE median | 2.88% | **2.66%** |

A **−0.37 pp** headline gain, driven almost entirely by Independence. This is an *in-sample,
validation-selected* upper bound (n = 8 regions/holiday) — treat as motivation, not a tuned result;
the rule must be fixed on held-out years before claiming it. **Recommended production model: a hybrid
that routes stable civic holidays (and NTE) to a same-holiday seasonal mean, and everything else to
the champion analog model.**

---

## 5. Reproducibility

```bash
python3 experiments/seasonal_naive_benchmark.py \
    experiments/experiment_2026_06_16_11_35_production_cap_H6_plotted/metrics.csv
```
Outputs `experiments/seasonal_naive_results.csv` (one row per region×holiday: champion, persist1,
mean2, mean3, mean4, cluster, n_prior) and prints the tables above. Champion run commit:
`10ff161` (branch `analog_holiday_H1_H2_H3`). Source: `holidays/holiday_demand_mx.csv` (hourly,
2020–2027, 2022 missing). Holiday matching via `holidays/holiday_selector_features.csv`
(`anchor_holiday_name` + `holiday_day_type`).

## 6. Caveats

- **Small samples:** 12 holidays, 8 regions each; per-holiday n is **8 cells** (2025-only holidays,
  incl. Independence) or 16 cells (both years). Medians preferred; per-holiday edges < ~0.5 pp are
  within the established ~0.4 pp noise floor. The headline Independence result rests on one year × 8
  regions — directional, not yet a held-out claim.
- **Trend / 2022 gap:** raw (unscaled) means of older instances are biased low by demand growth and by
  the missing 2022; a level-scaled seasonal naive would be a stronger baseline and is the natural
  next benchmark.
- **Naive uses no exogenous information** (weather), same as the champion — the comparison is
  method-vs-method on identical inputs.
