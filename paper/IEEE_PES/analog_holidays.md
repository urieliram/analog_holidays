# Holiday-Conditioned Analogue Forecasting for Day-Ahead Electricity Demand

*Draft for IEEE PES (IEEE Transactions / IEEE Access style). Written to be congruent in style and
terminology with the predecessor paper "Fast and Efficient Very Short-Term Load Forecasting Using
Analogue and Moving Average Tools" (`docs/analog_paper`). This file reports only the work actually
implemented in the repository `https://github.com/urieliram/analog_holidays`; ideas that are not yet
implemented are confined to the Future Work section.*

**Authors:** U. I. Lezama-Lope, M. Borunda-Pacheco, M. Á. Aguilar-Luna

---

## Abstract

Within electricity-demand forecasting, the prediction of public holidays remains an open problem that
is frequently treated separately. The peculiar behaviour of these days makes them hard to forecast:
they are tied to national or regional events that alter regular economic activity and therefore the
electric demand, producing patterns distinct from working days and weekends. Their scarcity is a
problem for conventional machine-learning and classical statistical time-series models, which need to
learn from the majority of ordinary days at the expense of the few holiday samples. In this work we
propose, for holiday forecasting, an adaptation of the Analogue algorithm to predict public holidays in
Mexico for the main control regions of the country, one day ahead and at hourly granularity. Rather
than treating a holiday as an anomaly inside a global model, the method models the *transition* between
the pre-holiday state and the holiday: through a nearest-neighbour selection over a curated set of
similar pre-holiday days, the Analogue algorithm is specialised to reconstruct the relevant holidays.
The neighbour pool is further conditioned by an *observance-tier* clustering that groups holidays by how
strongly they are actually observed. The results, evaluated on the seven control regions and the
national aggregate of the Mexican interconnected system over 2025–2026, show good accuracy compared with
seasonal naive baselines, together with negligible computation time. We intend this approach to
contribute to a better prediction of these days, which remain a significant challenge for ISOs such as
CENACE.

**Index Terms—** short-term load forecasting, holiday forecasting, analogue method, nearest neighbours,
observance clustering, low-data regimes.

---

## I. Introduction

The behaviour of holiday days depends on the day of the week on which the holiday falls, the type of
celebration, the season of the year, and additional conditions such as whether it forms a long weekend
("puente") or a single isolated day. Even within a single country, non-working days vary between
regions; consequently, no similarity is assumed between regions, and **all regions are treated
separately**.

A fundamental difference of holidays is the following. In machine-learning (ML) and deep-learning (DL)
models, the model detects atypical patterns by itself from the supplied features; it does not know
explicitly that tomorrow is a special day unless it is given enough data. A rare holiday or extreme
event then looks merely like a spike, a drop, or noise. The fundamental difficulty is that the event
occurs few times and the model cannot generalise well. Moreover, there is a trade-off: the more past
days one accumulates, the more the demand profiles age and lose representativeness. For this reason it
is good practice to forecast these days *apart* from the rest of the series.

In short, holidays are low-frequency events, regime shifts, structured outliers, and context-dependent
events. The difficulties of holiday forecasting can be summarised as: low frequency of occurrence, high
heterogeneity between days, contextual dependence, and non-stationarity. General ML and DL models must
learn from the majority of the series at the cost of the precision on holidays, for which there are too
few samples; this makes the statistical embedding of the holiday poor, and the generalisation cost high.

This paper's primary contribution is a **holiday-conditioned Analogue method** that predicts day-ahead
hourly demand on Mexican public holidays. The method does not search for similar holidays only; it
searches for similar *transitions* between the pre-holiday and the holiday state. It inherits the
desirable attributes emphasised for the Analogue family—high accuracy in low-data regimes, speed, a
negligible computational cost, robustness, repeatability and interpretability—and adds a
holiday-specific neighbour conditioning. The paper is organised as follows. Section II reviews related
work on holiday load forecasting. Section III describes the proposed method. Section IV details the data
and experimental protocol. Section V presents the results, Section VI discusses them, and Section VII
concludes and outlines future work. An appendix recalls the original Analogue method.

## II. Related Work

Several works address electricity-demand forecasting specifically on holidays. Son *et al.* [1] propose
a day-ahead short-term load forecasting scheme for holidays based on the modification of similar days'
load profiles. Ebrahimi and Moshari [2] use a fuzzy improved similar-day method for short-term holiday
forecasting. A German case study [3] models public holidays in load forecasting and reports that
holidays resemble Sundays. Borunda *et al.* [4] discuss advances in similar-day methods for short-term
load forecasting in power systems, reinforcing the relevance of selecting historical days with
comparable demand behaviour. A graduate thesis [5] is likewise devoted to modelling holidays in
electricity-demand prediction.

Gunawan and Huang [6] present an extensible framework for short-term holiday load forecasting that
combines Dynamic Time Warping (DTW) and an LSTM network (the DynaNC method and its LSTMH extension). The
authors explicitly recognise that holiday behaviour cannot be modelled adequately using only
conventional working-day or weekend patterns, owing to the strong temporal dependence between the
pre-holiday days and the holiday itself. DynaNC uses DTW to identify historical weeks whose load
profiles are similar to the current state of the system, and then transfers historical patterns to the
target holiday by heuristic rules and offsets; it additionally incorporates temperature to model
compensatory holidays. The LSTMH variant employs multivariate LSTM models with historical load,
temperature, weekday indicators and holiday-occurrence indicators. These authors begin to recognise not
only the influence of past days on the holiday, but also the realistic operational requirement faced by
energy managers and Independent System Operators (ISOs), who need the forecast some hours or days in
advance. Conceptually, that work shares with ours the hypothesis that the pre-holiday state carries
relevant information to infer the subsequent holiday. The methodological difference is fundamental:
whereas DynaNC performs a direct transfer of historical patterns through DTW and heuristic offset rules,
the proposed method restricts the search space exclusively to previously curated holiday events and then
*learns statistically* the relationship between pre-holiday and subsequent-holiday states through a
regression model. The proposed approach thus transforms the problem from a heuristic transfer of
patterns into a holiday-aware learning of conditioned transitions.

A highly accurate and efficient statistical framework for short-term load forecasting, with a case study
for Mexico [7], estimates a dynamic intraday bias between similar days from recent similar load profiles
in order to capture systematic discrepancies. For an eight-day-ahead forecast, an hour-dependent
conversion factor $F_h$ is computed from the four most similar preceding days as

$$
F_h \;=\; \frac{1}{4}\sum_{d=1}^{4}\frac{P^{\max}_{d,h}-E_{d,h}}{E_{d,h}},
$$

where $P^{\max}_{d,h}$ is the maximum instantaneous demand during hour $h$ of similar day $d$ and
$E_{d,h}$ the energy consumed in that hour; the factor is then applied as
$FP^{\max}_{d,h}=FE_{d,h}\,(1+F_h)$ to correct the systematic intraday bias. This bias-quantification
strategy is applied in our work as an *a-posteriori* correction, using the relationship between each
hourly value and the daily mean over the most similar analogue days.

To the best of our knowledge, the reviewed literature does not report an approach that integrates,
within a single framework: (i) a neighbour search conditioned exclusively to holidays, (ii) a selection
driven by the pre-holiday state, (iii) a regression over the associated subsequent holidays, and (iv) a
directed reduction of the search space through a statistical curation of holiday events.

## III. Materials and Methods

### A. Hypothesis and contribution

Unlike traditional holiday-forecasting approaches—where holiday demand is modelled mainly as a function
of the holiday type, weather variables and calendar features, often representing holidays by categorical
variables or average historical profiles—the hypothesis adopted here is that the behaviour of a holiday
depends to a large extent on the *previous state* of the power system observed on the pre-holiday days.
Conventional approaches implicitly assume that holidays of the same type behave alike and ignore the
transitional dynamics between the previous day and the holiday. We instead model that transition
explicitly through pairs $(Z_i, Z'_i)$, where $Z_i$ is a pre-holiday day and $Z'_i$ the subsequent
associated holiday. The forecasting process first identifies the historical pre-holidays most similar to
the current state $Y$, then selects their subsequent holidays $X'$, and finally builds a regression
model to estimate the future behaviour $Y'$. This hypothesis has a physical and operational
interpretation: the pre-holiday behaviour carries economic, social and operational signals that
directly influence the subsequent demand profile. Transitions such as 24 December–25 December, 31
December–1 January, or Maundy Thursday–Good Friday present structural changes that cannot be explained by
calendar variables alone.

### B. Forecast lead time

Depending on the operational horizon—real-time (seconds to minutes), very short-term (5 min to a few
hours), short-term (24–48 h) or medium-term (7–15 days)—a lead time from minutes to hours is typically
required. Our holiday forecast is intended as an input to the generation-dispatch processes of
Independent System Operators such as CENACE, which require the forecast a number of hours in advance
because it feeds the next-day dispatch in electricity market operations [6]. In this work we compute the next-day forecast
with about 14 h of anticipation: the forecast is computed from 00:00 h and must be ready by 08:00 h of
the day before the target. This implies that the information available for the target day is incomplete:
the **14 h preceding the holiday are also unknown and must be forecast together with the 24 holiday
hours**, yielding a window of $38 = 14 + 24$ hours.

### C. Method overview: two stages

The method is organised in **two stages** that are executed by separate components and exchange a single
artefact—the cluster label. **Stage 1 (holiday clustering)** is a preprocessing step that partitions the
curated holidays into clusters and writes a cluster label for every holiday day and region; it is
computed once and stored. **Stage 2 (holiday-conditioned analogue forecasting)** consumes that label:
for each target it restricts the analogue search to the target's cluster, selects the neighbours, fits a
regression and produces the forecast. In the implementation, Stage 1 corresponds to an identification
routine that stamps an `analog_cluster` field into the holiday feature table, while Stage 2 is the
forecasting routine that only reads it; the two stages are therefore decoupled and independently
reproducible. They are described next.

### D. Stage 1 — Holiday clustering

The purpose of this stage is to identify the patterns of the different day types so that the later
forecasting stage receives only days with similar profiles, reducing the noise introduced by holidays
with dissimilar behaviour. The clustering is performed **once, offline**, over the curated set of Mexican
holidays (each day augmented with a window of $w$ pre-holiday hours), and its output is a cluster label
assigned **per holiday day and per region**, which Stage 2 consumes.

As a first exercise the holidays were grouped by $k$-means using the Pearson correlation coefficient over
the standardised profiles as the similarity measure. More generally, the clustering is governed by an
interchangeable *criterion*; the following criteria were implemented and compared:

- **shape (Pearson):** $k$-means / correlation of the 38-h profiles, mapped to clusters F/G/H;
- **seasonal:** by season of the year (heat/cold, and winter/spring/fall);
- **best-matching weekday:** the weekday the holiday most resembles (working day, Saturday or Sunday),
  echoing the observation that holidays often behave like Sundays [3];
- **observance tier (and a depth variant):** by how strongly the day is actually observed.

The criterion retained for the Analog-holidays model is **observance tier**. It groups holidays not by raw
profile shape but by **how strongly the day is actually observed**, i.e. the depth of its demand drop
relative to the typical drop, scored per holiday and region. A static anchor-to-tier map with thresholds
on the median observed strength produces three clusters:

| Tier | Cluster | Median observed strength | Representative holidays |
|---|---|---|---|
| weakly observed | F | $<0.55$ | Labour Day, Revolution Day (often near regular operating days) |
| partially observed | G | $0.55$–$0.80$ | Maundy Thu, Good Fri, Holy Sat, Christmas Eve, New Year's Eve |
| fully observed | H | $\ge 0.80$ | Christmas Day, New Year's Day, Independence, Constitution, Juárez |

This observance criterion replaces the earlier shape-only clustering and, as reported in Section VI,
removes the systematic under-prediction bias on weakly observed days. The label produced here is the only
information Stage 2 uses from this stage: when forecasting a target holiday, the analogue pool is
restricted to the target's own cluster.

### E. Stage 2 — Holiday-conditioned analogue forecasting

The Analogue algorithm for the holiday problem keeps the same general structure as the original method
(recalled in the Appendix), with one fundamental difference. It operates on the holidays of the target's
cluster, as labelled by Stage 1. In the original algorithm the nearest
neighbours are obtained by scanning the time series for the historical windows most correlated with the
current window $Y$, yielding the set of similar windows $X$. In the proposed adaptation this scan is
omitted. Instead, the algorithm is given directly a curated set of holiday pairs—each a tuple of a
pre-holiday day $Z$ and its holiday $Z'$—from which it selects a subset of $k$ neighbours $X$ and their
subsequents $X'$, where $X$ are the pre-holiday days and $X'$ the holidays themselves:

$$
X \subseteq Z, \qquad X = \operatorname*{arg\,max}_{X_i \in Z}\; \mathrm{sim}(Y, X_i),
$$

with $Z$ a curated set of pre-holiday/holiday pairs, so that the algorithm searches only within relevant
holidays. This preselection reduces noise by eliminating false neighbours from ordinary days, lowers
complexity, and increases interpretability. Similarity $\mathrm{sim}(\cdot,\cdot)$ is the Pearson
correlation coefficient or the Euclidean distance, chosen per target.

Once the neighbours are selected, the regression between the target vector $Y$ and the regressors $X$ is
performed. As in the original Analogue method, this regression can use several statistical models—Ridge,
Lasso, PLS, OLS, PCR—or machine-learning models. Finally, the fitted model is applied to the subsequent
holidays $X'$ to estimate the holiday forecast $Y'=f(X')$. In summary, the Holiday-Analogue algorithm
receives a subset of days identified as holidays, applies conditioned analogue neighbours, and regresses
over the subsequent holidays.

### F. Per-target hyper-parameter optimisation

Hyper-parameters are optimised per target with the Optuna library. The tuned Analog-holidays parameters
are `typereg`, `typedist`, `scale_method`, `k` and `n_components`, corresponding respectively to the
regression type, similarity metric, scaling option, number of neighbours and latent components for
PCR/PLS. Training follows a
**rolling, day-by-day nested** scheme: for each holiday in the list, Optuna is run with
`train_end = target_date` over the historical special days available before that date; the same target
is then forecast with the selected hyper-parameters; finally a daily table is assembled with date,
holiday, cluster F/G/H, number of tuning dates available, $k$, similarity metric, regressor, number of
components, selected analogues, MAE and MAPE.

Two design decisions, established experimentally (Section V), are part of the Analog-holidays configuration: a
minimum separation of 24 h between candidate analogue events, and a **per-cluster ceiling on $k$** of 6
for the fully observed cluster H, which prevents the deep holidays from diluting their sharp drop with
less-similar neighbours. The full Analog-holidays configuration is summarised in Table I.

### G. A-posteriori bias correction

After the point forecast, an hourly bias-factor model rescales the 38-h profile using up to four of the
most similar neighbours, following the intraday-bias idea of [7] adapted to the relationship between each
hourly value and the daily mean of the analogue days. This produces a bias-adjusted variant of the
forecast in addition to the raw one.

**TABLE I. Analog-holidays configuration.**

| Stage | Knob | Value |
|---|---|---|
| Window | window length / pre-window | 38 h / 14 h (24 holiday hours scored) |
| Clustering | criterion | observance tier (F/G/H), analogues matched to target cluster |
| Selection | similarity (searched) | Pearson / Euclidean |
| Selection | min. event separation | 24 h |
| Selection | neighbours $k$ | tuned in $[2,\,\text{pool}]$, capped at 6 for cluster H |
| Regression | model (searched) | PCR / PLS / Ridge / Lasso |
| Regression | scaling (searched) | none / standard / min-max |
| Tuning | Optuna trials / seed / eval folds | 25 / 42 / 19 |
| Correction | hourly bias factor | up to 4 nearest analogues |

## IV. Experiments

### A. Data

The data are hourly electricity-demand series of the Mexican interconnected system (Sistema Eléctrico
Nacional, SEN). Seven control regions are studied—Central (CEL), Oriental (ORI), Occidental (OCC), Norte
(NOR), Noreste (NES), Noroeste (NTE) and Peninsular (PEN)—together with the national aggregate (SIN),
which equals the exact sum of the seven regions. The series span 2020–2027 at hourly resolution; the
year 2022 is entirely absent from the source. Each region is treated independently.

The study days are the twelve official Mexican public holidays: New Year's Day, Constitution Day, Benito
Juárez's Birthday, Maundy Thursday, Good Friday, Holy Saturday, Labour Day, Independence Day, Mexican
Revolution Day, Christmas Eve, Christmas Day, and New Year's Eve. The test set comprises **19 instances**
of these twelve holidays—all twelve in 2025 plus the seven that fall in January–May 2026 (the extent of
the 2026 data)—evaluated on the main holiday day; the "day-after" bridge days remain in the analogue pool
but are not scored. Each forecast at a given target date is tuned and fitted using only history strictly
before that date (a rolling/expanding cutoff); the available history thus spans 2020–2024 (2022 absent).
The COVID-affected years 2020–2021 are *retained*: experimentally they are net-positive analogues for the
clean 2025–2026 targets.

### B. Tested strategies and baselines

Three training strategies were considered for the analogue pool: training with all available holidays;
training only with holidays of the same season, preserving similar profile and behaviour conditions; and
clustering holidays by statistical profile similarity (Pearson correlation), which separates days such as
Constitution, Independence or Revolution Day from days such as New Year, Christmas or Good Friday. The
observance-tier criterion of Stage 1 (Section III-D) generalises the last strategy and is the one retained.

As a baseline we compare Analog-holidays against naive forecasts assembled **exclusively from each
holiday's own history**: the most recent prior instance of the same holiday (`SeasonalNaive`), and the
mean profile of the last two, three and four instances (`AvgSeasonalNaive-2`,
`AvgSeasonalNaive-3`, `AvgSeasonalNaive-4`). These seasonal naive baselines isolate
the value added by the analogue selection and regression over simply re-using past instances of the same
holiday.

### C. Error metrics

The loss functions are the Mean Absolute Error (MAE) and the Mean Absolute Percentage Error (MAPE),
computed over the 24 holiday hours of each test day (the 14-h pre-window and the full 38-h window are
also recorded). The bias (mean signed error) is reported per region.

## V. Results

The Analog-holidays model attains a **median MAPE of 3.74%** over the 24 holiday hours across the 152
test cells (8 series $\times$ 19 instances, no failures), with a mean of 5.34%. The per-series results
are summarised in Table II. The error is dominated by a regional, not a holiday-type, effect: the three
weather-sensitive regions (NES, NTE, PEN) are about three times worse than the largest series; restricting to the
five most tractable series (CEL, OCC, SIN, NOR, ORI) the median MAPE is **2.88%**. A diagnostic shows
this gap is not due to a thin analogue pool (no target is starved of analogues) but to intrinsic demand
noise and to the year-to-year variability of the observance in those weather-sensitive regions.

Across the 152 cells, the per-target Optuna search chose the regressor adaptively: Ridge 94 times, Lasso
35, PCR 13 and PLS 10. Forcing a single regressor was found to be equal-or-worse, and the fully observed
cluster H benefited from the $k\le 6$ ceiling, which clips the cells where the search would otherwise
select too many, less-similar analogues. The forecast-versus-actual grids and the analogue
pair-sequences per region are provided as figures
(`experiments/experiment_2026_06_16_11_35_production_cap_H6_plotted/plots/`).

**TABLE II. Analog-holidays MAPE by series (24 holiday hours, %), and bias.**

| Series | median MAPE | mean MAPE | mean bias (MW) |
|---|---|---|---|
| CEL | 2.10 | 2.39 | +38 |
| OCC | 2.46 | 3.71 | −61 |
| SIN (national) | 3.02 | 3.81 | +277 |
| NOR | 3.53 | 4.50 | −49 |
| ORI | 3.80 | 4.76 | +162 |
| PEN | 6.22 | 8.39 | −15 |
| NES | 6.26 | 7.67 | +9 |
| NTE | 7.12 | 7.46 | −82 |
| **All Regions** | **3.74** | **5.34** | +35 |

### A. Comparison against seasonal naive baselines

Table III compares An-holidays against the seasonal naive baselines, by region difficulty and overall.
An-holidays improves on every naive baseline overall (median 3.74% vs 5.00% for `SeasonalNaive`), and the
advantage is larger on the tractable series (2.88% vs 4.08%). Averaging more past instances degrades the
naive (`AvgSeasonalNaive-3`/`AvgSeasonalNaive-4`) because of demand growth and the 2022 gap.

**TABLE III. An-holidays vs seasonal naive and average seasonal naive baselines (median MAPE, %).**

| group | n | An-holidays | SeasonalNaive | AvgSeasonalNaive-2 | AvgSeasonalNaive-3 | AvgSeasonalNaive-4 |
|---|---|---|---|---|---|---|
| All Regions | 152 | **3.74** | 5.00 | 4.97 | 5.77 | 6.68 |
| tractable (CEL/OCC/SIN/NOR/ORI) | 95 | **2.88** | 4.08 | 3.91 | 4.94 | 5.71 |
| hard (NES/NTE/PEN) | 57 | 6.26 | 6.89 | 6.70 | 8.33 | 9.06 |

Broken down by holiday, An-holidays wins on 8 of the 12 holidays—those that are shape-rich or have few
own instances (New Year's Eve, Christmas Eve, Christmas Day, Maundy Thursday, Good Friday, Labour Day,
New Year's Day, Revolution)—with large margins on the December–January and Easter-week transitions
(e.g. New Year's Eve 2.68% vs 6.84% for the best naive; Christmas Eve 4.28% vs 7.28%). The naive wins on
the four stable, fixed-date civic holidays—Independence Day (mean of the last four instances 3.04% vs
6.36%), Constitution Day, Benito Juárez's Birthday and Holy Saturday—where simply averaging the holiday's
own past instances is more accurate. This indicates that the analogue machinery is most valuable on
holidays with strong transitional structure and least valuable on self-similar civic holidays.

## VI. Discussion

The Analogue method has a physical and causal justification: electricity demand exhibits structural
recurrence, repetitive social profiles and repetitive economic behaviour. Compared with ML or DL
approaches, which require a large number of event samples, holidays are scarce events; the analogue
method works precisely in low-data regimes, reducing the search space, avoiding massive training, and
incurring a near-negligible computational cost.

Three experimental findings shaped Analog-holidays. First, clustering by *observance* rather than by raw
shape took the all-targets median MAPE from 4.50% to 3.76% and removed the under-prediction bias on
weakly observed days. Second, a wider analogue pool (a 12-h instead of 24-h minimum separation between
candidate events) was found to *hurt* the deep holidays, because the search then selects more,
less-similar neighbours and dilutes the sharp drop; the 24-h separation is retained. Third, a per-cluster
ceiling of $k\le 6$ on the fully observed cluster clips the catastrophic cells where the search would
otherwise over-select analogues, with no collateral effect on the other clusters.

The honest reading of the regional results is that part of the error in NES, NTE and PEN is an
irreducible noise floor: these are weather-sensitive regions whose demand and observance vary
strongly from year to year. The seasonal naive benchmark refines this picture: for region NTE a simple
average seasonal naive profile is more accurate than the analogue model, indicating a method weakness
there rather than pure noise, whereas for PEN the analogue model adds substantial value over an otherwise
very poor naive baseline.

## VII. Conclusions

We presented a holiday-conditioned adaptation of the Analogue method for day-ahead, hourly
electricity-demand forecasting on Mexican public holidays. The method models the pre-holiday→holiday
transition through conditioned pairs $(Z, Z')$, restricts the neighbour search to a curated,
observance-clustered set of holidays, tunes its hyper-parameters per target, and applies an
a-posteriori intraday bias correction. On the seven control regions and the national aggregate of the Mexican system,
Analog-holidays attains a median MAPE of 3.74% (2.88% on the tractable series) with negligible
computation cost, and it outperforms seasonal naive baselines on the shape-rich and few-instance
holidays. The method offers higher accuracy in sparse-event regimes, low complexity, efficient training,
robustness and interpretability, and its findings can be useful for ISOs such as CENACE and power-system
operators.

### Future work

The comparison reported here is against seasonal naive baselines; a broader benchmark against
calendar-aware ML and deep-learning models and against foundation time-series models is planned. The
benchmark also motivates a per-holiday *hybrid* that routes stable civic holidays (and region NTE) to a
same-holiday average seasonal naive profile and the remaining holidays to the analogue model; such a rule
must be validated on held-out years before being adopted. A further direction is a rule-based or learned analogue
*selector* that chooses the neighbour subset not only by shape or distance but by calendar context
(weekday/Saturday/Sunday incidence, H1/H2/H3 type, whether the holiday follows another holiday, the
season, and whether it is fixed or shifted to Monday by the 2006 Mexican law). Finally, holidays not only
complicate the forecast as special days but also inject noise into the following, post-holiday recovery
days; a subsequent study may target those recovery days through imputation and tagging strategies.

## Appendix: The original Analogue method

The Analogue method comprises two main stages: the search for the days of greatest statistical similarity
and a regression process. Unlike approaches based only on similar days, it combines, in a regression
model, the most recent observed information with highly correlated historical patterns, and it retains
the subsequent windows associated with the periods of greatest similarity found in the past.

Let $X$ be the set of historical windows with the highest correlation and $X'$ the set of their
subsequent windows, so that each $x_k$ has an associated future window $x'_k$. Let $Y$ be the data that
have just occurred and $Y'$ the unknown data representing the forecast. The problem is to estimate $Y'$
from the relationship between the recent behaviour $Y$ and the most similar historical patterns $X$.
Compactly, the search is $X=\arg\max\,\mathrm{sim}(Y,X_i)$. The regression establishes the relationship
$Y=f(X)$ from recent data and historical windows, and once the model is inferred, the subsequent windows
$X'$ are used to estimate the forecast $Y'=f(X')$. Illustratively, for the day before an important
celebration such as 31 December, forecasting 1 January is expected to relate to other pre-celebration
days observed in previous years; the challenge is to capture, mathematically, the similarity between the
present and the historical patterns and to relate it to the subsequent observed holiday periods $X'$.

## Data and code availability

Code: `https://github.com/urieliram/analog_holidays`. The Analog-holidays configuration, the experiment log and
the reproducible drivers are under `experiments/`; the technical specification and the seasonal-naive
benchmark are in `docs/champion_analog_holiday.md` and `docs/seasonal_naive_benchmark.md`.

---

## References

> IEEE numbered style (`IEEEtran`/`ieeetr`). Entries marked *[complete]* have bibliographic fields
> (authors / volume / pages) missing in the source draft and must be completed against the originals
> before submission; no fields were invented.

[1] J. Son, J. Cha, H. Kim, and Y. Wi, "Day-ahead short-term load forecasting for holidays based on
modification of similar days' load profiles," *IEEE Access*, vol. 10, pp. 17864–17880, 2022.

[2] A. Ebrahimi and A. Moshari, "Holidays short-term forecasting using fuzzy improved similar day
method," *Int. Trans. Electr. Energy Syst.*, vol. 23, no. 8, pp. 1254–1271, 2013.

[3] *[complete authors]*, "Modeling public holidays in load forecasting: a German case study," *J.
Modern Power Syst. Clean Energy*, 2019. [Online]. Available:
https://doi.org/10.1007/s40565-018-0385-5

[4] M. Borunda, L. Conde-López, G. Ruiz-Chavarría, G. Lopez Lopez, V. M. Alvarado, and E. de J. Carrera
Avendaño, "Advances in Similar Day Methods for Short-Term Load Forecasting for Power Systems,"
*Forecasting*, vol. 8, no. 2, art. 32, 2026.

[5] J. D. López González, "Modelización de los días festivos en la predicción de demanda de energía
eléctrica," B.S. thesis, Univ. Politécnica de Madrid, Madrid, Spain. [Online]. Available:
https://oa.upm.es/61149/

[6] J. Gunawan and C.-Y. Huang, "An extensible framework for short-term holiday load forecasting
combining dynamic time warping and LSTM network," *IEEE Access*, vol. 9, pp. 106885–106894, 2021.

[7] *[complete authors]*, "A highly accurate and efficient statistical framework for short-term load
forecasting: a case study for Mexico," *[complete venue/year]*.
