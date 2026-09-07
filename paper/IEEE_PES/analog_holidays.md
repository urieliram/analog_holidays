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
propose, for holiday forecasting, an adaptation of the Analogue algorithm that predicts public holidays
one day ahead at hourly granularity. Rather than treating a holiday as an anomaly inside a global
model, the method models the *transition* between the pre-holiday state and the holiday: through a
nearest-neighbour selection over a curated set of similar pre-holiday days, the Analogue algorithm is
specialised to reconstruct the relevant holidays. The neighbour pool is conditioned by a holiday
clustering criterion, selected among seven alternatives by a controlled sweep. The method is evaluated
on **two independent power systems**: the seven control regions and the national aggregate of the
Mexican interconnected system (152 test cells) and the eight weather zones and system aggregate of
ERCOT, Texas (171 test cells). It attains a median MAPE of **3.79 %** and **6.04 %** respectively over
the 24 holiday hours, and it outperforms both a family of seasonal naive baselines and the classical
Similar-Days method on both systems, under a paired Wilcoxon test and with the comparison deliberately
biased in favour of the baselines. We further decompose the residual error, showing that it is almost
entirely a *level* (altitude) bias rather than a shape error, and we quantify with exogenous
reanalysis and archived forecast temperature that this residual level bias is largely weather-driven —
establishing the accuracy ceiling attainable without exogenous variables and identifying the single
covariate that can move it. We intend this approach to contribute to a better prediction of these days,
which remain a significant challenge for ISOs such as CENACE and ERCOT.

**Index Terms—** short-term load forecasting, holiday forecasting, analogue method, nearest neighbours,
similar-day methods, low-data regimes, benchmarking.

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
hourly demand on public holidays. The method does not search for similar holidays only; it searches for
similar *transitions* between the pre-holiday and the holiday state. It inherits the desirable
attributes emphasised for the Analogue family—high accuracy in low-data regimes, speed, a negligible
computational cost, robustness, repeatability and interpretability—and adds a holiday-specific
neighbour conditioning. Concretely, the contributions are:

1. **A holiday-conditioned analogue formulation** that restricts the neighbour search to curated
   pre-holiday/holiday pairs and learns the transition by a per-target regression (Section III).
2. **A controlled comparison of seven neighbour-conditioning criteria** on a common panel, from which
   the retained criterion is selected — together with an explicit statement of which differences
   between criteria are statistically significant and which are not (Section V-A).
3. **A benchmark against the classical Similar-Days method** in 54 configurations (three distances ×
   three combination rules × six neighbourhood sizes), which is the standard the holiday-forecasting
   literature actually uses, rather than against naive baselines alone (Section V-D).
4. **Replication on a second, independent power system** (ERCOT, Texas), which confirms the headline
   result and, equally importantly, shows which secondary findings do *not* transfer (Sections V-D and VI).
5. **A decomposition of the residual error into level and shape components**, showing that the
   remaining error is almost entirely a level bias, and a quantification with exogenous temperature —
   both reanalysis and *archived forecasts available at issue time* — of how much of that bias is
   weather-driven (Sections V-E and V-F).

The paper is organised as follows. Section II reviews related work on holiday load forecasting.
Section III describes the proposed method. Section IV details the data and experimental protocol.
Section V presents the results, Section VI discusses them, and Section VII concludes and outlines
future work. An appendix recalls the original Analogue method.

## II. Related Work

Several works address electricity-demand forecasting specifically on holidays. Son *et al.* [1] propose
a day-ahead short-term load forecasting scheme for holidays based on the modification of similar days'
load profiles. Ebrahimi and Moshari [2] use a fuzzy improved similar-day method for short-term holiday
forecasting. A German case study [3] models public holidays in load forecasting and reports that
holidays resemble Sundays. Borunda *et al.* [4] discuss advances in similar-day methods for short-term
load forecasting in power systems, reinforcing the relevance of selecting historical days with
comparable demand behaviour. A graduate thesis [5] is likewise devoted to modelling holidays in
electricity-demand prediction.

### A. The Similar-Days family, and why it is the correct benchmark

Because the Similar-Days (SD) method is the reference against which we benchmark in Section V-C, we
state it explicitly. In its canonical form it proceeds in four steps: (i) a *context vector* is built
for every candidate day from calendar and load descriptors; (ii) a *distance* is computed between the
target day's context and that of each candidate; (iii) the $k$ nearest candidates are retained; and
(iv) their load profiles are *combined* into the forecast. Variants differ in the three degrees of
freedom that these steps expose — the distance (calendar/context descriptors, profile shape, or a
hybrid), the combination rule (simple mean, distance-weighted mean, or a mean renormalised to the
target's recent level), and $k$. We implement the full $3\times3\times6 = 54$-configuration grid rather
than a single variant, so that the comparison is against the *best* member of the family and not
against a weak instance of it.

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
$FP^{\max}_{d,h}=FE_{d,h}\,(1+F_h)$ to correct the systematic intraday bias. We implemented this
bias-quantification strategy as an *a-posteriori* correction; Section III-G reports why it is **not**
retained in the final configuration.

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

### B. Forecast lead time and the operational constraint

Depending on the operational horizon—real-time (seconds to minutes), very short-term (5 min to a few
hours), short-term (24–48 h) or medium-term (7–15 days)—a lead time from minutes to hours is typically
required. Our holiday forecast is intended as an input to the generation-dispatch processes of
Independent System Operators such as CENACE and ERCOT, which require the forecast a number of hours in
advance because it feeds the next-day dispatch in electricity market operations [6]. In this work we
compute the next-day forecast with about 14 h of anticipation: the forecast is computed from 00:00 h and
must be ready by 08:00–10:00 h of the day before the target. This implies that the information available
for the target day is incomplete: the **14 h preceding the holiday are also unknown and must be forecast
together with the 24 holiday hours**, yielding a window of $38 = 14 + 24$ hours.

This lead time is not a modelling convenience but a hard operational boundary, and we treat it as such
throughout. Once issued, the forecast feeds price formation, generation scheduling and dispatch; there
is no later opportunity to revise it. Consequently **any quantity that becomes observable after the
issue instant is inadmissible inside the method**, however predictive it may be. Section V-E reports a
correction that would reduce the error by 15 % but violates this boundary, and is therefore presented as
a diagnostic and excluded from the configuration; Section V-F reports an exogenous signal that respects
it, and is therefore admissible. We make the distinction explicit because it is easy to cross
inadvertently, and crossing it invalidates the reported accuracy for operational use.

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
with dissimilar behaviour. The clustering is performed **once, offline**, over the curated set of
holidays (each day augmented with a window of $w$ pre-holiday hours), and its output is a cluster label
assigned **per holiday day and per region**, which Stage 2 consumes.

The clustering is governed by an interchangeable *criterion*. Rather than adopt one criterion by
argument, we implemented seven and compared them on a common panel under identical conditions
(Section V-A):

- **shape (Pearson):** $k$-means / correlation of the 38-h profiles, mapped to clusters F/G/H;
- **seasonal (heat/cold)** and **seasonal (winter/spring/fall):** by season of the year;
- **best-matching weekday:** the weekday the holiday most resembles (working day, Saturday or Sunday),
  echoing the observation that holidays often behave like Sundays [3];
- **observance tier** and **observance-tier depth:** by how strongly the day is actually observed,
  i.e. the depth of its demand drop relative to the typical drop, scored per holiday and region;
- **holiday identity:** one cluster per named holiday, so that a target is matched only against past
  instances of the *same* holiday.

The criterion retained for the final configuration is **holiday identity**. Its cluster map is trivial
to state — the cluster label *is* the holiday name — but it is not a trivial method: the analogue
machinery still selects $k$ neighbours by similarity of the pre-holiday window among past instances of
that holiday, and still regresses the transition. It is best understood as the *hardest possible*
conditioning, and it is the limiting case in which the analogue search is confined entirely within a
holiday's own history.

For reference, the observance-tier criterion — retained in an earlier version of this work and
statistically indistinguishable from holiday identity (Section V-A) — groups holidays by a static
anchor-to-tier map with thresholds on the median observed strength:

| Tier | Cluster | Median observed strength | Representative holidays (Mexico) |
|---|---|---|---|
| weakly observed | F | $<0.55$ | Labour Day, Revolution Day (often near regular operating days) |
| partially observed | G | $0.55$–$0.80$ | Maundy Thu, Good Fri, Holy Sat, Christmas Eve, New Year's Eve |
| fully observed | H | $\ge 0.80$ | Christmas Day, New Year's Day, Independence, Constitution, Juárez |

The label produced here is the only information Stage 2 uses from this stage: when forecasting a target
holiday, the analogue pool is restricted to the target's own cluster.

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

Two admissibility constraints are enforced on the candidate windows, both of which matter because the
hourly series is flattened into a single vector before the search:

- **Phase alignment.** A candidate window is admissible only if its start position is congruent, modulo
  24 h, with the target window's start. Without this constraint a candidate may be matched against the
  target at a different phase of the day, so that (for example) the morning ramp of one day is compared
  with the evening ramp of another.
- **Calendar-seam exclusion.** Deliberately excluded periods (Section IV-A) leave discontinuities in the
  flattened vector. A candidate window that spans such a discontinuity would splice non-contiguous
  calendar days into a single analogue; such candidates are discarded.

Once the neighbours are selected, the regression between the target vector $Y$ and the regressors $X$ is
performed. As in the original Analogue method, this regression can use several statistical models—Ridge,
Lasso, PLS, OLS, PCR—or machine-learning models. Finally, the fitted model is applied to the subsequent
holidays $X'$ to estimate the holiday forecast $Y'=f(X')$. In summary, the Holiday-Analogue algorithm
receives a subset of days identified as holidays, applies conditioned analogue neighbours, and regresses
over the subsequent holidays.

### F. Per-target hyper-parameter optimisation

Hyper-parameters are optimised per target with the Optuna library. The tuned Analog-holidays parameters
are `typereg`, `typedist`, `scale_method`, `k`, `n_components` and, for the penalised linear models, the
regularisation strength $\alpha$ — corresponding respectively to the regression type, similarity metric,
scaling option, number of neighbours, latent components for PCR/PLS, and penalty. Training follows a
**rolling, day-by-day nested** scheme: for each holiday in the list, Optuna is run with
`train_end = target_date` over the historical special days available before that date; the same target
is then forecast with the selected hyper-parameters; finally a daily table is assembled with date,
holiday, cluster, number of tuning dates available, $k$, similarity metric, regressor, number of
components, selected analogues, MAE and MAPE.

A minimum separation of 24 h between candidate analogue events is enforced, established experimentally
(Section VI). No per-cluster ceiling on $k$ is applied in the final configuration: under holiday-identity
conditioning the pools are small enough (median 6 candidates per series in the Mexican panel) that the
search selects $k\in[2,3]$ of its own accord, and imposing the ceiling of 6 used by the earlier
observance-tier configuration leaves the results bit-identical. The full configuration is summarised in
Table I.

### G. A-posteriori bias correction: implemented, evaluated, and not retained

Following the intraday-bias idea of [7], we implemented an hourly bias-factor model that rescales the
38-h profile using up to four of the most similar neighbours, adapted to the relationship between each
hourly value and the daily mean of the analogue days. On evaluation this correction is **not retained**,
for a reason that is structural rather than empirical: the correction is constructed to have zero mean
over the analogue set, so it can redistribute error *between hours* but cannot move the level of the
profile. Since the residual error is almost entirely a level bias and only marginally a shape error
(Section V-E), the correction addresses the component that is already small, and across the panel its
net effect on MAPE over the 24 holiday hours is not beneficial. We report it because the negative result
is informative: it is direct evidence that intraday-bias corrections of this family, which are standard
in the similar-day literature, are aimed at the wrong error component for holiday forecasting.

**TABLE I. Analog-holidays configuration (final).**

| Stage | Knob | Value |
|---|---|---|
| Window | window length / pre-window | 38 h / 14 h (24 holiday hours scored) |
| Clustering | criterion | holiday identity (one cluster per named holiday) |
| Selection | similarity (searched) | Pearson / Euclidean |
| Selection | min. event separation | 24 h |
| Selection | phase alignment | candidate start $\equiv$ target start (mod 24 h) |
| Selection | calendar seams | candidates spanning an excluded period are discarded |
| Selection | neighbours $k$ | tuned in $[2,\,\text{pool}]$, no per-cluster ceiling |
| Regression | model (searched) | PCR / PLS / Ridge / Lasso |
| Regression | penalty $\alpha$ (Ridge/Lasso, searched) | $10^{-3}$–$10^{3}$, log-uniform, on standardised predictors |
| Regression | latent components (PCR/PLS, searched) | bounded by the predictor block |
| Regression | scaling (searched) | none / standard / min-max |
| Tuning | Optuna trials / seed / eval folds | 25 / 42 / 19 |
| Correction | a-posteriori hourly bias factor | implemented, **not retained** (Section III-G) |
| Exogenous | weather | **none** (see Sections V-F and VI) |

## IV. Experiments

### A. Data

The method is evaluated on two independent systems, chosen because they differ in calendar, climate and
market structure while posing the same forecasting problem.

**Mexico (SEN).** Hourly electricity-demand series of the Sistema Eléctrico Nacional. Seven control
regions are studied—Central (CEL), Oriental (ORI), Occidental (OCC), Norte (NOR), Noreste (NES),
Noroeste (NTE) and Peninsular (PEN)—together with the national aggregate (SIN), which equals the exact
sum of the seven regions. The series span 2020–2027 at hourly resolution. The study days are the twelve
official Mexican public holidays: New Year's Day, Constitution Day, Benito Juárez's Birthday, Maundy
Thursday, Good Friday, Holy Saturday, Labour Day, Independence Day, Mexican Revolution Day, Christmas
Eve, Christmas Day, and New Year's Eve. The test set comprises **19 instances** of these twelve
holidays—all twelve in 2025 plus the seven that fall in January–May 2026—evaluated on the main holiday
day; the "day-after" bridge days remain in the analogue pool but are not scored. This yields
$8\times19 = 152$ test cells.

**ERCOT (Texas).** Hourly demand for the eight ERCOT **weather zones** (COAST, EAST, FAR WEST, NORTH,
NORTH CENTRAL, SOUTH, SOUTH CENTRAL, WEST) and the system aggregate, spanning 2016–2026. The study days
are 19 instances of 15 US federal and Texas state holidays observed between January 2025 and March 2026,
yielding $9\times19 = 171$ test cells. ERCOT's zonal decomposition is climatic rather than commercial,
which is what makes the weather analysis of Section V-F possible at zone level.

**Deliberately excluded periods.** The year 2022 is absent from the Mexican series and the year 2021
from the ERCOT series. These are **not** missing data: they are quarantine-era periods excluded at
source because demand behaviour was severely contaminated by atypical confinement patterns, which would
have injected spurious analogues into the pool. The exclusions are what motivate the calendar-seam
constraint of Section III-E, and they are the reason the constraint is not merely defensive: a candidate
window straddling such a boundary would join a pre-quarantine day to a post-quarantine day and present
the result to the regressor as a single continuous analogue. The COVID-affected years that are *retained*
(2020–2021 in Mexico) were verified experimentally to be net-positive analogues for the clean 2025–2026
targets.

Each forecast at a given target date is tuned and fitted using only history strictly before that date (a
rolling/expanding cutoff). Each region is treated independently throughout; no cross-regional pooling is
performed at any stage.

### B. Tested strategies and baselines

Beyond the seven clustering criteria of Section III-D, the method is compared against two independent
families of baseline.

**Seasonal naive baselines**, assembled exclusively from each holiday's own history: the most recent
prior instance of the same holiday, and the mean and median of its last two, three and four instances,
as well as the corresponding same-calendar-date variants. These isolate the value added by the analogue
selection and regression over simply re-using past instances of the same holiday.

**Classical Similar-Days**, in the full 54-configuration grid described in Section II-A: three distances
(context descriptors, profile shape, hybrid) × three combination rules (mean, distance-weighted mean,
level-renormalised mean) × $k \in \{1,\dots,6\}$.

The Similar-Days comparison is deliberately biased **in favour of the baseline** in three ways, so that
the reported advantage is a lower bound: (i) its configuration is chosen post hoc as the best of the 54
*on the test panel itself*, whereas the analogue configuration is fixed in advance; (ii) it is granted
the level-renormalisation step, which is its single largest lever; and (iii) it draws on a much larger
candidate pool per cell — a median of 67 candidates against the analogue method's 6 in Mexico, and 143
against 9–18 in ERCOT.

### C. Error metrics

The loss functions are the Mean Absolute Error (MAE) and the Mean Absolute Percentage Error (MAPE),
computed over the 24 holiday hours of each test day (the 14-h pre-window and the full 38-h window are
also recorded). We follow the sign convention $\mathrm{MPE} = (\text{actual} - \text{forecast})/\text{actual}$,
so that a negative MPE denotes over-forecast.

Because MAPE alone does not reveal *what kind* of error is being made, we additionally report a
**level/shape decomposition**. For a given cell, $|\mathrm{MPE}|$ is the part of the absolute error
attributable to a uniform vertical displacement of the whole profile, and the remainder
$\mathrm{MAPE} - |\mathrm{MPE}|$ is the part attributable to profile shape. The ratio
$|\mathrm{MPE}|/\mathrm{MAPE}$ therefore lies in $[0,1]$ and approaches 1 when the forecast has the
right shape at the wrong altitude.

All method comparisons are made **pairwise on the same cells** and tested with the Wilcoxon signed-rank
test; we report the number of paired cells, the fraction of cells won, the median paired difference, the
skill score $1 - \tilde{e}_{\text{method}}/\tilde{e}_{\text{baseline}}$ computed on medians, and the
$p$-value. Reporting only aggregate medians would hide the fact that these panels are strongly
heterogeneous across cells.

### D. Reproducibility and software corrections

The results reported here were produced after correcting six defects found in an audit of the
implementation. We list them because three of them silently altered the effective search space, so
earlier numbers from this code base are not comparable with the present ones:

1. PLS bounded `n_components` by the response block, which is one-dimensional in this problem; the tuned
   value was therefore pinned to 1 and the hyper-parameter was inert.
2. Ridge and Lasso were fitted on unstandardised predictors with the library default $\alpha=1$, making
   the penalty scale-dependent and, in practice, negligible; they now fit through a standardising
   pipeline with $\alpha$ tuned over $[10^{-3},10^{3}]$.
3. `scale_method` was read back from a field in which a resolved value never appears.
4. A user-supplied $k$ range with `max_k < min_k` produced an empty search silently; it now raises.
5. Candidate windows were not phase-aligned (Section III-E).
6. Candidate windows could span the deliberately excluded periods (Section III-E).

Defects 1–3 are the reason we do not report a regressor-selection ranking as a scientific finding: in the
pre-correction code the "choice of regressor" was substantially a choice of shrinkage strength, so any
such table would have measured the wrong thing. The post-correction regressor frequencies are reported
in Section V-B as a descriptive statistic only.

## V. Results

The Analog-holidays model attains a **median MAPE of 3.79 %** over the 24 holiday hours across the 152
Mexican test cells (mean 5.54 %, no failures) and **6.04 %** across the 171 ERCOT cells (mean 8.28 %, no
failures). The per-series results are given in Table II.

**TABLE II. Analog-holidays MAPE by series (24 holiday hours, %), and mean bias.**

| System | Series | median MAPE | mean MAPE | mean bias (MW) |
|---|---|---|---|---|
| SEN | CEL | 2.22 | 2.23 | +16 |
| SEN | SIN (national) | 2.41 | 3.23 | +219 |
| SEN | OCC | 3.20 | 3.20 | −60 |
| SEN | ORI | 3.51 | 5.14 | +51 |
| SEN | NOR | 4.10 | 4.54 | −33 |
| SEN | NTE | 4.40 | 5.86 | −58 |
| SEN | NES | 6.70 | 8.58 | −7 |
| SEN | PEN | 8.54 | 11.51 | −42 |
| **SEN** | **all (152 cells)** | **3.79** | **5.54** | +11 |
| ERCOT | FAR WEST | 2.23 | 2.78 | −0.2 |
| ERCOT | ERCOT (aggregate) | 5.39 | 6.31 | −81 |
| ERCOT | COAST | 5.54 | 7.42 | +201 |
| ERCOT | SOUTH | 5.71 | 8.42 | +86 |
| ERCOT | WEST | 6.35 | 7.13 | −6 |
| ERCOT | NORTH | 6.75 | 8.70 | +23 |
| ERCOT | SOUTH CENTRAL | 8.26 | 10.96 | +24 |
| ERCOT | EAST | 10.75 | 12.00 | +11 |
| ERCOT | NORTH CENTRAL | 10.90 | 10.82 | −189 |
| **ERCOT** | **all (171 cells)** | **6.04** | **8.28** | +8 |

In both systems the error is dominated by a **regional**, not a holiday-type, effect. In Mexico the
weather-sensitive regions PEN (8.54 %) and NES (6.70 %) are three to four times worse than CEL (2.22 %);
restricting to the five most tractable series (CEL, SIN, OCC, ORI, NOR) the median MAPE is 2.82 %,
against 6.69 % for the remaining three.
In ERCOT the spread is wider still, from 2.23 % (FAR WEST) to 10.90 % (NORTH CENTRAL). A diagnostic
confirms this gap is not due to a thin analogue pool — no target is starved of analogues — which
motivates the investigation of Sections V-E and V-F.

### A. Comparison of clustering criteria

Table III reports the seven criteria of Section III-D on the identical 152-cell Mexican panel, with all
other settings fixed. Each criterion is compared pairwise against the retained one.

**TABLE III. Neighbour-conditioning criteria (Mexican panel, 152 cells).**

| Criterion | clusters | median pool / series | median MAPE | mean MAPE | mean $k$ | cells won by holiday-identity | $p$ |
|---|---|---|---|---|---|---|---|
| **holiday identity** | 12 | 6 | **3.789** | 5.536 | 2.34 | — | — |
| observance tier | 3 | 34 | 3.891 | 5.539 | 3.96 | 46.7 % | 0.99 |
| observance-tier depth | 4 | 18 | 4.218 | 5.423 | 4.01 | 46.1 % | 0.55 |
| shape (Pearson, F/G/H) | 3 | 25 | 4.554 | 5.932 | 4.67 | 55.3 % | 0.098 |
| best-matching weekday | 7 | 9 | 4.642 | 5.710 | 4.54 | 55.3 % | 0.22 |
| seasonal (winter/spring/fall) | 3 | 35 | 4.917 | 6.863 | 4.62 | 59.9 % | $2.8\times10^{-4}$ |
| seasonal (heat/cold) | 2 | 40 | 5.633 | 7.025 | 10.28 | 61.2 % | $9.9\times10^{-5}$ |

Two readings must be separated here, and we state the negative one first because it constrains what may
be claimed. **The difference between the two leading criteria is not statistically significant**:
holiday identity improves the median by 0.10 pp over observance tier, but it wins only 46.7 % of the
paired cells and the Wilcoxon test returns $p=0.99$. On this evidence the two are tied, and the choice
between them should be made on grounds other than accuracy — we retain holiday identity because it
requires no tuned thresholds and no observance model, and is therefore simpler to specify and to port to
a new system. The claim that *is* supported is the ordering against the seasonal criteria, which are
significantly worse ($p<10^{-3}$).

The table also exhibits a clear monotone relation between the coarseness of the conditioning and the
number of neighbours the search selects: as the median pool grows from 6 to 40, mean $k$ grows from 2.34
to 10.28 and the median error grows with it. This is the mechanism behind the criterion ranking — a
coarse cluster does not merely admit more candidates, it induces the search to *use* more of them, and
the additional neighbours are by construction less similar to the target.

### B. Selected hyper-parameters

Across the 152 Mexican cells the per-target search selected Ridge 74 times, Lasso 71, PLS 6 and PCR 1;
across the 171 ERCOT cells, Ridge 147, PCR 8, Lasso 8 and PLS 8. Pearson similarity was preferred in
56 % of Mexican cells and 87 % of ERCOT cells. The selected $k$ was 2–3 in Mexico (mean 2.34) and 2–6 in
ERCOT (mean 4.65), the difference reflecting the deeper ERCOT history. As stated in Section IV-D, these
frequencies are reported as a description of the search's behaviour and not as evidence that one
regressor is superior; the pre-correction code could not have supported such a claim, and the corrected
code has not been used to test it.

### C. Comparison against seasonal naive baselines

Analog-holidays outperforms every seasonal naive variant on the Mexican panel. Table IV reports the four
strongest baselines of the eighteen evaluated.

**TABLE IV. Analog-holidays vs the strongest seasonal naive baselines (152 paired cells).**

| Baseline | baseline median MAPE | analogue median MAPE | cells won | skill | $p$ |
|---|---|---|---|---|---|
| mean of last 2 instances of the same holiday | 5.570 | **3.789** | 63.2 % | +0.320 | $9.6\times10^{-6}$ |
| median of prior instances of the same holiday | 6.078 | **3.789** | 73.0 % | +0.377 | $3.7\times10^{-9}$ |
| most recent prior instance of the same holiday | 6.279 | **3.789** | 70.4 % | +0.397 | $1.2\times10^{-9}$ |
| mean of prior instances of the same holiday | 7.298 | **3.789** | 73.7 % | +0.481 | $1.1\times10^{-10}$ |

The strongest naive baseline is the mean of the two most recent instances of the same holiday (5.570 %);
averaging more instances degrades it, because of demand growth and the excluded period. The analogue
method improves on it by 1.78 pp, a skill of +0.320, winning 63 % of cells at $p\approx10^{-5}$.

### D. Comparison against the classical Similar-Days method

Table V reports the benchmark against the 54-configuration Similar-Days grid, on both systems.

**TABLE V. Analog-holidays vs classical Similar-Days (best of 54 configurations).**

| | Mexico (152 cells) | ERCOT (171 cells) |
|---|---|---|
| Analog-holidays | **3.789 %** | **6.039 %** |
| Similar-Days, best configuration | 4.279 % (`ctx`, scaled, $k{=}5$) | 7.166 % (`hybrid`, scaled, $k{=}4$) |
| SD candidate pool per cell (median) | 67 | 143 |
| cells won by the analogue method | 62.5 % | 65.5 % |
| median paired difference | −0.465 pp | −1.234 pp |
| skill | **+0.114** | **+0.157** |
| $p$ (Wilcoxon signed-rank) | $7.6\times10^{-5}$ | $8.5\times10^{-6}$ |

The result replicates on the second system with a *larger* margin, despite Similar-Days drawing on
eight to sixteen times more candidates per cell in ERCOT. Two secondary observations are consistent
across both systems. First, the **level-renormalisation combination rule dominates**: within Similar-Days
the best `scaled` configuration beats the best `mean` configuration by 0.79 pp in Mexico (4.279 vs
5.069) and by 1.94 pp in ERCOT (7.166 vs 9.110). Without it, Similar-Days is no better than a seasonal
naive. Second, the entire advantage of the analogue method is in the level and not the shape, which we
quantify next.

One secondary observation does **not** replicate, and we flag it explicitly rather than suppress it. In
Mexico the analogue advantage is concentrated in the December–January block, consistent with the
hypothesis that the method wins where a holiday sits inside a multi-day atypical sequence whose
pre-holiday trajectory carries the information. In ERCOT that pattern does not hold: New Year's Day
*inverts* to favour Similar-Days, the Christmas block yields only a modest advantage, and the isolated
federal Mondays (Presidents' Day, Martin Luther King Jr. Day, Memorial Day) favour the analogue method —
the opposite of the Mexican pattern for isolated civic holidays. Only Day after Thanksgiving, the
clearest multi-day sequence in the ERCOT calendar, points the expected way. Within Similar-Days the
preferred distance family also differs between systems (`ctx` in Mexico, `hybrid` in ERCOT). We
therefore report the comparative result as established and the *mechanism* behind it as an open question
supported in one system and not the other.

### E. What kind of error remains: level, not shape

Table VI decomposes the residual error using the level/shape split of Section IV-C.

**TABLE VI. Level/shape decomposition of the residual error (median over cells, pp).**

| System | Method | MAPE | level $\lvert\mathrm{MPE}\rvert$ | shape | ratio $\lvert\mathrm{MPE}\rvert/\mathrm{MAPE}$ |
|---|---|---|---|---|---|
| SEN | Analog-holidays | 3.789 | 3.014 | 0.775 | 0.905 |
| SEN | Similar-Days (best) | 4.279 | 3.478 | 0.801 | — |
| ERCOT | Analog-holidays | 6.039 | 4.586 | 1.453 | 0.944 |
| ERCOT | Similar-Days (best) | 7.166 | 6.142 | 1.024 | — |

The forecast has essentially the right shape at the wrong altitude: 90–94 % of the absolute error is a
uniform vertical displacement. In Mexico the two methods have an *identical* shape error (0.775 vs
0.801 pp) and the entire 0.49 pp advantage is level. In ERCOT the analogue method is in fact *worse* on
shape (1.453 vs 1.024 pp) and wins by a large enough level margin (4.586 vs 6.142) to overturn it.

This finding reorients the problem. The shape of a holiday profile is, in both systems, essentially
solved by the analogue machinery; what remains unsolved is predicting the *altitude* at which that
profile will sit. It also explains the negative result of Section III-G: a zero-mean intraday bias
correction operates on the shape component, which is the component that is already small.

A further diagnostic sharpens the target. Across the 26 controlled experiments performed in this work —
spanning clustering criteria, neighbour ceilings, separation windows and regressor menus — the choice of
method configuration explains **0.4 %** of the variance of the level bias, whereas the (region, date)
cell identity explains 78–80 % of it. The residual bias is therefore not a property of the model but of
the day and place, which is what led us to look for an exogenous cause.

We also evaluated a *level anchor* built from the 14 pre-holiday hours: rescaling the holiday profile by
the bias observed over the pre-holiday window reduces the level error by about 25 % and the median MAPE
from 3.79 % to 3.20 %. **This result is reported as a diagnostic and excluded from the method**, because
those 14 hours are not observable at the issue instant defined in Section III-B; using them would be a
look-ahead. We tested two ex-ante replacements that use only information available at issue time, and
both are significantly worse than no correction at all, because the regression intercept already anchors
the forecast to the current level and the additional factor double-counts it. The 0.59 pp gap between
the anchored and unanchored results is therefore best read as a *measure of how much same-day level
information the method is missing*, not as an available improvement.

### F. How much of the residual level bias is weather

The residual level bias behaves like a same-day, system-wide shock rather than a per-region model
failure: the mean pairwise correlation of cell-level MPE across regions on the same date is 0.455 in
ERCOT and 0.204 in Mexico, and the date-common component accounts for 58.1 % and 25.1 % of the bias
variance respectively. This is the signature of a common exogenous driver.

Because the ERCOT load zones are the system's own *weather* zones, each zone admits a representative
temperature without any load-allocation step. We therefore obtained hourly temperature for eighteen
population-weighted urban points aggregated to the eight zones, from two sources: ERA5 reanalysis
(observed, 2016–2026) and, critically, the **archived model forecasts at $D{-}1$ and $D{-}2$ lead** — that
is, what a forecaster actually held at the issue instant. The predictor is the *degree-day anomaly*: the
daily mean of $|T - 18.3\,^\circ\mathrm{C}|$ minus a $\pm7$-day-smoothed day-of-year climatology. Raw
temperature is a much weaker predictor, because demand rises on both sides of the balance point and the
two directions partly cancel.

**TABLE VII. Variance of the level bias explained by the degree-day anomaly (ERCOT).**

| Temperature source | Admissible at issue time | per cell ($n{=}171$), Pearson $R^2$ | Spearman $R^2$ | $p$ | system shock ($n{=}19$), $R^2$ |
|---|---|---|---|---|---|
| ERA5 observed | no (oracle) | 24.5 % | 18.4 % | $4.7\times10^{-9}$ | 42.3 % |
| **forecast, $D{-}1$ lead** | **yes** | **21.7 %** | **16.5 %** | $3.4\times10^{-8}$ | 34.8 % |
| forecast, $D{-}2$ lead | yes | 19.9 % | 16.7 % | $2.8\times10^{-8}$ | 33.0 % |

Three conclusions follow. First, the residual level bias **is** substantially weather: a single exogenous
variable explains 18–25 % of its variance, against 0.4 % for the entire space of method configurations
explored in this work. Second, **the signal survives the operational constraint**: at $D{-}1$ lead the
weather forecast has a mean absolute error of 0.72 degree-days and a correlation of 0.979 with the
observed value, and about 89 % of the explanatory power is retained. Unlike the level anchor of Section
V-E, this correction is admissible. Third, the two extreme cells of the panel are both weather events in
opposite directions — the January 2025 cold snap (degree-day anomaly +9.7, under-forecast +16.5 %) and a
mild New Year 2026 (over-forecast −18.6 %) — which is the mechanism made visible.

Two honest qualifications. The 42.3 % figure for the system-wide shock rests on $n=19$ dates and leans on
the January 2025 cold snap; excluding it, the figure falls to 12.1 % and loses significance ($p=0.11$).
The defensible number is the per-cell one, which is robust ($n=171$, rank-based, $p<10^{-7}$). Separately,
we tested whether what matters is the *change* in temperature between the observed pre-holiday window and
the holiday — the quantity the method structurally cannot know — and it is not: that predictor explains
only 1–3 %. What matters is the level of the anomaly, not its change across the issue boundary.

## VI. Discussion

The Analogue method has a physical and causal justification: electricity demand exhibits structural
recurrence, repetitive social profiles and repetitive economic behaviour. Compared with ML or DL
approaches, which require a large number of event samples, holidays are scarce events; the analogue
method works precisely in low-data regimes, reducing the search space, avoiding massive training, and
incurring a near-negligible computational cost.

**On the neighbour pool.** The criterion sweep of Section V-A supports a single mechanism: the error
grows with the number of neighbours the search actually uses, and coarse clusters induce the search to
use more. This also explains a result we previously attributed to a different cause. Widening the pool
by relaxing the minimum separation between candidate events from 24 h to 12 h degrades the deep
holidays; we had explained this as dilution of the sharp demand drop by less-similar neighbours. The
audit of Section IV-D shows the explanation was incomplete: at 12 h separation, candidates were also
being admitted at the wrong phase of the day, and candidates spanning the excluded periods were being
spliced. Both are now excluded structurally (Section III-E), and the 24-h separation is retained on the
evidence of the sweep rather than on the earlier reasoning.

**On the regional gap.** An earlier version of this work described the error in the weather-sensitive
Mexican regions as an irreducible noise floor. Section V-F shows this framing was wrong, or at least
premature: in ERCOT, where zonal temperature is directly available, a single exogenous variable explains
a fifth of the bias variance, and the regions where the method performs worst are precisely the ones
whose demand is most temperature-elastic. The gap is not irreducible noise; it is a missing covariate.
We expect the same to hold for NES, NTE and PEN, though we have not yet obtained the corresponding
Mexican weather series and therefore do not claim it.

**On what the comparison establishes and what it does not.** Analog-holidays outperforms both a family
of seasonal naive baselines and the best of 54 classical Similar-Days configurations, on two independent
systems, under a paired test, with the comparison biased in favour of the baselines. That claim is
robust. The *explanation* we favour — that the advantage arises where a holiday sits inside a multi-day
atypical sequence — is supported in Mexico and contradicted in ERCOT (Section V-D), and we present it as
a hypothesis rather than a finding. Both methods, moreover, operate without weather; since Similar-Days
is a day-descriptor method it would likely gain more from temperature than the analogue method would, so
the defensible statement is that the analogue method is superior *given the same information set*, not
in general.

**On the location of the remaining error.** The most consequential result of this work is arguably
negative. The shape of the holiday profile is solved; the altitude is not, and the altitude is where
roughly 90 % of the error lives. Method-space search is exhausted for this component — twenty-six controlled
experiments moved 0.4 % of its variance. Progress requires exogenous information, and Section V-F
identifies which: temperature, in degree-day-anomaly form, available at issue time with 89 % of its
explanatory power intact.

## VII. Conclusions

We presented a holiday-conditioned adaptation of the Analogue method for day-ahead, hourly
electricity-demand forecasting on public holidays. The method models the pre-holiday→holiday transition
through conditioned pairs $(Z, Z')$, restricts the neighbour search to a curated, holiday-clustered set,
enforces phase alignment and calendar-seam admissibility on the candidates, and tunes its
hyper-parameters per target.

On the Mexican interconnected system it attains a median MAPE of 3.79 % over 152 test cells, and on
ERCOT 6.04 % over 171 cells, in both cases with negligible computation cost and no failures. It
outperforms every seasonal naive baseline (best: 5.57 %, skill +0.320) and the best of 54 classical
Similar-Days configurations on both systems (Mexico 4.28 %, skill +0.114, $p=8\times10^{-5}$; ERCOT
7.17 %, skill +0.157, $p=8\times10^{-6}$), with the comparison deliberately biased toward the baselines.

We further establish where the remaining error is and what it is made of. It is a level bias, not a
shape error, in both systems; it is a same-day, system-wide shock rather than a per-region model
failure; and in ERCOT a single exogenous variable — the degree-day anomaly — explains 18–25 % of its
variance, against 0.4 % for the entire space of method configurations explored here. Crucially, that
signal is still present when the variable is restricted to what a forecaster actually holds at issue
time. This establishes both the accuracy ceiling attainable without exogenous variables and the specific
covariate required to move it. The method offers higher accuracy in sparse-event regimes, low
complexity, efficient training, robustness and interpretability, and its findings can be useful for ISOs
such as CENACE and ERCOT and for power-system operators generally.

### Future work

The immediate priority is to **integrate the degree-day anomaly as a level correction** using
issue-time-admissible forecast temperature, and to measure the resulting reduction in MAPE; Section V-F
establishes the signal exists but stops short of exploiting it. Obtaining the corresponding weather
series for the Mexican control regions would allow the same analysis there, and would test the
conjecture that the NES/NTE/PEN gap is a missing covariate rather than noise.

A second priority is a benchmark against the **operational forecast of the system operator itself**;
ERCOT publishes a seven-day load forecast by weather zone, which would place the method against the
incumbent rather than against reconstructed baselines. A broader benchmark against calendar-aware ML and
deep-learning models and against foundation time-series models is also planned.

The Similar-Days benchmark motivates a per-holiday *hybrid* that routes stable civic holidays to a
same-holiday average profile and the remaining holidays to the analogue model; the divergence of the
per-holiday pattern between the two systems (Section V-D) means such a rule must be validated on
held-out years and on both systems before adoption. A further direction is a rule-based or learned
analogue *selector* that chooses the neighbour subset by calendar context (weekday incidence, holiday
type, whether the holiday follows another holiday, the season, and whether it is fixed or shifted to
Monday by the 2006 Mexican law). Finally, holidays not only complicate the forecast as special days but
also inject noise into the following, post-holiday recovery days; a subsequent study may target those
recovery days through imputation and tagging strategies.

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

Code: `https://github.com/urieliram/analog_holidays`. The reproducible drivers are under `experiments/`:
`run_criterion.py` and `run_criterion_ercot.py` reproduce the Mexican and ERCOT panels for any of the
seven clustering criteria; `similar_days_benchmark.py` reproduces the 54-configuration Similar-Days grid
on either system; `seasonal_naive_ceiling.py` reproduces the naive baselines; `fetch_weather_ercot.py`
and `fetch_weather_forecast_ercot.py` retrieve the observed and archived-forecast temperature series;
`analyze_weather_bias.py` and `analyze_weather_exante.py` reproduce Table VII.

The complete configuration required to reproduce the headline result, and its comparison against the
seasonal naive family, is documented in `docs/RESULTADO_TECHO.md`; the Similar-Days benchmark and its
ERCOT replication are in `docs/BENCHMARK_SIMILAR_DAYS.md`. Per-cell result tables are in
`docs/similar_days_benchmark.csv`, `docs/similar_days_benchmark_ercot.csv`,
`docs/seasonal_naive_ceiling.csv`, `docs/weather_bias_cells.csv` and `docs/weather_exante_cells.csv`.
Every experiment reported here retains its metrics, manifest and notes under
`experiments/experiment_<timestamp>_<name>/`; the champion runs are
`experiment_2026_08_25_00_37_criterion_holiday_identity` (Mexico) and
`experiment_2026_08_31_06_38_ercot_criterion_holiday_identity` (ERCOT). Temperature data are from the
ERA5 reanalysis and archived operational forecasts, retrieved through the Open-Meteo API.

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

[8] H. Hersbach *et al.*, "The ERA5 global reanalysis," *Q. J. R. Meteorol. Soc.*, vol. 146, no. 730,
pp. 1999–2049, 2020.
