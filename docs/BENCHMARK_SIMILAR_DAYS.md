# Benchmark: analog-holidays vs Similar-Days clásico

**Fecha:** 2026-08-30 · **Panel:** MX (SEN), 8 series × 19 fechas = **152 celdas pareadas**
**Script:** `experiments/similar_days_benchmark.py` · **Datos celda a celda:** `docs/similar_days_benchmark.csv`

Responde a la sugerencia de Mónica Boruinda: contrastar el método propuesto contra el
Similar-Days tradicional, no solo contra líneas base naive.

---

## 1. Veredicto

| Método | mediana MAPE_24 |
|---|---|
| **analog-holidays** (campeón, ver `RESULTADO_TECHO.md`) | **3.789 %** |
| Similar-Days, mejor de 54 configuraciones (`ctx_scaled_k5`) | 4.279 % |
| Similar-Days, configuración por defecto razonable (`ctx_scaled_k3`) | 4.377 % |
| Naive estacional más fuerte (media de 2 instancias previas) | 5.570 % |

| Comparación pareada | analog gana | delta mediana | skill | p (Wilcoxon) |
|---|---|---|---|---|
| vs Similar-Days **oráculo** | 62 % | −0.465 pp | **+0.114** | **0.00008** |
| vs Similar-Days **por defecto** | 64 % | −0.588 pp | **+0.134** | **0.00003** |

**analog-holidays le gana a Similar-Days de forma estadísticamente significativa**, y lo hace
contra una versión de Similar-Days deliberadamente favorecida (§4).

---

## 2. Qué se implementó

Siguiendo `similar_days_holidays.md` al pie de la letra.

### Distancias (§2 del doc metodológico)

| Variante | Definición |
|---|---|
| **`ctx`** (Opción A) | suma ponderada sobre variables de contexto: tipo de festivo (0 mismo / 0.3 misma familia / 1 distinta), día de semana (0 / 0.3 ambos fin de semana / 1), mes circular normalizado, nivel reciente de demanda |
| **`shape`** (Opción B) | `1 − Pearson` sobre la firma reciente de 38 h antes de emitir |
| **`hybrid`** (Opción C) | `α·ctx + (1−α)·shape`, α = 0.5, ambas normalizadas |

Familias de festivo usadas: `invierno` (Nochebuena, Navidad, Fin de Año, Año Nuevo),
`semana_santa` (Jueves y Viernes Santo, Sábado de Gloria), `civico` (Constitución, Juárez,
Trabajo, Independencia, Revolución).

### Combinaciones (§3 del doc metodológico)

| Variante | Definición |
|---|---|
| **`mean`** | promedio simple de los k perfiles vecinos (§3.1) |
| **`wmean`** | promedio ponderado, kernel exponencial sobre la distancia (§3.2) |
| **`scaled`** | ponderado **y renormalizado por nivel**: cada vecino se reescala por `nivel_reciente_objetivo / nivel_reciente_vecino` antes de promediar (§3.4) |

**Rejilla completa:** 3 distancias × 3 combinaciones × k ∈ {1,…,6} = **54 configuraciones**.

### Igualdad de condiciones

- Mismas 152 celdas, mismos perfiles reales, misma métrica (MAPE sobre las 24 h del festivo).
- Mismo corte temporal: candidatos **estrictamente anteriores** a la fecha objetivo; la ventana
  reciente termina a las **10:00 del día previo** (momento de emisión). Cero información futura.
- Candidatos disponibles por celda: mediana **67** (min 56, max 79).

### ⚠️ Restricción honesta: sin clima

Cuando se corrió este benchmark el repo no tenía ninguna variable meteorológica, así que el bloque
de clima del vector de contexto (§2.1a) **se omitió en ambos métodos**. Los dos operan con el mismo
conjunto de información —calendario + historia de demanda— que es lo que hace justa la comparación.

**Este caveat sigue vigente y está en el paper** (§V-D y §VI). Similar-Days es, por diseño, un
método de descriptores a nivel día, así que probablemente se beneficiaría *más* que analog-holidays
de incorporar temperatura. La conclusión defendible es *"con el mismo conjunto de información,
analog-holidays domina"*, no *"analog-holidays domina a Similar-Days en general"*.

> **Actualización 2026-09-02 — el clima ya se midió, pero eso NO invalida ni cambia este benchmark.**
> Se obtuvo temperatura horaria para las zonas de ERCOT y se cuantificó que la anomalía de
> grados-día explica 18–25 % de la varianza del sesgo de nivel (ver `RESULTADO_TECHO.md` §8.5 y el
> paper §V-F). Ese análisis es *diagnóstico*: mide qué parte del error residual es clima. **Ninguno
> de los dos métodos de esta comparación usa temperatura**, así que las cifras de abajo no cambian
> y la igualdad de condiciones se mantiene. Lo que sí cambia es la lectura del caveat: ahora sabemos
> el tamaño aproximado del premio que Similar-Days podría reclamar si se le diera clima, y sigue
> siendo una comparación pendiente, no una que ya perdimos.

---

## 3. El hallazgo clave: dónde gana cada uno

La ventaja **no está repartida** — se concentra donde la trayectoria previa al festivo carga
información.

| Festivo | analog | Similar-Days | skill |
|---|---|---|---|
| **Christmas Day** | 4.011 % | **15.256 %** | **+0.737** |
| **New Year's Eve** | 4.022 % | 11.365 % | **+0.646** |
| **New Year's Day** | 5.409 % | 12.604 % | **+0.571** |
| **Christmas Eve** | 4.043 % | 7.493 % | +0.460 |
| Holy Saturday | 3.512 % | 5.956 % | +0.410 |
| Good Friday | 3.700 % | 4.759 % | +0.223 |
| Constitution Day | 2.670 % | 3.282 % | +0.186 |
| Mexican Revolution Day | 2.323 % | 2.715 % | +0.144 |
| Maundy Thursday | 2.926 % | 3.291 % | +0.111 |
| Benito Juarez's Birthday | 3.471 % | 3.488 % | +0.005 |
| Labor Day | 4.087 % | 3.791 % | −0.078 |
| **Independence Day** | 4.550 % | **3.161 %** | **−0.440** |

**Interpretación.** analog-holidays arrasa en el bloque **diciembre–enero** (Nochebuena → Navidad →
Fin de Año → Año Nuevo), donde el festivo está inmerso en una secuencia de varios días atípicos
consecutivos y la víspera contiene muchísima información sobre lo que viene. Similar-Days, que sólo
compara descriptores del día, no puede ver esa transición y se equivoca por 11–15 %.

En cambio, en festivos **cívicos aislados** —Independencia, Trabajo— rodeados de días normales,
Similar-Days iguala o gana: ahí el contexto de calendario es suficiente y la búsqueda por forma no
aporta.

**Esto es exactamente la tesis de `similar_days_vs_analog.md` §4, ahora demostrada empíricamente
festivo por festivo.** Es el argumento más fuerte que tiene el paper.

### Por región

| Región | analog | Similar-Days | skill |
|---|---|---|---|
| ORI | 3.507 % | 6.572 % | +0.466 |
| SIN | 2.406 % | 3.142 % | +0.234 |
| NOR | 4.097 % | 5.275 % | +0.223 |
| CEL | 2.224 % | 2.657 % | +0.163 |
| NTE | 4.404 % | 4.757 % | +0.074 |
| NES | 6.696 % | 7.153 % | +0.064 |
| OCC | 3.199 % | 3.171 % | −0.009 |
| PEN | 8.541 % | 7.712 % | −0.107 |

Positivo en 6 de 8 regiones. PEN —la más volátil y presumiblemente la más sensible al clima— es la
única donde Similar-Days gana con claridad.

---

## 4. Por qué la comparación es conservadora (favorece a Similar-Days)

Tres decisiones deliberadas para no montar un hombre de paja:

1. **Se le dio ventaja de oráculo.** La configuración reportada (`ctx_scaled_k5`) es la mejor de las
   54 **elegida a posteriori sobre el mismo panel de prueba**. analog-holidays no recibe ese trato:
   su config sale de tuning Optuna por objetivo, sin ver el resultado. Aun así analog gana. Con una
   config por defecto razonable la ventaja de analog crece (+0.134).
2. **Se implementó la renormalización de nivel (§3.4)**, que resulta ser el lever más importante de
   Similar-Days (tabla abajo). Omitirla habría hecho parecer a Similar-Days mucho peor de lo que es.
3. **Pool amplio**: 67 candidatos por celda en mediana, contra los ~6 del filtro duro del campeón.

### La renormalización de nivel es decisiva para Similar-Days

| distancia | k | sin escalar | escalado | ganancia |
|---|---|---|---|---|
| ctx | 2 | 5.556 % | 4.429 % | **−1.127 pp** |
| ctx | 3 | 5.296 % | 4.377 % | −0.919 pp |
| ctx | 4 | 5.630 % | 4.510 % | −1.120 pp |
| shape | 2 | 5.692 % | 5.284 % | −0.408 pp |
| shape | 3 | 5.853 % | 5.339 % | −0.514 pp |
| shape | 4 | 5.945 % | 5.374 % | −0.571 pp |
| hybrid | 2 | 5.515 % | 4.738 % | −0.777 pp |
| hybrid | 3 | 5.420 % | 4.820 % | −0.599 pp |
| hybrid | 4 | 5.556 % | 4.662 % | −0.894 pp |

Consistente con el diagnóstico general del proyecto: **el error es de nivel, no de forma.** Un
Similar-Days que promedia perfiles crudos hereda el nivel de los vecinos; reescalarlo al nivel actual
del sistema le quita ~1 pp. Sin ese paso, Similar-Days queda al nivel del naive estacional (5.3–5.9 %).

### Descomposición nivel/forma

| Método | MAPE | nivel (\|MPE\|) | forma |
|---|---|---|---|
| analog-holidays | 3.789 % | **3.014 %** | 0.775 pp |
| Similar-Days (mejor) | 4.279 % | 3.478 % | 0.801 pp |

Ambos tienen prácticamente el mismo error de forma (~0.8 pp). **Toda la ventaja de analog-holidays
está en el nivel.**

### Cuál distancia gana dentro de Similar-Days

| familia | mejor config | mediana |
|---|---|---|
| `ctx` | ctx_scaled_k5 | **4.279 %** |
| `hybrid` | hybrid_scaled_k6 | 4.487 % |
| `shape` | shape_scaled_k1 | 5.277 % |

Dentro de Similar-Days, el **contexto de calendario supera a la forma reciente**. Nótese el contraste:
analog-holidays *también* usa forma, y gana. La diferencia no es la señal sino cómo se combina —
regresión local (PCR/PLS/Ridge/Lasso) contra promedio ponderado.

---

## 4-bis. Réplica en ERCOT — el titular se sostiene, el mecanismo no

Segundo sistema, calendario distinto, clima distinto: **9 series × 19 fechas = 171 celdas**,
misma maquinaria (`experiments/run_criterion_ercot.py` + `similar_days_benchmark.py ercot`).
ERCOT tiene más historia (2016 en adelante contra 2020 en MX): 15 clusters, pool 9–18 por serie,
y Similar-Days dispone de **143 candidatos** por celda en mediana.

| | MX | ERCOT |
|---|---|---|
| analog-holidays | 3.789 % | **6.039 %** |
| Similar-Days (mejor de 54) | 4.279 % | 7.166 % (`hybrid_scaled_k4`) |
| **skill** | **+0.114** | **+0.157** |
| analog gana | 62 % | **65 %** |
| p (Wilcoxon) | 0.00008 | **0.00001** |

### ✅ Lo que replica

1. **analog-holidays gana, y con margen mayor** (+0.157 vs +0.114), sobre 171 celdas independientes.
2. **La renormalización de nivel sigue siendo decisiva para Similar-Days**, y aún más fuerte:
   **−2.55 pp** en ERCOT (`ctx k=2`: 11.426 % → 8.875 %) contra −1.13 pp en MX. Sin ese paso
   Similar-Days es inservible en ERCOT.
3. **La ventaja de analog vuelve a estar en el nivel**: nivel 4.586 % contra 6.142 % de Similar-Days.

### ❌ Lo que NO replica

**El patrón por festivo es distinto, y eso debilita la explicación mecanicista.**

| ERCOT | analog | SD | skill |
|---|---|---|---|
| **Day after Thanksgiving** | 2.141 % | 4.585 % | **+0.533** |
| Memorial Day | 8.401 % | 16.400 % | +0.488 |
| Lyndon B. Johnson Day | 2.129 % | 3.686 % | +0.423 |
| San Jacinto Day | 6.039 % | 10.448 % | +0.422 |
| Presidents' Day | 7.070 % | 11.555 % | +0.388 |
| Veterans Day | 5.941 % | 9.564 % | +0.379 |
| Labor Day | 5.532 % | 7.978 % | +0.307 |
| Martin Luther King Jr. Day | 11.478 % | 16.413 % | +0.301 |
| Christmas Day | 5.473 % | 7.384 % | +0.259 |
| Christmas Eve | 5.365 % | 6.657 % | +0.194 |
| Thanksgiving Day | 4.586 % | 5.135 % | +0.107 |
| Texas Independence Day | 6.046 % | 6.612 % | +0.086 |
| Independence Day | 8.030 % | 7.589 % | −0.058 |
| Juneteenth | 4.948 % | 3.427 % | −0.444 |
| **New Year's Day** | 11.942 % | 8.083 % | **−0.477** |

Contradicciones concretas con MX:

- **Año Nuevo se invierte**: en MX el analog ganaba con skill **+0.571**; en ERCOT **pierde** con
  −0.477. No es escasez de pool (4.61 candidatos, en el rango medio de ERCOT) — la hipótesis se
  verificó y se descartó.
- **El bloque navideño es apenas modesto** en ERCOT (Navidad +0.259, Nochebuena +0.194) contra el
  dominio aplastante en MX (+0.737, +0.460).
- **Los lunes federales aislados favorecen al analog** (Presidents' +0.388, MLK +0.301, Memorial
  +0.488), justo lo contrario del patrón mexicano, donde los cívicos aislados favorecían a
  Similar-Days.
- Lo único que sí apunta en la dirección esperada es **Day after Thanksgiving (+0.533)**, la
  secuencia multi-día más clara del calendario ERCOT, que resulta ser la mayor ventaja del panel.

También cambia la descomposición: en MX ambos métodos tenían el mismo error de forma (~0.8 pp); en
ERCOT **el analog tiene peor forma** (1.453 pp contra 1.024 pp) y compensa con mucho mejor nivel.

### Qué implica para el paper

**La afirmación robusta —y la única que sostendría ante un revisor— es que analog-holidays supera a
Similar-Days en dos sistemas independientes, con el mismo conjunto de información.** Eso replica
limpiamente y con margen creciente.

**La explicación mecanicista de la §3 (la ventaja viene de las secuencias multi-día) está respaldada
en MX pero no en ERCOT**, así que hay que presentarla como hipótesis con evidencia mixta, no como
hallazgo establecido. Escribirla como conclusión sería sobre-interpretar un solo sistema.

Nota adicional: dentro de Similar-Days, ERCOT prefiere la distancia **híbrida** (7.166 %) sobre la de
contexto (8.267 %), mientras MX prefería la de contexto puro (4.279 %). Otra señal de que la
estructura de similitud no es universal entre sistemas.

---

## 5. Redacción para el paper — **ya incorporada**

> **Estado:** este texto ya está integrado en `paper/IEEE_PES/analog_holidays.md` §V-D, con las
> tablas correspondientes (Tabla V) y la salvedad del mecanismo en §VI. Se conserva aquí como
> referencia de la redacción acordada.


> We benchmark the proposed method against a classical Similar-Days implementation following the
> canonical five-step formulation, with three distance definitions (calendar-context, recent-shape,
> and a hybrid), three combination rules (simple mean, distance-weighted mean, and level-renormalised
> weighted mean), and k ∈ {1,…,6} — 54 configurations in total. Both methods are restricted to the
> same information set (calendar and demand history; no weather covariates are available) and are
> scored on the identical 152-cell panel with the same issue-time cutoff. Against the best-performing
> Similar-Days configuration, selected post hoc on the test panel and therefore favourable to the
> baseline, the proposed method attains a median MAPE of 3.79 % versus 4.28 % (skill +0.11, paired
> Wilcoxon p = 8·10⁻⁵). The advantage is strongly concentrated in the December–January holiday block
> — Christmas Day +0.74, New Year's Eve +0.65, New Year's Day +0.57 — where the target is embedded in
> a multi-day sequence of atypical days that day-level descriptors cannot represent, while for
> isolated civic holidays such as Independence Day the calendar-context baseline is competitive or
> better. The experiment was replicated on an independent system (ERCOT, 9 zones × 19 holidays = 171
> cells, a different holiday calendar and climate), where the proposed method again outperforms the
> best Similar-Days configuration (6.04 % vs 7.17 %, skill +0.16, p = 1·10⁻⁵). In both systems the
> entire advantage is attributable to the level component rather than to profile shape, and in both
> the level renormalisation of the Similar-Days baseline is essential — omitting it degrades it to
> seasonal-naive performance. The per-holiday distribution of the advantage, however, differs between
> the two systems, so the mechanism behind the gain remains an open question rather than an
> established result.

---

## 6. Reproducción

```bash
cd /home/uriel/GIT
MPLBACKEND=Agg python3 analog_holidays/experiments/similar_days_benchmark.py
```

Requiere que exista la corrida campeona (`experiment_*_criterion_holiday_identity`); ver
`RESULTADO_TECHO.md` §10. Salida: tabla en consola + `docs/similar_days_benchmark.csv` con las 152
celdas y las 54 configuraciones.
