# Resultado techo sin variables exógenas — receta exacta y reproducible

**Fecha:** 2026-08-25 · **Rama:** `analog_holiday_H1_H2_H3` · **Dataset:** MX (SEN)

Este documento contiene **todo** lo necesario para reproducir el mejor resultado alcanzable
con el método analógico usando únicamente historia de demanda y calendario, y compararlo
contra la línea base naive estacional. Nada queda implícito.

---

## 0. Resumen ejecutivo

| Configuración | mediana MAPE_24 | media |
|---|---|---|
| **TECHO OPERATIVO** (`holiday_identity`, sin ancla) | **3.789 %** | 5.536 % |
| Línea base previa (`observance_tier`, código con bugs) | 3.739 % | 5.336 % |
| Naive estacional más fuerte (media/mediana de las 2 instancias previas) | 5.570 % | — |
| Naive estacional "mismo día del calendario", mediana | 8.640 % | — |
| *(cota superior teórica, NO operativa: + ancla de nivel)* | *3.201 %* | *4.528 %* |

**Skill contra el naive más fuerte: +0.320** (el modelo gana en el 63 % de las celdas, p = 0.00001).
Contra el naive por fecha de calendario, skill **+0.561**.

> ⚠️ **Sobre la fila 2.** El techo operativo (3.789 %) no es mejor que la línea base previa
> (3.739 %) en la mediana. La diferencia **no es significativa** y ambas están dentro del piso de
> ruido. Lo que cambió es que el pipeline ahora es *correcto*: ver §6.

> ⛔ **Sobre la última fila.** El ancla de nivel da 3.201 % pero **no es usable en operación**:
> requiere observar la tarde-noche del día previo, que no ha ocurrido cuando se emite el
> pronóstico. Se documenta sólo como cota superior. Ver §8.

> ✅ **Actualización 2026-09-02.** El techo de este documento sigue en pie: 3.789 % es lo máximo
> alcanzable sin variables exógenas. Lo que cambió es que **la variable exógena que falta ya está
> identificada y medida**: la anomalía de grados-día explica 18–25 % de la varianza del sesgo de
> nivel, contra 0.4 % de todo el espacio de configuraciones del método, y **sobrevive a la
> restricción operativa** (89 % de la señal se conserva usando pronóstico de temperatura de D−1,
> no observación). Ver §8.5, que dejó de ser una conjetura y pasó a ser un resultado.

---

## 1. Datos de entrada

| Ítem | Valor |
|---|---|
| Archivo | `analog_holidays/holidays/holiday_demand_mx.csv` |
| MD5 | `3976461335850dd160342b0de643a676` |
| Rango nominal | 2020-01-02 00:00 → 2027-12-31 23:00 (61 296 filas horarias) |
| Último dato **real** | 2026-05-17 23:00 (las filas posteriores están vacías; son relleno de calendario, **no** hay fuga) |
| Días completos utilizables | 1 961 |
| Series de demanda | 8 (`SEN_demand_{CEL,NES,NOR,NTE,OCC,ORI,PEN,SIN}`) |
| Formato | `ds` + una columna por región + una columna `<region>_holiday` (bandera 0/1) |

### Exclusiones deliberadas (NO son datos faltantes)

Los días de cuarentena se excluyeron a propósito por estar contaminados por comportamiento
atípico. Las costuras resultantes son:

| Costura | Días omitidos |
|---|---|
| 2021-12-31 → 2023-01-01 | 365 (todo 2022) |
| 2020-04-05 → 2020-04-07 | 1 |
| 2021-04-04 → 2021-04-06 | 1 |

Estas costuras **importan** porque el pipeline aplana los días en un vector horario continuo:
ver el arreglo #5 en §6.

### Tablas de selector

| Archivo | MD5 |
|---|---|
| `holidays/holiday_selector_features.csv` | `c79438df1813093df5239675e1992c26` |
| `holidays/holiday_selector_priors.csv` | `d741726681124cd2b82f192e4f5c4068` |

---

## 2. Panel de evaluación (152 celdas)

**8 series × 19 fechas objetivo = 152 celdas.** Entrenamiento *rolling*: para cada fecha objetivo
el corte de entrenamiento es esa misma fecha (`train_end = target_date`), de modo que nunca se
usa información posterior.

```python
FULL_TARGETS = [
    ("2025-01-01", "New Year's Day"),        ("2025-02-03", "Constitution Day"),
    ("2025-03-17", "Benito Juarez's Birthday"), ("2025-04-17", "Maundy Thursday"),
    ("2025-04-18", "Good Friday"),           ("2025-04-19", "Holy Saturday"),
    ("2025-05-01", "Labor Day"),             ("2025-09-16", "Independence Day"),
    ("2025-11-17", "Mexican Revolution Day"),("2025-12-24", "Christmas Eve"),
    ("2025-12-25", "Christmas Day"),         ("2025-12-31", "New Year's Eve"),
    ("2026-01-01", "New Year's Day"),        ("2026-02-02", "Constitution Day"),
    ("2026-03-16", "Benito Juarez's Birthday"), ("2026-04-02", "Maundy Thursday"),
    ("2026-04-03", "Good Friday"),           ("2026-04-04", "Holy Saturday"),
    ("2026-05-01", "Labor Day"),
]
```

---

## 3. Configuración exacta del campeón

Corrida de referencia: `experiments/experiment_2026_08_25_07_17_criterion_holiday_identity`

### 3.1 Ventana

| Parámetro | Valor | Significado |
|---|---|---|
| `season_length` | **38** | horas pronosticadas por bloque |
| `forecast_start_offset_hours` | **14** | el pronóstico arranca 14 h antes de medianoche, es decir a las **10:00 del día previo** |
| Horas puntuadas | **24** | 00:00–23:00 del día festivo (`mape_24_pct`) |
| `special_labels` | `("holiday",)` | etiquetas que marcan día especial |

La ventana de 38 h = 14 h de "cabeza" (tarde-noche del día previo) + 24 h del festivo.

### 3.2 Criterio de cluster — **`holiday_identity`**

Cada festivo distinto (por `anchor_holiday_name`) es su propio cluster: el selector sólo
empareja un objetivo contra **otras instancias del mismo festivo** (filtro duro Similar-Days).

- 12 clusters, pool mediano **6** candidatos por serie, k medio resultante **2.34**
- `use_cluster = True`, `match_target_cluster = True`, `cluster_column = "analog_cluster"`

**Generación del CSV de selector** (ver §5.1 para el script):

```python
assign_holiday_selector_analog_clusters(
    df_selector=sel,            # holiday_selector_features.csv sin columnas analog_*
    df_priors=pri,              # holiday_selector_priors.csv
    criterion="holiday_identity",
    group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
    cluster_labels=("F", "G", "H"),   # se auto-extiende de F..Z según haga falta
)
```

### 3.3 Selección de análogos

| Parámetro | Valor |
|---|---|
| `typedist` inicial | `pearson` |
| `k` inicial | `None` (lo decide Optuna) |
| `min_special_points` | **24** |
| `min_event_gap` | **24** ← *debe ser 24; ver arreglo #4 en §6* |
| `max_events` | `None` |
| `recent_weekend_analogs` | **0** |
| `phase_period` | **24** (lo pasa la capa de festivos automáticamente) |
| `seam_positions` | calculado de las fechas (automático) |

### 3.4 Regresión

| Parámetro | Valor |
|---|---|
| `typereg` inicial | `PCR` |
| `n_components` inicial | 2 |
| `scale_method` inicial | `None` |
| `regressor_params` inicial | `{}` |
| `levels` (intervalos) | `[50, 80, 95]` |

### 3.5 Optuna

| Parámetro | Valor |
|---|---|
| `n_trials` | **25** |
| `timeout_sec` | **300** |
| `random_seed` | **42** |
| `max_eval_dates` | **19** |
| `optuna_min_k` | **2** |
| `optuna_min_k_by_cluster` | `{"F": 2, "G": 2, "H": 2}` |
| `optuna_max_k_by_cluster` | **`{}` (vacío)** |
| `typedist_choices` | `["pearson", "euclidian"]` |
| `typereg_choices` | `["PCR", "PLS", "RidgeReg", "LassoReg"]` |
| `scale_method_choices` | `[None, "standard", "minmax"]` |

> **Importante — el cap de k va vacío.** El cap `{"H": 6}` de producción se calibró para las
> letras F/G/H de `observance_tier`. Bajo `holiday_identity` la letra `H` es **otro festivo
> distinto**, así que aplicarlo compara manzanas con peras. Con `holiday_identity` el cap es
> además innecesario: el pool de ~2–3 candidatos limita k por sí solo.

### 3.6 Regresores elegidos por Optuna (resultado, no entrada)

`RidgeReg` 74 · `LassoReg` 71 · `PLS` 6 · `PCR` 1

---

## 4. Entorno

| Ítem | Valor |
|---|---|
| Python | 3.10 (`/usr/bin/python3`) |
| Backend matplotlib | **`MPLBACKEND=Agg`** ← obligatorio; sin esto el proceso aborta con Qt |
| Directorio de trabajo | `/home/uriel/GIT` (el paquete se importa como `analog_holidays`) |
| Suite de pruebas | 41/41 en verde |

```bash
cd /home/uriel/GIT && MPLBACKEND=Agg python3 -m pytest analog_holidays/tests/ -q
```

---

## 5. Cómo reproducirlo, paso a paso

### 5.1 Generar el CSV de selector con el criterio

`shared/regen_analog_clusters.py` reescribe el CSV compartido **en sitio**, lo que impide correr
variantes en paralelo. Usa este generador, que escribe un archivo por criterio:

```python
# gen_selectors.py
import sys; sys.path.insert(0, "/home/uriel/GIT")
import pandas as pd
from pathlib import Path
from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

HOL = Path("/home/uriel/GIT/analog_holidays/holidays")
OUT = Path("/home/uriel/GIT/analog_holidays/experiments/selectors"); OUT.mkdir(parents=True, exist_ok=True)
DROP = ["analog_cluster", "analog_cluster_criterion", "analog_criterion", "analog_criterion_value"]

sel0 = pd.read_csv(HOL / "holiday_selector_features.csv", parse_dates=["date"])
pri  = pd.read_csv(HOL / "holiday_selector_priors.csv")

criterion = "holiday_identity"
res = assign_holiday_selector_analog_clusters(
    df_selector=sel0.drop(columns=DROP, errors="ignore"), df_priors=pri,
    criterion=criterion, group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
    cluster_labels=("F", "G", "H"))
df = res["df_selector_clusters"]
prior_col = res.get("analog_criterion_prior_col")
drop = ["analog_criterion", "analog_criterion_value"] + ([prior_col] if prior_col else [])
(df.drop(columns=drop, errors="ignore")
   .assign(analog_cluster_criterion=criterion)
   .sort_values(["unique_id", "date", "holiday_name"])
   .reset_index(drop=True)
   .to_csv(OUT / f"selector_{criterion}.csv", index=False))
```

**Verificación esperada:** 12 clusters, 0 NaN sobre 648 filas, pool por serie min 5 / med 6 / max 12.

### 5.2 Correr el panel

El runner (`experiments/run_deep_regression_sweep.py`) toma la ruta del selector de una global de
módulo, así que hay que reapuntarla **y reconstruir los lookups** antes de llamar a `run_variant`:

```python
# run_criterion.py  —  uso: python3 run_criterion.py holiday_identity
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/uriel/GIT/analog_holidays/experiments")
sys.path.insert(0, "/home/uriel/GIT")
import run_deep_regression_sweep as base

path = Path("/home/uriel/GIT/analog_holidays/experiments/selectors/selector_holiday_identity.csv")
base.SELECTOR_FEATURES_PATH = path
df = pd.read_csv(path, parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df["unique_id"] = df["unique_id"].astype(str)
base.selector_features_df = df
col = base.CLUSTER_COLUMN
base.selector_cluster_lookup_by_id = {
    uid: (df.loc[df.unique_id == uid].dropna(subset=[col])
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")[col].to_dict())
    for uid in base.series_unique_ids}
base.selector_anchor_lookup_by_id = {
    uid: (df.loc[df.unique_id == uid].dropna(subset=["anchor_holiday_name"])
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["anchor_holiday_name"].to_dict())
    for uid in base.series_unique_ids}

base.OPTUNA_MAX_K_BY_CLUSTER = {}          # ← vacío, ver §3.5
name, med, mean, picks = base.run_variant(
    ["PCR", "PLS", "RidgeReg", "LassoReg"], "criterion_holiday_identity",
    target_items=FULL_TARGETS, min_event_gap=24)
```

```bash
cd /home/uriel/GIT && MPLBACKEND=Agg python3 run_criterion.py holiday_identity
```

**Costo:** ~5.3 s por celda de tuning, ~17 min para las 152 celdas en un solo proceso
(8 núcleos, ~20 min si corres varias variantes en paralelo).

**Salida:** `experiments/experiment_<timestamp>_criterion_holiday_identity/` con `metrics.csv`,
`summary.csv`, `manifest.yaml` y `notes.md`.

---

## 6. Los siete arreglos de código — **imprescindibles**

Sin estos arreglos el resultado **no** es reproducible, porque el buscador de hiperparámetros
estaba explorando un espacio distinto del que decía explorar. (El paper los agrupa en seis en su
§IV-D, fusionando los defectos 2 y 3, que son dos caras del mismo problema de penalización.)

| # | Archivo | Defecto | Arreglo |
|---|---|---|---|
| 1 | `analog/analog.py` | **PLS ignoraba `n_components`**: el `min(...)` incluía el número de columnas de `Y`, que siendo 1-D valía 1, fijando `n_components=1` siempre. Medido: `max\|pred(nc)−pred(nc=1)\| = 0.000e+00` para nc ∈ {1,2,3,5,8}. | Acotar sólo por el bloque de predictores. |
| 2 | `analog/analog.py` | **Ridge/Lasso eran OLS disfrazado**: `alpha=0.1` fijo sobre MW crudos (~5×10⁴) no penaliza nada. Diferencia contra `LinearReg`: 3.6e-06 MW. | Estandarizar los predictores dentro del estimador (`make_pipeline(StandardScaler(), Ridge(...))`) y exponer `alpha` a Optuna en `[1e-3, 1e3]` log. |
| 3 | `analog/analog_holidays.py` | `alpha` nunca entraba a la búsqueda (`_suggest_optuna_regressor_params` devolvía `{}` para Ridge/Lasso). | Añadido `_PENALIZED_LINEAR_REGRESSORS` y la sugerencia de `alpha`. |
| 4 | `analog/analog_holidays.py` | **`scale_method` se leía de donde nunca aparece**: se recuperaba de `study.best_params`, pero un valor ya resuelto no se registra ahí, así que la corrida final podía usar un escalado distinto del que ganó la búsqueda. | Guardarlo como `trial.set_user_attr("scale_method", ...)` y leerlo de `best_trial.user_attrs`, con `best_params` como respaldo. |
| 5 | `analog/analog_holidays.py` | **Rango de `k` vacío en silencio**: un `optuna_max_k < optuna_min_k` suministrado por el usuario producía una búsqueda vacía sin avisar. | `ValueError` explícito. |
| 6 | `analog/analog_special_days.py` | **Sin alineación de fase**: `_select_special_positions` no tenía lógica de hora-del-día. Con `min_event_gap=12` la mitad del pool quedaba 12 h desfasada (mapeando las 22:00 sobre las 10:00). Con `gap=24` funcionaba **por coincidencia aritmética**. | Parámetro explícito `phase_period` (la capa de festivos pasa 24). Verificado: con gap=24 el resultado es **idéntico** (258 = 258 posiciones); con gap=12 pasa de 396 (138 mal alineadas) a 258 correctas. |
| 7 | `analog/analog_holidays.py` + `analog_special_days.py` | **Ventanas que cruzan costura**: al aplanar los días, una ventana de 38 h podía empalmar los dos lados de una exclusión COVID, emparejando el contexto pre-festivo de un año con el festivo de otro. Medido en CEL: 4 de 97 ventanas, **3 de ellas en Año Nuevo** (contexto 2021-12-30 → festivo 2023-01-01). | `_calendar_seam_positions()` + parámetro `seam_positions`, que descarta candidatos cuyo span `[pos, pos+2·vsele)` cruza una costura. Elimina exactamente esas 4 y conserva las otras 93. |

**Efecto neto de los arreglos sobre el error: ninguno significativo** (3.739 % → 3.805 %,
p = 0.37). Pero cambiaron radicalmente qué modelo elige la búsqueda:

| regresor | antes | después |
|---|---|---|
| PLS | 10 | **51** |
| RidgeReg | 94 | 68 |
| LassoReg | 35 | 16 |
| PCR | 13 | 17 |

Esto es en sí un resultado: se puede cambiar sustancialmente el motor de regresión y el error no
se mueve, porque el error **no es de modelo, es de nivel** (§7).

---

## 7. Comparación contra el naive estacional

### 7.1 Definición de las líneas base

Para cada (región, festivo objetivo), se toman las instancias **previas** del mismo festivo y se
combinan sus perfiles de 24 h. Dos reglas de emparejamiento, porque difieren en los festivos
movibles:

- **`by_date`** — mismo día del calendario (mes-día) en años anteriores. Es la lectura literal de
  "la misma fecha de años pasados". Para Semana Santa empareja fechas que **no** son el festivo.
- **`by_holiday`** — mismo festivo por nombre en años anteriores. Sigue a la Pascua cuando se mueve.
  **Es la comparación justa**, y es la más exigente.

Combinadores: `mean` y `median` sobre todas las instancias previas, y sobre las últimas 2, 3 y 4.
Se puntúan exactamente igual que el modelo: MAPE sobre las 24 h del festivo.

Instancias previas disponibles: `by_date` mediana 4 (min 3, max 5) · `by_holiday` mediana 5 (min 4, max 10).

### 7.2 Resultados pareados (152 celdas)

| Línea base | mediana MAPE_24 | modelo gana | delta | **skill** | p |
|---|---|---|---|---|---|
| **`by_holiday` media de 2** | **5.570 %** | 63 % | −1.271 pp | **+0.320** | 0.00001 |
| `by_holiday` mediana de 2 | 5.570 % | 63 % | −1.271 pp | +0.320 | 0.00001 |
| `by_holiday` mediana (todas) | 6.078 % | 73 % | −2.531 pp | +0.377 | <1e-5 |
| `by_holiday` último año | 6.279 % | 70 % | −1.660 pp | +0.397 | <1e-5 |
| `by_holiday` mediana de 4 | 6.277 % | 76 % | −2.468 pp | +0.396 | <1e-5 |
| `by_holiday` media de 3 | 6.480 % | 71 % | −2.289 pp | +0.415 | <1e-5 |
| `by_holiday` media de 4 | 6.661 % | 74 % | −2.681 pp | +0.431 | <1e-5 |
| `by_holiday` media (todas) | 7.298 % | 74 % | −3.231 pp | +0.481 | <1e-5 |
| `by_date` media (todas) | 7.177 % | 70 % | −2.677 pp | +0.472 | <1e-5 |
| `by_date` último año | 8.014 % | 74 % | −3.009 pp | +0.527 | <1e-5 |
| `by_date` mediana (todas) | 8.640 % | 76 % | −4.087 pp | +0.561 | <1e-5 |
| `by_date` mediana de 3 | 10.068 % | 74 % | −5.054 pp | +0.624 | <1e-5 |

`skill = 1 − MAPE_modelo / MAPE_naive` sobre medianas. Prueba: Wilcoxon de rangos con signo pareado.

**El modelo le gana a todas las variantes, todas significativas.** El naive más difícil de batir es
promediar (o medianar — dan lo mismo con n=2) **las 2 instancias más recientes del mismo festivo**:
5.570 %, skill **+0.320**.

**Observación relevante:** promediar *más* años empeora el naive (2 instancias → 5.57 %, 4 → 6.66 %,
todas → 7.30 %). Es la misma dilución que afecta al método analógico: meter años lejanos arrastra
crecimiento de carga y observancias distintas.

### 7.3 Skill por región

| Región | modelo | naive (`by_holiday` mediana) | skill |
|---|---|---|---|
| SIN | 2.406 % | 6.284 % | **+0.617** |
| NOR | 4.097 % | 10.332 % | +0.603 |
| PEN | 8.541 % | 15.648 % | +0.454 |
| ORI | 3.507 % | 6.223 % | +0.436 |
| OCC | 3.199 % | 5.553 % | +0.424 |
| NES | 6.696 % | 10.700 % | +0.374 |
| CEL | 2.224 % | 3.452 % | +0.356 |
| NTE | 4.404 % | 5.971 % | +0.262 |

**Skill positivo en las 8 regiones.** PEN, que tiene el peor MAPE absoluto (8.54 %), tiene skill
+0.454: es una región genuinamente volátil donde el método aporta mucho, no una donde falle.

### 7.4 Skill por festivo

| Festivo | modelo | naive | skill |
|---|---|---|---|
| Labor Day | 4.087 % | 13.148 % | **+0.689** |
| Holy Saturday | 3.512 % | 10.825 % | +0.676 |
| Mexican Revolution Day | 2.323 % | 7.054 % | +0.671 |
| Good Friday | 3.700 % | 10.185 % | +0.637 |
| Maundy Thursday | 2.926 % | 6.522 % | +0.551 |
| Christmas Eve | 4.043 % | 8.611 % | +0.530 |
| Christmas Day | 4.011 % | 7.032 % | +0.430 |
| Constitution Day | 2.670 % | 4.242 % | +0.371 |
| Benito Juarez's Birthday | 3.471 % | 4.908 % | +0.293 |
| New Year's Eve | 4.022 % | 4.969 % | +0.190 |
| **New Year's Day** | 5.409 % | 4.960 % | **−0.091** |
| **Independence Day** | 4.550 % | 3.305 % | **−0.377** |

**En 10 de 12 festivos el modelo gana.** Pierde en Año Nuevo e Independencia — los dos con víspera
de celebración nocturna intensa, donde el contexto pre-festivo es él mismo anómalo.

Script: `seasonal_naive_ceiling.py`; salida celda a celda en `seasonal_naive_ceiling.csv`.

---

## 8. Por qué esto es el techo (y qué queda fuera)

### 8.1 El error es de nivel, no de forma

Sobre 3 446 filas de 26 experimentos: la razón `|MPE| / MAPE` tiene **mediana 0.963**, y el
**40.2 %** de las filas tiene razón exactamente 1.000 (las 24 horas comparten un solo signo).
Ponderado, **89.3 %** del error es corrimiento vertical puro.

En el campeón: MAPE 3.789 % = **nivel 3.014 %** + forma 0.775 pp.

### 8.2 Todos los ejes internos están agotados

| Eje | Evidencia |
|---|---|
| Regresor / alpha / n_components | Los 5 arreglos cambiaron radicalmente la selección de modelo y el error no se movió (p = 0.37). |
| Configuración global | Los 26 experimentos explican **0.4 %** de la varianza del sesgo y 4.4 % del MAPE; la celda (región, fecha) explica **78–80 %**. |
| k | Explica ~0.2 % de la varianza intra-celda. `selected_analogs == k` en el 100 % de las filas (nunca hubo escasez). |
| Criterio de cluster | **El mejor lever interno**: mueve el nivel de 5.16 % (`heat_cold`) a 3.01 % (`holiday_identity`), −42 %. Pero ~53 % de eso es sólo dilución de k, no semántica. |

### 8.3 El nivel es irreducible desde la demanda — probado por construcción

Se implementaron dos correcciones de nivel que usan **exclusivamente** información disponible al
momento de emisión (`_ex_ante_level_factors`):

- `exante_level` = nivel observado reciente ÷ nivel pre-festivo de los análogos → **+2.398 pp peor** (p < 1e-5, gana 20 %)
- `exante_drop` = nivel observado × caída propia de los análogos (estimando de razón) → **+0.551 pp peor** (p = 0.00007, gana 35 %)

**Ambas empeoran significativamente.** La causa es **doble conteo**: el intercepto de la regresión
ya ancla el pronóstico al nivel actual (87 % del nivel del pronóstico es persistencia pre-festivo),
así que multiplicar por una razón nivel-actual/nivel-análogos aplica el crecimiento de carga dos
veces. Evidencia: el factor ex-ante es > 1 en el **81 %** de las celdas (mediana 1.0333) mientras el
factor válido lo es en sólo 48 % (mediana 0.9990); tras aplicarlo, el 76 % de las celdas quedan
sobre-pronosticadas.

**Conclusión: el método ya extrae toda la señal de nivel que contiene la historia de demanda.**

### 8.4 La cota superior y por qué no es operativa

El ancla de nivel sobre la cabeza observada (`mape_24_head_ratio_pct`) da **3.201 %** (−15.5 %,
p = 0.00001, gana 65 %), con el nivel bajando de 3.014 % a 2.024 %.

**No es usable.** El pronóstico se emite en la mañana del día previo y alimenta fijación de precios,
planeación de generación y despacho; esas decisiones quedan comprometidas y recalcular después no
tiene sentido. El ancla mide el error del modelo en la tarde-noche del día previo, que **no ha
ocurrido** al emitir.

Sirve como **cota superior**: cuantifica en −15.5 % lo que valdría tener información perfecta de
nivel al emitir. Ese es el premio que justifica meter una variable exógena.

Nota: dos ideas de compuerta fueron probadas y **refutadas** — por magnitud de `|mpe_14|` (nunca
mejora) y por "la víspera también es festivo" (activamente peor: 3.517 % vs 3.201 %, p = 0.0015;
la premisa está al revés, cuando la víspera es festivo el ancla funciona *mejor*).

### 8.5 Lo único que queda: variables exógenas — **ya no es conjetura, está medido**

Cuando se escribió este documento no había ninguna variable meteorológica en el pipeline, y la
atribución al clima descansaba en evidencia circunstancial:

- La correlación cruzada entre regiones **el mismo día** es 0.455 en ERCOT y 0.204 en SEN; el
  componente común-por-fecha explica **58.1 %** de la varianza del sesgo en ERCOT.
- Firma concreta: MLK 2025-01-20 (helada en Texas) se sub-pronosticó **+25 a +38 pp
  simultáneamente en 4 zonas ERCOT**.
- Los 22 festivos cambian de signo de sesgo entre regiones, así que el calendario solo no puede
  dar el nivel.

**El 2026-09-02 se obtuvo la temperatura y la conjetura quedó confirmada.** Las ocho zonas de
demanda de ERCOT *son* sus zonas climáticas oficiales, así que cada una admite una temperatura
representativa sin repartir carga. Se bajó temperatura horaria de Open-Meteo (gratis, sin API key)
para 18 puntos urbanos ponderados por población, agregados a las 8 zonas más el sistema.

El predictor no es la temperatura cruda sino la **anomalía de grados-día**: media diaria de
|T − 18.3 °C| menos su climatología del día del año suavizada ±7 días. La temperatura cruda es mucho
más débil (R² 5.7 %) porque la demanda sube en ambas direcciones desde el punto de balance y los dos
sentidos se cancelan.

| Fuente de temperatura | ¿Admisible al emitir? | R² por celda (n=171) | Spearman R² | p | choque sistémico (n=19) |
|---|---|---|---|---|---|
| ERA5 observado | ❌ oráculo | 24.5 % | 18.4 % | 4.7e-9 | 42.3 % |
| **Pronóstico D−1** | **✅ sí** | **21.7 %** | **16.5 %** | **3.4e-8** | 34.8 % |
| Pronóstico D−2 | ✅ sí | 19.9 % | 16.7 % | 2.8e-8 | 33.0 % |

Tres lecturas:

1. **El clima sí es la causa.** Una sola variable exógena explica 18–25 % de la varianza del sesgo,
   contra **0.4 %** de los 26 experimentos de método completos (§8.2). Es ~50× más explicativa que
   todo lo que se probó por dentro.
2. **La señal sobrevive a la restricción operativa**, que es lo que la distingue del ancla de nivel
   de §8.4. A D−1 el pronóstico de temperatura tiene MAE de 0.72 grados-día y r = 0.979 contra lo
   observado, y se conserva el **89 %** del poder explicativo. Esta corrección **sí** es usable.
3. **Los dos peores errores del panel son eventos climáticos opuestos**: la helada de enero 2025
   (anomalía +9.7, sub-pronóstico +16.5 %) y un Año Nuevo 2026 templado (sobre-pronóstico −18.6 %).

⚠️ **Dos honestidades.** El 42.3 % del choque sistémico se apoya en n=19 fechas y en la helada de
MLK; excluyéndola cae a 12.1 % y pierde significancia (p = 0.11). **La cifra defendible es la de por
celda** (n=171, rango, p < 1e-7). Y se probó si lo que importa es el *cambio* de temperatura entre la
ventana observada y el festivo —lo que el método estructuralmente no puede saber— y **no lo es**: ese
predictor explica sólo 1–3 %. Importa el nivel de la anomalía, no su cambio.

**Qué falta.** Este análisis es de ERCOT, donde la zonificación es climática. No se ha obtenido la
serie meteorológica equivalente para las regiones de control mexicanas, así que la conjetura de que
el hueco de NES/NTE/PEN es también clima **no está probada para México**. Y medir la señal no es
explotarla: el siguiente paso es meter la anomalía como corrección de nivel y medir cuánto baja
el 3.789 %.

---

## 9. Archivos de referencia

| Qué | Dónde |
|---|---|
| Corrida campeona | `experiments/experiment_2026_08_25_00_37_criterion_holiday_identity/` (la misma que cita el paper; las re-corridas `06_40` y `07_17` son bit-idénticas: MAPE y regresor coinciden celda a celda) |
| Métricas celda a celda | `.../metrics.csv` (152 filas) |
| Config completa | `.../manifest.yaml` |
| Naive estacional pareado, celda a celda | `docs/seasonal_naive_ceiling.csv` (152 filas) |
| Scripts de reproducción | `experiments/{gen_selectors,run_criterion,seasonal_naive_ceiling}.py` |
| Barrido de los 7 criterios | `experiments/experiment_2026_08_25_00_3*_criterion_*/` |
| Ablación del confundido k | `experiments/experiment_2026_08_25_00_59_criterion_*_kcap*/` |
| Benchmark Similar-Days | `docs/BENCHMARK_SIMILAR_DAYS.md`, `docs/similar_days_benchmark{,_ercot}.csv` |
| Temperatura observada (ERA5) | `holidays/weather_ercot.csv` — 89 856 h × 9 zonas, 2016–2026 |
| Temperatura pronosticada D−1/D−2 | `holidays/weather_forecast_ercot.csv` — formato largo, 210 384 filas |
| Descarga de clima | `experiments/fetch_weather_ercot.py`, `experiments/fetch_weather_forecast_ercot.py` |
| Análisis de clima (§8.5) | `experiments/analyze_weather_bias.py`, `experiments/analyze_weather_exante.py` |
| Clima celda a celda | `docs/weather_bias_cells.csv`, `docs/weather_exante_cells.csv` |

### Columnas de métricas relevantes en `metrics.csv`

| Columna | Significado |
|---|---|
| `mape_24_pct` | **métrica principal** — MAPE sobre las 24 h del festivo |
| `mpe_24_pct` | error porcentual con signo; `(actual − pronóstico)/actual`. **Negativo = sobre-pronóstico** |
| `bias_24` | sesgo medio en MW |
| `mape_38_pct` | MAPE sobre la ventana completa de 38 h |
| `mape_24_bias_adjusted_pct` | corrección de factor horario (**de media cero por construcción**; no mueve el nivel) |
| `mape_24_head_ratio_pct` | ancla de nivel — **cota superior, no operativa** |
| `mape_24_exante_level_pct`, `mape_24_exante_drop_pct` | correcciones ex-ante — ambas empeoran |
| `k`, `selected_analogs` | análogos pedidos / realizados (iguales en el 100 % de los casos) |
| `typereg`, `scale_method`, `n_components`, `regressor_params` | config ganadora de Optuna en esa celda |

---

## 10. Reproducción en una sola pasada

Los tres scripts viven en `analog_holidays/experiments/` y ya están apuntados a rutas del repo.
Ejecutar **siempre desde `/home/uriel/GIT`** (el paquete se importa como `analog_holidays`) y
**siempre con `MPLBACKEND=Agg`**.

```bash
cd /home/uriel/GIT

# 0) verificar entorno — los cinco arreglos de §6 deben estar aplicados
MPLBACKEND=Agg python3 -m pytest analog_holidays/tests/ -q        # esperado: 41 passed

# 1) generar el selector con el criterio campeón
#    -> analog_holidays/experiments/selectors/selector_holiday_identity.csv
#    esperado: clusters=12  NaN=0/648  pool/serie min=5 med=6 max=12
MPLBACKEND=Agg python3 analog_holidays/experiments/gen_selectors.py

# 2) correr el panel de 152 celdas (~17 min)
#    -> analog_holidays/experiments/experiment_<timestamp>_criterion_holiday_identity/
MPLBACKEND=Agg python3 analog_holidays/experiments/run_criterion.py holiday_identity

# 3) comparar contra el naive estacional (usa la corrida más reciente automáticamente)
MPLBACKEND=Agg python3 analog_holidays/experiments/seasonal_naive_ceiling.py
```

Scripts auxiliares, en el mismo directorio:

| Script | Para qué |
|---|---|
| `compare_criteria.py` | tabla comparativa de los 7 criterios sobre el panel |
| `analyze_anchor.py` | efecto del ancla de nivel (cota superior) |
| `analyze_exante.py` | cuánto de esa ganancia sobrevive con información de emisión |

`run_criterion.py` acepta un segundo argumento opcional con un cap uniforme de k
(`python3 run_criterion.py observance_tier 3`), que es como se corrió la ablación de §8.2.

**Valores esperados** (semilla 42, mismos datos):

| Métrica | Valor |
|---|---|
| mediana `mape_24_pct` | **3.789 %** |
| media `mape_24_pct` | 5.536 % |
| mediana `\|mpe_24_pct\|` | 3.014 % |
| celdas fallidas | **0** de 152 |
| naive más fuerte (`by_holiday` media de 2) | 5.570 % |
| skill contra ese naive | **+0.320** |

Si la mediana no da 3.789 %, revisa en este orden: (1) que la suite dé 41/41 —los cinco arreglos de
§6 tienen que estar aplicados—, (2) que `OPTUNA_MAX_K_BY_CLUSTER` esté **vacío**, (3) que el MD5 del
CSV de demanda coincida, (4) que `random_seed = 42` y `n_trials = 25`.
