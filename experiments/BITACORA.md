# Bitácora de experimentos — clustering de festivos por observancia

Registro narrativo que conecta la serie de experimentos de junio 2026 sobre cómo agrupar
los días análogos antes de pronosticar festivos. Complementa los `notes.md` de cada corrida
(detalle por experimento) y el `README.md` (convención). Lee de arriba hacia abajo: cada
bloque es una hipótesis, su corrida y lo que aprendimos.

Métrica primaria: **mediana de `mape_24_pct`** (24 h del festivo), 8 regiones SEN, cutoff
rolling. "shape" = criterio `shape_pearson_CDE_map_FGH`; "observance" = `observance_tier`.

## Resumen de corridas

| Experimento | Qué cambió | Targets | Mediana mape_24 | Veredicto |
|---|---|---|---|---|
| `2026_06_13_13_27` | baseline shape, k=10 Ridge | 2025-26 (19) | **4.50%** | baseline de referencia |
| `2026_06_14_21_46` | **observance_tier (3-tier)** | 2025-26 (19) | **3.76%** | ✅ **ganador, robusto** |
| `2026_06_15_08_36` | observance_tier_depth (4-tier) + MIN_K=4 | 2025-26 (19) | 3.77% | empate (dos ejes confundidos) |
| `2026_06_15_08_57` | 4-tier, MIN_K=2 (aislamiento) | 2025-26 (19) | 3.80% | ❌ el split no aporta |
| `2026_06_15_10_04` | 3-tier + MIN_K por tier {F2,G2,H4} | 2025-26 (19) | 4.00% | ❌ k-por-tier no aporta |
| `2026_06_15_11_12` | 3-tier, set ampliado | 2023-26 (42) | 4.07% (limpio 3.86%) | ✅ valida; contaminación en 2023 |
| `2026_06_15_11_51` | 3-tier, historia recortada a 2023+ | 2025-26 (19) | 4.41% | ✅ más historia gana |

## Hallazgos (en orden de descubrimiento)

### 1. Diagnóstico: hay DOS modos de error (no uno)
`shared/observed_strength.py` puntúa, por festivo/región, la caída real de demanda contra la
caída típica. Resultado:
- **Modo A — fallo de profundidad:** festivos plenos profundos (Navidad, Año Nuevo,
  Independencia) con MAPE alto porque acertar la *magnitud/forma* de una caída genuina es
  difícil. No es problema de observancia. Sobre-predice. Driver de las peores MAPE.
- **Modo B — sobre-tratar un festivo blando:** días poco observados (Labor Day fuerza ~0.47,
  Revolución, Sábado Santo, Jueves Santo) que el modelo trata como festivo pleno pero la
  realidad es casi laboral → **sub-predicción sistemática (+bias ~1.7%)**. Esto confirma la
  intuición del "día sustituido / puente".

### 2. `observance_tier` (3 niveles) — la ganancia robusta
Criterio nuevo en `ANALOG_CLUSTER_CRITERIA_CATALOG` (`shared/identify_holidays.py`): mapa
estático anchor→tier (working/partial/full → clusters F/G/H), umbrales <0.55 / 0.55-0.80 /
≥0.80 sobre la fuerza de observancia mediana. Ataca el Modo B agrupando los análogos por qué
tan inhábil es el día.
- **4.50% → 3.76%** (vs baseline shape). H1 −2.02 pp. El +bias de festivos blandos
  desaparece (cluster G partial: +1.73% → +0.42%).
- Validado en 42 fechas: se sostiene en **3.86%** sobre 35 fechas limpias post-pandemia —
  no era ruido de 19 fechas.

### 3. Callejones sin salida (descartados con disciplina)
- **4-tier `observance_tier_depth`** (parte `full` en cívico ~15% caída vs profundo ~30%):
  aislado, empata/pierde marginalmente (3.80% vs 3.76%, 10-9 por fecha = ruido). **No aporta.**
- **MIN_K por tier** {F2,G2,H4} sobre 3-tier: 4.00%, **peor**. Razón: el tier `H=full`
  mezcla cívicos (quieren k≥4) y profundos (mejores con k=2); un solo piso no sirve a ambos.
- Lección: el split-por-profundidad y el k-por-tier son **complementarios, no alternativas** —
  ninguno ayuda solo. La única combinación motivada sin probar: 4-tier + {cívico:4, profundo:2}.
- **Perspectiva:** todas las variantes de observancia caen en 3.76–4.00%, ~0.5–0.7 pp mejor
  que shape (4.50%); las diferencias entre ellas son **piso de ruido** sobre n=19.

### 4. Contaminación de pandemia — real pero localizada
Datos: 2020, 2021, 2023–2027 — **2022 falta por completo**. Por el hueco, un target de 2023
solo tiene análogos de 2020–2021 (COVID). Eso hace a 2023 el peor año (mediana 4.99% vs
3.5–3.9 limpios); caso extremo **Navidad 2023 = 14.11%**. A partir de 2024 se diluye.

### 5. Más historia mejora (incluso con pandemia)
Verificado: los targets 2025–2026 **ya usan toda la historia** (Navidad 2025 jala 6/10
análogos de 2020–2021). Recortar a 2023+ (sin pandemia) **empeora**: 4.41% vs 3.76%, 13/19
fechas peor (Navidad 2025 4.33→8.80). Con solo 2 años limpios cada cluster tiene muy pocos
análogos; los años COVID, aun contaminados, son **net-positivos** para targets limpios. La
pandemia solo es tóxica cuando es la *única* historia (targets 2023). No hay datos antes de
2020-01-02.

### 6. El eje de REGRESIÓN está agotado para el Modo A (festivos profundos)
Barrido de regresión scopeado a las 6 fechas profundas (Modo A: Año Nuevo, Independencia,
Nochebuena, Navidad, Fin de Año) × 8 regiones, todo lo demás en producción (`observance_tier`,
MIN_K=2, **gap=24**, historia completa). Ancla = `control` (búsqueda adaptativa de producción
PCR/PLS/Ridge/Lasso). Driver reproducible: `experiments/run_deep_regression_sweep.py`
(+ `analyze_deep_regression_sweep.py`). Verificado fiel: control sobre Navidad/SIN = RidgeReg
k=6 mape_24 3.3787%, idéntico a `14_21_46`.

| Variante (deep, 48 filas) | Mediana mape_24 | Media | vs control |
|---|---|---|---|
| `control` (búsqueda adaptativa) | **3.605%** | 5.600% | ancla |
| RF_enriched (+RandomForest) | 3.467% | 5.512% | −0.14 pp (ruido; RF nunca elegido) |
| PLS forzado | 3.637% | 5.740% | +0.03 pp (empate) |
| PCR forzado | 4.343% | 5.694% | +0.74 pp |
| LassoReg forzado | 4.469% | 6.979% | +0.87 pp |
| RidgeReg forzado | 4.684% | 7.326% | +1.08 pp |

- **Forzar un solo regresor es igual-o-peor que la búsqueda adaptativa.** El mejor forzado (PLS)
  apenas empata; forzar Ridge/Lasso dispara la media (7.33 / 6.98) por celdas catastróficas
  (OCC Navidad 2.40→**25.19%**, SIN Navidad 3.38→**15.45%**, NES 3.06→10.63): restringir el
  espacio a un regresor desestabiliza el tuning conjunto de k/scale en regiones volátiles.
- **RF no aporta:** nunca fue elegido (k=2-8 análogos, muy pocos para árboles); en RF_enriched
  Optuna eligió Ridge 38 / PLS 10. La "mejora" (−0.14 pp, 13-11-24 win/loss) es jitter de Optuna.
- **Piso de ruido:** un cambio benigno en el seed `initial_k` mueve la mediana profunda 0.42 pp
  por sí solo (control 3.605 vs baseline logueado 4.028, misma gap/búsqueda). Todo dentro de
  ~0.4 pp es ruido.
- **Veredicto: el Modo A NO es un problema de elección de regresor.** Optuna ya converge a la
  mezcla correcta (~60% Ridge, 25% Lasso). El residual de festivos profundos vive en la SEÑAL
  análoga / magnitud, no en el método de regresión → atacarlo por el **pool de análogos** o
  **corrección de magnitud/sesgo scopeada a profundo**, no por el regresor. Búsqueda adaptativa
  se queda en producción sin cambios.

### 7. `MIN_EVENT_GAP` 24 vs 12 — más pool ≠ mejor; los profundos quieren POCOS análogos
Eje único sobre las 19 fechas, todo lo demás en producción. Bajar gap 24→12 solapa ventanas-candidato
cada 12h (~duplica el pool). Driver: `experiments/run_gap_experiment.py`. Hipótesis: más análogos
ayudan a las celdas profundas hambrientas. **Refutada — y al revés:**

| Segmento | gap=24 (prod) | gap=12 | Δ mediana |
|---|---|---|---|
| ALL (152) | **3.810%** | 3.987% | +0.18 pp ❌ |
| **deep (48)** | **3.605%** | 4.519% | **+0.92 pp** ❌❌ |
| soft (104) | 3.882% | 3.862% | −0.02 pp (empate) |

- El daño se concentra en los profundos: **Navidad +3.27 pp** (2.96→6.23), **Fin de Año +2.63 pp**,
  **Independencia +1.20 pp**. Los soft no se mueven (−0.02 pp).
- Mecanismo: gap=12 mete ventanas solapadas (copias desplazadas, no análogos independientes) → Optuna
  sube `k` (Navidad PEN k=6→11, Independencia SIN k=6→11) → **diluye la señal nítida de la caída
  profunda con análogos menos similares.** Los festivos profundos quieren POCOS análogos muy similares.
- Esto refuerza el hallazgo 6 por otra vía: el residual profundo es de **calidad/similitud de la
  señal análoga**, no de cantidad ni de regresor. La dirección correcta es *mejor pool* (recencia/
  calidad, profundidad), no *más pool*.
- **Veredicto: gap=24 gana. Revertido el `MIN_EVENT_GAP=12` sin commitear a 24** (deuda resuelta con
  evidencia; HEAD ya era 24). El exp. `11_51` (única corrida a 12) tiene su número en una config peor
  — su conclusión direccional ("más historia gana") probablemente se sostiene pero su mediana absoluta
  está inflada; re-verificar a 24 si se cita.

### 8. Techo de `k` en el cluster H — guarda de cola de bajo riesgo
A gap=24 los profundos siguen a `k`: mediana por bucket k≤3 → 3.99%, **k4-6 → 3.31% (el dulce)**,
**k7+ → 12.22%** (n=4). No es "usar k=2" sino "no pasar de ~6". Mecanismo nuevo
`OPTUNA_MAX_K_BY_CLUSTER` (espejo del MIN_K; param `optuna_max_k` en `tune_analog_holidays_optuna`).
Driver: `experiments/run_kcap_experiment.py`. No-op verificado (cap=None = control exacto).

| Variante | all med | all mean | deep med | deep mean | celdas H tocadas |
|---|---|---|---|---|---|
| control (sin cap) | 3.810 | 5.469 | 3.605 | 5.600 | — |
| **cap H=6** | 3.739 | 5.336 | 3.516 | 5.477 | 9/64 (k8→6), Δ −2.21 pp |
| cap H=4 | 3.643 | 5.221 | 3.593 | 5.209 | 29/64, Δ −1.13 pp (con regresiones) |

- **cap H=6** recorta solo las 9 celdas con k=8 → 6: −2.21 pp en ellas (deep −1.97, civic −2.33),
  wins grandes (un cívico 17.51→7.16). **Sin colateral**: F/G y el resto de H sin cambio; soft plano.
  La mejora del all-mean (−0.13 pp) es **toda** de clipear esas 9 catastróficas → reducción de cola,
  visible en la media, no en la mediana (3.810→3.739 = dentro del piso de ruido ~0.4 pp).
- **Los cívicos-en-H también sobre-seleccionan k y se benefician** → cluster H (observancia plena) es
  el scope correcto; **no hace falta el split por profundidad (4-tier).**
- **cap H=4** exprime más en agregado pero es riesgoso: forzar k=4 donde k5-6 era óptimo dispara
  regresiones (una celda deep 9.36→**17.95**). No vale la varianza.
- **Cierra el arco 6-7-8:** el residual profundo es de **calidad/similitud de la señal análoga**
  (no regresor, no cantidad). El cap a k≤6 lo ataja por la vía de no diluir la caída nítida.
- **Veredicto: `OPTUNA_MAX_K_BY_CLUSTER={'H':6}` es una guarda de cola gratis** (no puede dañar).
  Mediana dentro de ruido; ganancia real en media/cola. **ADOPTADO en producción** (notebook In[3]
  define el dict, In[7] pasa `optuna_max_k=_target_max_k` al tune; param `optuna_max_k` en el core).

### 9. Diagnóstico regional — el gap NES/NTE/PEN es ruido IRREDUCIBLE, no pool pobre
Las 3 peores regiones (NES 6.26%, PEN 6.22%, NTE 7.12% mediana mape_24) vs las 3 mejores
(CEL 2.10%, OCC 2.46%, SIN 3.02%) — spread 3×. Tres hipótesis, `experiments/diagnose_regional.py`
(sobre la corrida con plots `...production_cap_H6_plotted`):

| Hipótesis | Medición | Veredicto |
|---|---|---|
| H1 pool pobre/escaso | `selected_analogs` vs `k` | ❌ **RECHAZADA**: 0 celdas starved; pool idéntico (mismos festivos) |
| H2 demanda más ruidosa | rRMSE día-normal vs climatología (dow,hour) | ✅ corr **0.53**: NES/NTE/PEN 18-20% vs buenas 4-11% |
| H3 observancia inconsistente | std año-a-año del drop del festivo | ✅ corr **0.55**: NES/NTE/PEN 14-17% vs buenas 7-10% |

- **No es problema de método ni de pool.** Las pair-sequences lo muestran: en CEL los análogos
  (gris) están apretados y el target cae limpio entre ellos; en NES están **dispersos** (no concuerdan).
- H2 y H3 son la misma raíz: **estas regiones son intrínsecamente menos predecibles** — más ruido en
  días normales Y drops de festivo más variables año-a-año. El método análogo reduce el ruido pero no
  puede vencer la varianza irreducible. Buena parte del 6-7% es **piso de ruido, no gap arreglable.**
- Anomalía NOR: 29% ruido / 20.5 var-drop pero 3.5% MAPE → es **estacional** (CV 32%), no aleatorio;
  los proxies crudos lo sobreestiman. Para NES/NTE/PEN las dos señales independientes coinciden.
- Hipótesis de fondo (sin probar, fuera del pipeline análogo): NES (Monterrey, industrial), NTE,
  PEN (Yucatán, AC) son **sensibles al clima** → su variabilidad probablemente la dirige la
  temperatura, que el método análogo (sin covariables) no ve. Un covariate de clima sería el lever real.
- **Implicación:** no perseguir bajar el MAPE de estas regiones a niveles de CEL (es irreducible).
  Medir honesto **por región** y un agregado consciente del piso; lo accionable es intervalos más
  anchos / probabilístico y bias por región, no más tuning de punto. El headline (3.74%) está
  dominado por estas regiones difíciles.

### 10. Medición honesta (skill vs persistencia) — refina el hallazgo 9
`experiments/report_honest.py`: por región, MAPE del método vs baseline naive de **persistencia**
(mismo festivo, año previo), y skill = 1 − método/persistencia.

| Región | método_med | persist_med | skill | lectura |
|---|---|---|---|---|
| tratables (CEL/SIN/OCC/NOR/ORI) | **2.88** | 4.08 | 0.37 | método rinde parejo |
| PEN | 6.22 | **14.74** | **0.43** | genuinamente volátil; el método gana mucho |
| NES | 6.26 | 6.03 | 0.17 | marginal vs naive |
| **NTE** | 7.12 | **4.97** | **−0.50** | **el método es PEOR que naive** |

- **Headline reencuadrado (confirmado):** tratables = **2.88%** vs raw 3.74%. El número global lo
  inflan las 3 difíciles; el método en regiones normales es ~2.9%. Se ve mejor de lo que dice el 3.74%.
- **CORRIGE el hallazgo 9:** NO todo el gap es irreducible. La persistencia lo descompone:
  - **PEN = irreducible** (persistencia 14.74; festivos muy variables) y ahí el método ya hace su trabajo.
  - **NTE = fallo de método ARREGLABLE** (persistencia 4.97 < método 7.12; sus festivos SÍ son
    predecibles año-a-año, pero el análogo selecciona mal / sobre-regresiona y los empeora).
  - **NES = marginal** (método ≈ naive).
- El proxy de "drop_yoy_std" del hallazgo 9 sobreestimó la inconsistencia de NTE (mezclaba festivos
  distintos); persistencia mismo-festivo muestra que NTE es predecible. Caveat: n=19/región, señal direccional.
- **Lever concreto que surge:** donde el método pierde vs naive (NTE, y NES marginal), **piso/mezcla
  con persistencia** (fallback a último año cuando los análogos no concuerdan) o revisar la selección
  de análogos de NTE. Barato y dentro del pipeline.

## Configuración de producción (estado actual)
- **Criterio:** 3-tier `observance_tier` (F=working, G=partial, H=full) — en M_identify y en
  `holidays/holiday_selector_features.csv`.
- **Historia:** completa, `HISTORY_START = None` (2020–2024).
- **MIN_K:** 2 uniforme. (`OPTUNA_MIN_K_BY_CLUSTER` cableado pero neutralizado a {F2,G2,H2}.)
- **MAX_K:** `OPTUNA_MAX_K_BY_CLUSTER = {'H': 6}` (guarda de cola, hallazgo 8). F/G sin cap.
- **MIN_EVENT_GAP:** **24** (gap=12 probado peor, hallazgo 7). Búsqueda de regresor: adaptiva
  PCR/PLS/Ridge/Lasso (hallazgo 6: no forzar uno).
- **Targets:** 2025–2026 (19 fechas).
- **Headline honesto:** mediana mape_24 ≈ **3.76%** (≈3.86% en ventana limpia 2024+).

## Herramientas dejadas en el repo
- `shared/observed_strength.py` — diagnóstico de observancia (solo lectura).
- `shared/regen_analog_clusters.py` — re-sella el selector con cualquier criterio (con backup).
- Knobs en `P_..._cluster.ipynb`: `OPTUNA_MIN_K_BY_CLUSTER` (cell 4/8), `HISTORY_START` (cell 3).
- Criterios `observance_tier` y `observance_tier_depth` + tests en `tests/test_analog_package.py`.

## Hilos abiertos (no urgentes)
1. **Pesar análogos por recencia/calidad** en vez de cortar: conservar tamaño de pool pero
   descontar perfiles de confinamiento — captura lo mejor de ambos.
2. **Modo A (profundidad):** ~~explorar más componentes / regresor scoped al cluster profundo~~
   → **DESCARTADO** (hallazgo 6): el eje de regresión no aporta. El residual profundo es de
   señal/magnitud → probar **pesado del pool de análogos** o **corrección de magnitud/sesgo
   scopeada a profundo** (no el regresor). Atar con el hilo 1 (pesar por recencia/calidad).
3. **Ampliar el set de validación** cuando entren más años limpios, para que las medianas
   dejen de ser ruido de 19 fechas.
