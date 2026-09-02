# Similar Days vs. Analog Holidays — diferencia metodológica

**Alcance.** Documento de contraste entre la metodología clásica de *Similar Days* (días similares) y
el método **Analog** propuesto en este repo para el pronóstico de demanda en días festivos. Define la
distinción conceptual, la formaliza, y aporta el párrafo listo para paper. Para el detalle de cada
enfoque por separado ver [`similar_days_holidays.md`](similar_days_holidays.md) (base conceptual de
Similar Days) y [`champion_analog_holiday.md`](champion_analog_holiday.md) (implementación Analog
productiva).

---

## 1. La diferencia central

> **Similar Days** busca **días históricos** parecidos al día objetivo por **contexto**.
>
> El método **Analog** busca **ventanas históricas** parecidas por **forma temporal** y usa sus
> **subsecuentes $X'$** para pronosticar el futuro.

En una frase: Similar Days compara el *día objetivo* con otros días usando **metadatos**; Analog
compara la *historia reciente de la serie* con **patrones históricos similares** y usa lo que ocurrió
después de cada patrón como trayectoria análoga.

---

## 2. Comparación lado a lado

| Aspecto | Similar Days | Método Analog (propuesto) |
|---|---|---|
| **Unidad de comparación** | Día completo | Ventana temporal $X$ / patrón previo |
| **Criterio principal** | Calendario, clima, tipo de día, holiday | Similitud de forma entre secuencias |
| **Pregunta que responde** | "¿Qué días se parecen a este holiday?" | "¿Dónde ocurrió antes un patrón parecido al actual?" |
| **Búsqueda** | Sobre días históricos etiquetados | Sobre subsecuencias de la serie |
| **Entrada clave** | Vector de contexto $z(d)$ | Ventana reciente $Y$ |
| **Vecinos** | Días similares | Ventanas históricas similares $X_i$ |
| **Salida usada** | Perfil observado del día similar | Subsecuente $X'_i$ de cada ventana similar |
| **Distancias típicas** | Euclidiana, reglas, pesos por calendario/clima | Pearson, Euclidiana, DTW |
| **Interpretabilidad** | Alta — por calendario/contexto | Alta — por patrones históricos análogos |
| **Mejor para** | Holidays, días especiales, demanda con efecto calendario fuerte | Series con patrones repetitivos y memoria temporal |
| **Riesgo** | Depende de elegir bien las variables de contexto | Puede hallar patrones similares pero con contexto distinto |

---

## 3. Formalización del contraste

**Similar Days** — el vecino es el *día histórico*. Se comparan días por contexto,

$$d^{*} \sim d_i \quad\text{(por contexto)},$$

se seleccionan los $k$ vecinos

$$N_k(d^{*}) = \{d_1, \dots, d_k\},$$

y el pronóstico se arma con sus **perfiles observados**:

$$\hat{Y}(d^{*}) = \sum_{d_i \in N_k(d^{*})} \omega_i\, Y_{d_i}.$$

**Método Analog** — el vecino es la *ventana histórica*. Se toma una ventana reciente del objetivo

$$Y = (y_{t-v_1+1}, \dots, y_t),$$

se buscan ventanas históricas similares en forma

$$X_i = (y_{i-v_1+1}, \dots, y_i),$$

y se usan sus **futuros observados** (las subsecuentes)

$$X'_i = (y_{i+1}, \dots, y_{i+v_2})$$

para construir el pronóstico del horizonte:

$$\hat{Y}' = \sum_{i=1}^{k} \omega_i\, X'_i.$$

En compacto:

$$
\underbrace{d^{*} \sim d_i \;\Rightarrow\; \hat{Y}(d^{*}) \leftarrow Y_{d_i}}_{\text{Similar Days}}
\qquad\text{vs.}\qquad
\underbrace{Y \sim X_i \;\Rightarrow\; \hat{Y}' \leftarrow X'_i}_{\text{Analog}}
$$

La diferencia fuerte: Similar Days compara el día objetivo con otros días usando **metadatos
exógenos a nivel día**; Analog compara la **historia reciente de la serie** con patrones históricos
similares y proyecta su continuación.

---

## 4. Por qué Analog es más potente para holidays

Para festivos, la variante Analog no solo dice:

> "busco otros holidays parecidos"

sino:

> "dado este contexto de holiday, busco patrones previos similares y uso lo que ocurrió después como
> trayectoria análoga."

Es decir, no se limita a descriptores de calendario/clima del día: **explota la forma temporal de la
demanda que precede al festivo** (la víspera, los días previos, el régimen reciente del sistema), que
es justo donde se concentra la información predictiva en festivos.

El precio es el riesgo señalado en la tabla: una ventana puede tener una forma muy parecida pero
provenir de un **contexto distinto**. Por eso, en la práctica, los dos enfoques son complementarios —
el filtrado por tipo/familia de holiday (Similar Days) acota el universo de candidatos, y la búsqueda
por forma (Analog) selecciona y pondera dentro de él.

### 5.1 Matiz de implementación — el paso de combinación no es un promedio simple

La fórmula $\hat{Y}' = \sum_i \omega_i\, X'_i$ de la §3 es la forma *conceptual* (analog por promedio
ponderado). La implementación campeona de este repo va más allá en dos puntos, y conviene tenerlos
presentes para no subrepresentar el método frente a Similar Days:

1. **Pool pre-filtrado por observancia, no por calendario.** Los candidatos no se restringen por
   "mismo tipo de festivo", sino por el **clúster de observancia** del objetivo
   (`MATCH_TARGET_CLUSTER`): qué tan fuerte se observa realmente el día, medido por la profundidad de
   la caída de demanda (clústeres F/G/H). Es un filtrado por *comportamiento observado*, más fino que
   el filtrado por familia de Similar Days.

2. **La combinación es una regresión con reducción de dimensionalidad, no un promedio.** En vez de
   $\sum_i \omega_i X'_i$, se ajusta un modelo local (**PCR / PLS / Ridge / Lasso**, elegido por
   *tuning* Optuna por objetivo) que mapea los análogos $\{X_i\}$ a la trayectoria objetivo, seguido
   de un **ajuste de sesgo horario** (factor model con hasta 4 vecinos). Esto permite ajustar **nivel
   y forma** —no solo copiar el promedio de los subsecuentes— cuando los análogos comparten forma pero
   difieren en magnitud por temperatura o crecimiento del sistema.

En notación, el paso final es $\hat{Y}' = g\big(\{X'_i\}_{i\in N_k}\big)$ con $g$ una regresión local
ajustada por objetivo, del cual el promedio ponderado es el caso particular. Ver
[`champion_analog_holiday.md`](champion_analog_holiday.md) §2.2–2.6 para los parámetros exactos.

---

## 5. Redacción para paper

**English (paper-ready):**

> *Similar Days* methods identify historical days that resemble the target day according to calendar,
> weather, or operational attributes. In contrast, the proposed *Analog* method performs a
> sequence-based search: historical pre-holiday windows $X$ are matched against the recent target
> window $Y$, and the subsequent holiday trajectories $X'$ are used as empirical analogs for the
> forecast horizon $Y'$. Therefore, while Similar Days relies primarily on exogenous day-level
> descriptors, the proposed method exploits the temporal shape of the demand trajectory preceding the
> holiday.

**Español:**

> Los métodos *Similar Days* identifican días históricos parecidos al día objetivo usando atributos de
> calendario, clima u operación. En contraste, el método *Analog* propuesto realiza una búsqueda
> basada en secuencias: las ventanas históricas previas al holiday $X$ se comparan con la ventana
> reciente objetivo $Y$, y las trayectorias subsecuentes del holiday $X'$ se usan como analogías
> empíricas para pronosticar $Y'$. Por lo tanto, mientras Similar Days depende principalmente de
> descriptores exógenos a nivel día, el método propuesto explota la forma temporal de la demanda antes
> del holiday.
