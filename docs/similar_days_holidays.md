# Similar Days para pronóstico de demanda en días festivos

**Alcance.** Nota metodológica sobre *Similar Days* (días análogos) aplicada al pronóstico de la
demanda eléctrica en días festivos (*holidays*). Documenta la idea central, cómo definir
"parecido", cómo construir el pronóstico una vez encontrados los vecinos, y cómo aterrizar el método
al caso concreto de demanda con festivos. Es un documento de referencia conceptual; para la
implementación productiva de este repo ver [`champion_analog_holiday.md`](champion_analog_holiday.md).

---

## 0. ¿Por qué un método distinto para festivos?

Un pronóstico de demanda horaria/diaria normal se apoya en **estacionalidad semanal**, **temperatura**,
**rezagos recientes** y **tendencia**. Pero los festivos *rompen* esa estructura: cambian el nivel,
la forma horaria, la hora del pico e incluso la relación con la temperatura.

La gran diferencia conceptual es pasar de:

> "usa la semana pasada / el mismo día de la semana"

a:

> "usa festivos históricamente comparables al festivo actual".

A un festivo no le conviene mezclarse con "cualquier lunes" o "cualquier martes", sino con **días
especiales comparables**: otros *Christmas*, otros *Independence Day*, otros puentes, o al menos otros
días con un patrón operativo y social parecido.

---

## 1. La metodología en una sola vista

Versión canónica en 5 pasos para pronosticar el perfil de demanda del festivo objetivo $d^{*}$
(por ejemplo, las 24 horas del próximo 25 de diciembre o del próximo lunes feriado).

1. **Definir el festivo objetivo** y construir su vector de contexto con la información disponible
   *antes* de que ocurra el día.
2. **Construir el mismo vector** para cada día histórico elegible.
3. **Calcular una distancia** $\Delta(d^{*}, d)$ entre el objetivo y cada festivo histórico.
4. **Seleccionar los $k$ vecinos** más cercanos $N_k(d^{*})$.
5. **Construir el pronóstico** a partir de los perfiles observados de esos vecinos (promedio,
   promedio ponderado, regresión local o perfil escalado).

### Paso 1 — Vector de contexto del objetivo

$$z(d^{*}) = \{\text{tipo de festivo},\ \text{día de la semana},\ \text{mes/temporada},\ \text{temperatura esperada},\ \text{demanda reciente},\ \dots\}$$

Variables de contexto típicas:

| Bloque | Variables |
|---|---|
| **Calendario / tipo** | tipo de festivo (Navidad, Año Nuevo, Independencia…); fijo o movible; si cae lunes/fin de semana/mitad de semana; víspera o post-festivo; si forma puente o periodo vacacional |
| **Clima** | temperatura mín/máx/promedio esperada; humedad; lluvia; sensación térmica |
| **Estado reciente de la demanda** | demanda de ayer; demanda de hace 7 días; promedio últimas 24 h / 48 h; carga de la víspera |
| **Contexto estacional** | mes / estación; vacaciones escolares / verano / invierno; cercanía a otros festivos |

### Paso 2 — Buscar festivos "parecidos" en el histórico

Para cada día histórico $d$ del conjunto de entrenamiento se construye el mismo vector $z(d)$, se
calcula la distancia $\Delta(d^{*}, d)$, y se conservan los $k$ más cercanos:

$$N_k(d^{*}) = \{d_1, d_2, \dots, d_k\}$$

---

## 2. ¿Cómo se define "parecido" en festivos?

Este es el corazón del método. No basta con "día parecido por temperatura": no todos los festivos se
comportan igual.

- *Christmas* no se parece a un *Labor Day*.
- Un lunes feriado no se parece necesariamente a un 25 de diciembre.
- La víspera de un festivo puede parecerse más a otra víspera que al festivo mismo.
- Algunos festivos se parecen por **tipo de actividad** (industria cerrada, comercio abierto, turismo
  alto, etc.).

Por eso la similitud se construye con tres bloques: **calendario/tipo**, **meteorología** y **patrón
reciente de demanda antes del festivo**.

### 2.1 Opción A — Distancia sobre variables de contexto

$$\Delta(d^{*}, d) = \sum_{j=1}^{p} w_j\, \delta_j\big(z_j(d^{*}),\, z_j(d)\big)$$

donde $z_j$ es la variable $j$, $w_j$ su peso, y $\delta_j$ la distancia para esa variable.

**Distancias por variable (ejemplos):**

**a) Temperatura** — escalar o sobre el perfil de 24 h:

$$\delta_T = \lvert T(d^{*}) - T(d) \rvert \qquad\text{o}\qquad \delta_T = \big\lVert T_{1:24}(d^{*}) - T_{1:24}(d) \big\rVert_2$$

**b) Tipo de festivo** — indicador simple o refinado por familia:

$$\delta_{\text{holtype}} = \begin{cases} 0 & \text{mismo tipo} \\ 1 & \text{distinto} \end{cases}
\qquad
\delta_{\text{holtype}} = \begin{cases} 0 & \text{festivo exacto} \\ 0.3 & \text{misma familia} \\ 1 & \text{familia distinta} \end{cases}$$

Por ejemplo, *Christmas Eve* y *Christmas Day* pueden ser "familia navideña"; varios puentes oficiales
otra familia; festivos religiosos otra más.

**c) Día de la semana** — con penalización menor si ambos son fin de semana:

$$\delta_{\text{dow}} = \begin{cases} 0 & \text{mismo día} \\ 0.3 & \text{ambos fin de semana} \\ 1 & \text{distintos} \end{cases}$$

**d) Estado reciente de la demanda** — p. ej. demanda promedio de las 24 h previas:

$$\delta_{\text{load-recent}} = \lvert \bar{L}^{*} - \bar{L}_d \rvert$$

### 2.2 Opción B — Similitud del patrón reciente (forma)

En vez de comparar solo metadatos, se compara la **forma reciente** de la demanda antes del festivo.
Se toma una "firma reciente" de las últimas $m$ horas observadas antes del objetivo:

$$x^{*} = (y_{t-m+1}, \dots, y_t)$$

y la ventana equivalente antes de cada festivo histórico:

$$x_d = (y_{d-m+1}, \dots, y_d)$$

Se comparan con:

- **Euclidiana:** $\lVert x^{*} - x_d \rVert_2$
- **Correlación de Pearson:** $1 - \rho(x^{*}, x_d)$
- **DTW** si se quiere tolerar pequeños desplazamientos de forma.

Esta parte es potente porque la demanda en festivos depende mucho de cómo venía comportándose el
sistema en la víspera y los días previos.

### 2.3 Opción C — Mezcla de contexto + patrón reciente (recomendada)

$$\Delta(d^{*}, d) = \alpha\, \Delta_{\text{holiday-context}}(d^{*}, d) + (1-\alpha)\, \Delta_{\text{recent-shape}}(d^{*}, d)$$

- $\Delta_{\text{holiday-context}}$: tipo de festivo, día de semana, puente, clima, temporada.
- $\Delta_{\text{recent-shape}}$: similitud entre las últimas horas antes del objetivo y antes del histórico.

---

## 3. Una vez encontrados los vecinos, ¿cómo se pronostica?

Sea $Y_d = (y_{d,1}, \dots, y_{d,H})$ el perfil observado del festivo $d$, con horizonte $H = 24$.

### 3.1 Promedio simple de perfiles vecinos

$$\hat{Y}(d^{*}) = \frac{1}{k} \sum_{i=1}^{k} Y_{d_i}
\qquad\Longleftrightarrow\qquad
\hat{y}_h(d^{*}) = \frac{1}{k} \sum_{i=1}^{k} y_{d_i, h},\quad h = 1, \dots, 24$$

### 3.2 Promedio ponderado por cercanía

$$\hat{Y}(d^{*}) = \sum_{i=1}^{k} \omega_i\, Y_{d_i}
\qquad
\omega_i = \frac{\exp(-\lambda \Delta_i)}{\sum_{j=1}^{k} \exp(-\lambda \Delta_j)}
\quad\text{o}\quad
\omega_i = \frac{1/\Delta_i}{\sum_{j=1}^{k} 1/\Delta_j}$$

Los festivos más parecidos pesan más.

### 3.3 Regresión local sobre los vecinos

En lugar de promediar, se ajusta un modelo local solo con los $k$ vecinos:

$$\hat{y}_h = f_h(X_{\text{vecinos}})$$

con regresión lineal, **ridge/lasso**, **PCR/PLS** o random forest local. Útil cuando dos festivos
tienen forma parecida pero **nivel distinto** por temperatura o crecimiento del sistema.

### 3.4 "Copiar y escalar" el festivo más parecido

Se toma el perfil del vecino más cercano $Y_{d_1}$ y se reescala por nivel o temperatura:

$$\hat{Y}(d^{*}) = a + b\, Y_{d_1}
\qquad\text{o}\qquad
\hat{Y}(d^{*}) = \frac{\text{nivel reciente objetivo}}{\text{nivel reciente del vecino}}\, Y_{d_1}$$

Sirve cuando la **forma** se conserva pero el **nivel** cambia.

---

## 4. Esquema matemático completo

Serie horaria de demanda $y_t$; pronosticar el festivo $d^{*}$ con horizonte $H = 24$.

**Contexto del objetivo:**

$$z^{*} = \big(\text{holiday\_type}(d^{*}),\ \text{dow}(d^{*}),\ \text{bridge}(d^{*}),\ T_{\max}(d^{*}),\ T_{\min}(d^{*}),\ \bar{y}_{t-24:t-1},\ y_{t-24},\ y_{t-168}\big)$$

donde $y_{t-24}$ es la demanda del día previo a la misma hora y $y_{t-168}$ la de hace una semana.

**Distancia a cada histórico:** $\ \Delta(d^{*}, d) = \sum_{j=1}^{p} w_j\, \delta_j(z_j^{*}, z_j(d))$

**Selección de vecinos:** $\ N_k(d^{*}) = \operatorname{arg\,min}^{(k)}_{d \in D_{\text{holidays}}} \Delta(d^{*}, d)$

**Pronóstico (promedio ponderado):**

$$\hat{Y}(d^{*}) = \sum_{d \in N_k(d^{*})} \omega_d\, Y_d,
\qquad
\omega_d = \frac{\exp(-\lambda \Delta(d^{*}, d))}{\sum_{j \in N_k(d^{*})} \exp(-\lambda \Delta(d^{*}, j))}$$

---

## 5. Pseudocódigo

```text
Similar Days clásico para festivos
Input:
  - Serie histórica de demanda y_t
  - Variables exógenas / calendario
  - Festivo objetivo d*
  - Número de vecinos k

1. Construir vector de contexto z(d*) del festivo objetivo
2. Definir candidatos históricos: festivos / días especiales comparables
3. Para cada festivo histórico d elegible:
      a) construir z(d)
      b) calcular distancia Δ(d*, d)
4. Ordenar festivos por menor distancia
5. Seleccionar los k más cercanos N_k(d*)
6. Recuperar los perfiles reales Y_d de esos k festivos
7. Construir pronóstico:
      - promedio simple, o
      - promedio ponderado, o
      - regresión local con esos vecinos
8. Devolver el perfil pronosticado del festivo d*
```

---

## 6. Variantes comunes en la literatura

- **Variante 1 — Filtro duro + ranking.** Primero reglas de elegibilidad (mismo tipo de festivo,
  misma temporada, excluir festivos demasiado viejos, exigir comparabilidad víspera/festivo/post), y
  *después* rankear por distancia. Ej.: candidatos = "todos los *Christmas Day* históricos"; si hay
  pocos, ampliar a "familia navideña"; dentro de esos, elegir por temperatura y patrón reciente.
- **Variante 2 — Festivos por familia o clúster.** Agrupar primero los festivos por comportamiento
  (Navidad/Año Nuevo, cívicos, puentes de lunes, religiosos, vacacionales). Un nuevo festivo se asigna
  a una familia y solo busca vecinos dentro de ella. Muy útil con **pocos años de historia**.
- **Variante 3 — Similar days + modelo local.** Los vecinos no se promedian: seleccionan el
  **subconjunto de entrenamiento** de un modelo local (regresión o red).
- **Variante 4 — Ajuste por tendencia.** Como el sistema crece con los años, un festivo de hace 5–8
  años se ajusta: $\hat{Y}(d^{*}) = \gamma\, Y_{d_i}$, donde $\gamma$ depende del crecimiento anual,
  del nivel reciente, o de la relación entre la víspera actual y la del festivo histórico.

---

## 7. Ventajas y desventajas

**Ventajas**

1. **Muy interpretable:** "el pronóstico se construyó con 8 festivos históricos parecidos en tipo,
   temperatura y patrón reciente de demanda".
2. **Captura patrones atípicos** que un modelo semanal normal no ve: trata el festivo como un
   **régimen especial**.
3. **Aprovecha conocimiento de calendario:** "*Christmas Eve* solo se compara con *Christmas Eve*",
   "puentes solo con puentes".
4. **Funciona bien cuando la forma del festivo se repite** año con año (huella horaria característica).

**Desventajas**

1. **Pocos datos:** quizá solo 5–10 observaciones de un festivo específico.
2. **La definición de similitud es crítica:** mezclar festivos de naturaleza distinta degrada mucho.
3. **Cambio estructural del sistema:** un *Christmas* de hace 8 años puede no representar el sistema
   actual.
4. **Dependencia del contexto exacto:** si cayó en lunes o viernes, si hubo puente largo, si estuvo
   pegado a fin de semana, si hubo clima extremo.

---

## 8. Aterrizaje al caso de demanda con festivos

### A) No tratar todos los festivos igual — separar por familias

| Familia | Ejemplos | Características |
|---|---|---|
| **1. Cierre general / familiar** | Christmas, New Year, Thanksgiving | fuerte cambio de forma horaria; caída o desplazamiento del pico; mucho efecto residencial/comercial |
| **2. Lunes festivos / puentes** | Constitution Day movido a lunes, Revolution Day movido a lunes, Labor Day con puente | parecen mezcla de domingo + lunes; efecto claro de puente y turismo |
| **3. Religiosos o locales** | — | la forma depende mucho de la región y la actividad local |

### B) Contexto del objetivo

$$z(d^{*}) = (\text{holiday\_family},\ \text{holiday\_exact},\ \text{dow},\ \text{bridge},\ \text{season},\ T_{\max},\ T_{\min},\ \text{load\_yesterday},\ \text{load\_lastweek},\ \text{recent\_profile})$$

### C) Distancia híbrida recomendada

$$\Delta(d^{*}, d) = w_1\, \Delta_{\text{holiday-type}} + w_2\, \Delta_{\text{calendar}} + w_3\, \Delta_{\text{temperature}} + w_4\, \Delta_{\text{recent-load-shape}}$$

- $\Delta_{\text{holiday-type}}$: penaliza si el histórico no es del mismo tipo o familia.
- $\Delta_{\text{calendar}}$: si cae lunes/viernes/fin de semana; si es víspera/festivo/post; si hay puente.
- $\Delta_{\text{temperature}}$: temperatura esperada del objetivo vs. observada en el histórico.
- $\Delta_{\text{recent-load-shape}}$: demanda de las últimas 24–72 h antes del objetivo vs. antes del histórico.

### D) Forecast final — tres buenas opciones para festivos

1. **Promedio ponderado** de vecinos — simple y muy interpretable.
2. **Vecino más parecido escalado** — útil con pocos datos.
3. **Regresión local** sobre vecinos — la mejor si hay suficientes años, porque ajusta por clima y
   nivel reciente.

---

## 9. Recomendación concreta — *Hybrid Similar Holiday Days*

1. **Restringir candidatos por tipo:** mismo festivo exacto, o misma familia, o misma clase
   operacional (festivo fuerte, puente, víspera, post-festivo).
2. **Distancia híbrida** por candidato:
   $$\Delta(d^{*}, d) = w_1\, \Delta_{\text{recent-shape}} + w_2\, \Delta_{\text{temperature}} + w_3\, \Delta_{\text{holiday-type}} + w_4\, \Delta_{\text{calendar}}$$
3. **Tomar $k$ vecinos** $N_k(d^{*})$.
4. **En vez de promedio simple,** usar promedio ponderado o regresión local:
   $$\hat{y}_h = \sum_{d \in N_k(d^{*})} \omega_d\, y_{d,h} \qquad\text{o ajustar Ridge/PCR/PLS solo con esos vecinos.}$$

**Por tipo de festivo:**

- **Grandes y distintivos** (Christmas, New Year, Thanksgiving): mismo festivo histórico; si no
  alcanza, ampliar a la familia; forecast con weighted average o perfil escalado.
- **Lunes festivos / puentes:** festivos de la misma familia; mucho peso a si cae en lunes, a la
  temperatura, a la demanda de la víspera y a la del viernes/domingo previos.
- **Vísperas y post-festivos:** tratarlos como **clases separadas** (*holiday eve* / *holiday* /
  *day after holiday*), porque sus perfiles son muy distintos entre sí.

---

## 10. En una frase

> **Similar Days para demanda con festivos:** identificar festivos históricos comparables al objetivo
> usando tipo de festivo, calendario, clima y patrón reciente de demanda, y usar sus perfiles de
> carga —promediados, ponderados o ajustados localmente— para construir el pronóstico del festivo
> futuro.

---

### Relación con la implementación de este repo

Este documento es la base conceptual del pipeline analógico productivo. Para la configuración
campeona concreta (selección de análogos, regresión con reducción de dimensionalidad, ajuste de
sesgo y *tuning* rolling por objetivo) ver [`champion_analog_holiday.md`](champion_analog_holiday.md)
y el benchmark en [`seasonal_naive_benchmark.md`](seasonal_naive_benchmark.md).
