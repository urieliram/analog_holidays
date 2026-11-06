# Benchmark: analog-holidays vs el pronóstico operativo de CENACE

**Fecha:** 2026-09-08 · **Panel:** 4 festivos × 8 series = 32 celdas · **Script:** `experiments/benchmark_cenace.py`

Hasta ahora el método sólo se había medido contra líneas base *reconstruidas por
nosotros* —naive estacional y Similar-Days—. Esta es la primera comparación contra el
**incumbente real**: el pronóstico que CENACE publica y usa en operación.

---

## 1. Veredicto

**CENACE gana, y por un margen amplio.**

| | mediana MAPE_24 | media |
|---|---|---|
| analog-holidays | 3.726 % | 5.060 % |
| **CENACE** | **2.235 %** | 3.194 % |

analog gana el **21.9 %** de las celdas · delta mediano **+1.265 pp** · skill **−0.667** · **p = 0.0026**

No hay forma de presentar esto como empate. El método pierde contra el operador.

---

## 2. Pero la derrota no es uniforme

| festivo | analog | CENACE | delta | gana analog | n |
|---|---|---|---|---|---|
| Jueves Santo (2026-04-02) | 2.905 | **1.437** | +1.468 | 12 % | 8 |
| Viernes Santo (2026-04-03) | 3.297 | **1.720** | +1.577 | 25 % | 8 |
| Sábado Santo (2026-04-04) | 3.820 | **1.843** | +1.977 | 12 % | 8 |
| Día del Trabajo (2026-05-01) | **5.645** | 6.139 | −0.494 | 38 % | 8 |

- **Semana Santa (24 celdas):** CENACE 1.608 % vs analog 3.092 %, p = 0.0003. Aplastante.
- **Día del Trabajo (8 celdas):** analog 5.645 % vs CENACE 6.139 %, p = 0.95. **Empate estadístico**, y es el día más difícil para ambos.

Tres de los cuatro festivos son un solo bloque de Semana Santa. El único festivo
independiente del panel es justamente donde el método no pierde.

---

## 3. Lo que más daña la premisa del paper

El artículo argumenta que los festivos son especialmente difíciles y por eso merecen
tratamiento aparte. Para CENACE **eso no se sostiene**:

| | días | mediana MAPE | media |
|---|---|---|---|
| día ordinario | 1236 | 2.154 % | 3.084 % |
| festivo | 38 | 2.321 % | 3.095 % |

El error de CENACE en festivos es apenas **1.08×** el de un día ordinario. Por región la
razón va de **0.36** (NTE, SIN — les va *mejor* en festivos) a **2.12** (NES).

---

## 4. Verificaciones antes de creerlo

Dos riesgos reales que habrían invalidado la medición:

**a) Alineación horaria.** La base etiqueta inicio de intervalo y el CSV del repo fin de
intervalo. Probando desfases, `+1 h` lleva el MAPE de CENACE de 3.357 % a **1.878 %**. El
benchmark aplica el ajuste. Sin él, CENACE aparecería artificialmente peor.

**b) ¿Es ex ante o nowcast?** Los archivos diarios se descargan a las **23:05 del mismo día
que cubren**, así que un pronóstico revisado intradía era plausible. Refutado por el perfil
de error por hora, que es plano:

| horas | MAPE mediano |
|---|---|
| 0–5 | 1.522 % |
| 6–17 | 1.475 % |
| 18–23 | 1.690 % |

Un nowcast daría error casi nulo en las horas tempranas. Es un pronóstico genuino.

⚠️ Lo que **no** se pudo verificar: el horizonte exacto de emisión no está registrado en la
base. Es el pronóstico operativo del día publicado por CENACE, la referencia más cercana a
nuestra emisión a D−1, pero no está confirmado hora por hora.

---

## 5. Qué implica para el paper

**No se puede reclamar superioridad operativa.** Lo defendible sigue siendo lo que ya está
escrito: el método supera al naive estacional y a Similar-Days **con el mismo conjunto de
información**. Esta comparación debe reportarse, no omitirse.

Dos atenuantes que son reales, no excusas:

1. **CENACE usa clima y nosotros no.** Ya medimos que la anomalía de grados-día explica
   18–25 % de la varianza del sesgo de nivel (`RESULTADO_TECHO.md` §8.5). Es exactamente la
   brecha que este benchmark expone, ahora con un precio concreto: ~1.5 pp de MAPE.
2. **Panel chico y sesgado**: 4 festivos, 3 de ellos un mismo bloque de Semana Santa.

La lectura constructiva es que este benchmark **fija el objetivo**. El método sin variables
exógenas está a 1.5 pp del operador; el clima es la vía conocida para cerrar esa distancia,
y ya está cuantificada. Conviene repetir esta comparación cuando el tramo con pronóstico
CENACE cubra más festivos —16 de septiembre, 16 de noviembre, Navidad y Año Nuevo de 2026—
y después de integrar la corrección climática.

---

## 6. Reproducción

```bash
/usr/bin/python3 analog_holidays/experiments/sync_demand_from_pml.py
/usr/bin/python3 analog_holidays/experiments/benchmark_cenace.py
```

Salida: `docs/benchmark_cenace_cells.csv` (32 celdas pareadas) y
`docs/benchmark_cenace_daily.csv` (1274 series-día para el contexto de §3).
