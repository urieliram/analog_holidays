"""Similar-Days clásico como benchmark contra analog-holidays.

Implementa la metodología de `docs/similar_days_holidays.md` sobre el MISMO panel
que el campeón (8 series x 19 fechas = 152 celdas), con el mismo corte temporal y
la misma métrica, de modo que la comparación es pareada celda a celda.

Variantes de distancia (docs §2):
  ctx     Opción A — distancia sobre variables de contexto (tipo de festivo, día de
          semana, temporada, nivel reciente de demanda)
  shape   Opción B — similitud de la forma reciente (últimas m horas observadas)
  hybrid  Opción C — alpha*ctx + (1-alpha)*shape

Variantes de combinación (docs §3):
  mean    promedio simple de los perfiles vecinos
  wmean   promedio ponderado por cercanía, kernel exponencial
  scaled  promedio ponderado y RENORMALIZADO por nivel: cada vecino se reescala por
          nivel_reciente_objetivo / nivel_reciente_vecino antes de promediar (§3.4)

Restricción honesta: NO hay variables meteorológicas en el repo, así que el bloque de
clima del vector de contexto se omite. Ambos métodos quedan con el mismo conjunto de
información (calendario + historia de demanda), que es lo que hace justa la comparación.

Uso:  cd /home/uriel/GIT && MPLBACKEND=Agg python3 analog_holidays/experiments/similar_days_benchmark.py
"""
from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/home/uriel/GIT/analog_holidays")
EXP = ROOT / "experiments"
OUT = ROOT / "docs"

SEASON_LENGTH = 38           # ventana reciente, igual que el analog
ISSUE_HOUR = 10              # el pronóstico se emite a las 10:00 del día previo
HOURS = 24

# Familias de festivo (docs §2.1b: 0 mismo festivo, 0.3 misma familia, 1 distinta)
FAMILY_MX = {
    "Christmas Eve": "invierno", "Christmas Day": "invierno",
    "New Year's Eve": "invierno", "New Year's Day": "invierno",
    "Maundy Thursday": "semana_santa", "Good Friday": "semana_santa",
    "Holy Saturday": "semana_santa",
    "Constitution Day": "civico", "Benito Juarez's Birthday": "civico",
    "Labor Day": "civico", "Independence Day": "civico",
    "Mexican Revolution Day": "civico",
}
FAMILY_ERCOT = {
    "Christmas Eve": "invierno", "Christmas Day": "invierno",
    "New Year's Day": "invierno",
    "Thanksgiving Day": "accion_gracias", "Day after Thanksgiving": "accion_gracias",
    "Martin Luther King Jr. Day": "lunes_federal", "Presidents' Day": "lunes_federal",
    "Memorial Day": "lunes_federal", "Labor Day": "lunes_federal",
    "Independence Day": "patriotico", "Veterans Day": "patriotico",
    "Juneteenth National Independence Day": "patriotico",
    "Texas Independence Day": "texas", "San Jacinto Day": "texas",
    "Lyndon B. Johnson Day": "texas",
}

DATASETS = {
    "mx": {
        "demand": ROOT / "holidays" / "holiday_demand_mx.csv",
        "selector": ROOT / "holidays" / "holiday_selector_features.csv",
        "panel_glob": "experiment_*criterion_holiday_identity",
        "family": FAMILY_MX,
        "out_csv": "similar_days_benchmark.csv",
    },
    "ercot": {
        "demand": ROOT / "holidays" / "holiday_demand_ercot.csv",
        "selector": ROOT / "holidays" / "holiday_selector_features_ercot.csv",
        "panel_glob": "experiment_*ercot_criterion_holiday_identity",
        "family": FAMILY_ERCOT,
        "out_csv": "similar_days_benchmark_ercot.csv",
    },
}
CFG: dict = DATASETS["mx"]          # se sobrescribe en main() según sys.argv


# ─────────────────────────────────────────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    """Las 152 celdas del campeón, con su MAPE para comparar pareado."""
    ds = [d for d in sorted(glob.glob(str(EXP / CFG["panel_glob"]))) if "_kcap" not in d]
    # el glob de MX es un prefijo del de ERCOT, así que hay que excluirlo explícitamente
    if not CFG["panel_glob"].startswith("experiment_*ercot"):
        ds = [d for d in ds if "ercot" not in Path(d).name]
    for x in reversed(ds):
        df = pd.read_csv(Path(x) / "metrics.csv")
        if "mape_24_pct" in df.columns:
            df["target_date"] = pd.to_datetime(df["target_date"])
            return df[["unique_id", "target_date", "holiday_label",
                       "mape_24_pct", "mpe_24_pct", "k"]].copy()
    raise SystemExit("no se encontró la corrida campeona")


def load_series(col: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = pd.read_csv(CFG["demand"], parse_dates=["ds"])
    d["date"] = d.ds.dt.normalize()
    d["h"] = d.ds.dt.hour
    W = d.pivot_table(index="date", columns="h", values=col).dropna().sort_index()
    return W, W.to_numpy(dtype=np.float64)


def recent_window(W: pd.DataFrame, day: pd.Timestamp, hours: int) -> np.ndarray | None:
    """Últimas `hours` horas observadas antes de emitir para el festivo `day`.

    El corte es a las ISSUE_HOUR del día previo, así que la ventana termina ahí y
    todo lo que contiene ya ocurrió cuando se emite el pronóstico.
    """
    end_day = day - pd.Timedelta(days=1)
    if end_day not in W.index:
        return None
    idx = W.index.get_loc(end_day)
    flat = W.iloc[: idx + 1].to_numpy(dtype=np.float64).reshape(-1)
    flat = flat[: len(flat) - (HOURS - ISSUE_HOUR)]      # recorta 14 h no observadas
    if flat.size < hours:
        return None
    return flat[-hours:]


def context_distance(t: dict, c: dict, level_scale: float) -> float:
    """docs §2.1 — suma ponderada de distancias por variable (sin bloque de clima)."""
    fam = CFG["family"]
    d_hol = 0.0 if t["anchor"] == c["anchor"] else (
        0.3 if fam.get(t["anchor"]) is not None
        and fam.get(t["anchor"]) == fam.get(c["anchor"]) else 1.0)

    if t["dow"] == c["dow"]:
        d_dow = 0.0
    elif t["dow"] >= 5 and c["dow"] >= 5:
        d_dow = 0.3
    else:
        d_dow = 1.0

    dm = abs(t["month"] - c["month"])
    d_month = min(dm, 12 - dm) / 6.0                      # circular, normalizada

    d_level = abs(t["level"] - c["level"]) / level_scale if level_scale > 0 else 0.0

    return 1.0 * d_hol + 0.5 * d_dow + 0.3 * d_month + 0.5 * min(d_level, 2.0)


def shape_distance(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """docs §2.2 — 1 - Pearson sobre la firma reciente (invariante a escala)."""
    if x_t is None or x_c is None or x_t.size != x_c.size:
        return np.nan
    if np.std(x_t) < 1e-9 or np.std(x_c) < 1e-9:
        return 1.0
    return float(1.0 - np.corrcoef(x_t, x_c)[0, 1])


def combine(profiles: np.ndarray, dists: np.ndarray, scales: np.ndarray,
            how: str, lam: float = 3.0) -> np.ndarray:
    """docs §3 — promedio simple / ponderado / reescalado por nivel."""
    if how == "scaled":
        profiles = profiles * scales[:, None]
    if how == "mean":
        return profiles.mean(axis=0)
    dn = dists - np.nanmin(dists)
    w = np.exp(-lam * dn / (np.nanstd(dn) + 1e-9))
    w = w / w.sum()
    return (profiles * w[:, None]).sum(axis=0)


def mape(a: np.ndarray, p: np.ndarray) -> float:
    den = np.where(np.abs(a) > 1e-9, np.abs(a), np.nan)
    return float(np.nanmean(np.abs(a - p) / den * 100.0))


def mpe(a: np.ndarray, p: np.ndarray) -> float:
    den = np.where(np.abs(a) > 1e-9, np.abs(a), np.nan)
    return float(np.nanmean((a - p) / den * 100.0))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global CFG
    key = sys.argv[1] if len(sys.argv) > 1 else "mx"
    if key not in DATASETS:
        raise SystemExit(f"dataset desconocido {key!r}; opciones: {sorted(DATASETS)}")
    CFG = DATASETS[key]
    print(f"=== DATASET: {key.upper()} ===\n")
    panel = load_panel()
    sel = pd.read_csv(CFG["selector"], parse_dates=["date"])
    sel["date"] = sel["date"].dt.normalize()
    meta = (sel.dropna(subset=["anchor_holiday_name"])
               .drop_duplicates(subset=["date"])
               .set_index("date")[["anchor_holiday_name", "holiday_day_type"]])

    DISTANCES = ["ctx", "shape", "hybrid"]
    COMBOS = ["mean", "wmean", "scaled"]
    KS = [1, 2, 3, 4, 5, 6]
    ALPHA = 0.5

    rows = []
    for uid, g in panel.groupby("unique_id"):
        W, _ = load_series(uid)
        level_scale = float(np.nanstd(W.to_numpy().mean(axis=1)))
        special_days = [d for d in W.index if d in meta.index]

        for _, r in g.iterrows():
            tdate = pd.Timestamp(r.target_date).normalize()
            if tdate not in W.index or tdate not in meta.index:
                continue
            actual = W.loc[tdate].to_numpy(dtype=np.float64)
            x_t = recent_window(W, tdate, SEASON_LENGTH)
            if x_t is None:
                continue
            t_level = float(np.mean(x_t))
            t_ctx = {"anchor": meta.loc[tdate, "anchor_holiday_name"],
                     "dow": tdate.dayofweek, "month": tdate.month, "level": t_level}

            cands = []
            for cday in special_days:
                if cday >= tdate:
                    continue                                  # estrictamente pasado
                x_c = recent_window(W, cday, SEASON_LENGTH)
                if x_c is None:
                    continue
                c_ctx = {"anchor": meta.loc[cday, "anchor_holiday_name"],
                         "dow": cday.dayofweek, "month": cday.month,
                         "level": float(np.mean(x_c))}
                d_ctx = context_distance(t_ctx, c_ctx, level_scale)
                d_shp = shape_distance(x_t, x_c)
                cands.append({
                    "date": cday, "d_ctx": d_ctx, "d_shape": d_shp,
                    "d_hybrid": ALPHA * (d_ctx / 2.3) + (1 - ALPHA) * (d_shp / 2.0),
                    "profile": W.loc[cday].to_numpy(dtype=np.float64),
                    "scale": t_level / c_ctx["level"] if c_ctx["level"] > 1e-9 else 1.0,
                })
            if len(cands) < 2:
                continue

            C = pd.DataFrame(cands)
            rec = {"unique_id": uid, "target_date": tdate.strftime("%Y-%m-%d"),
                   "holiday_label": r.holiday_label,
                   "model_mape": r.mape_24_pct, "model_mpe": r.mpe_24_pct,
                   "n_cand": len(C)}

            for dist, combo, k in itertools.product(DISTANCES, COMBOS, KS):
                col = {"ctx": "d_ctx", "shape": "d_shape", "hybrid": "d_hybrid"}[dist]
                sub = C.nsmallest(k, col)
                if len(sub) < min(k, 2):
                    continue
                P = np.vstack(sub["profile"].to_numpy())
                pred = combine(P, sub[col].to_numpy(dtype=float),
                               sub["scale"].to_numpy(dtype=float), combo)
                rec[f"SD_{dist}_{combo}_k{k}_mape"] = mape(actual, pred)
                rec[f"SD_{dist}_{combo}_k{k}_mpe"] = mpe(actual, pred)
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / CFG["out_csv"], index=False)
    print(f"celdas evaluadas: {len(df)} (panel campeón: {len(panel)})")
    print(f"candidatos por celda: med={df.n_cand.median():.0f} "
          f"min={df.n_cand.min()} max={df.n_cand.max()}\n")

    mcols = [c for c in df.columns if c.endswith("_mape") and c.startswith("SD_")]
    summ = (df[mcols].median().sort_values().rename("mediana").to_frame())
    summ["config"] = summ.index.str.replace("SD_", "", regex=False).str.replace("_mape", "", regex=False)

    print("=== TOP 15 CONFIGURACIONES SIMILAR-DAYS (mediana MAPE_24) ===")
    print(f"  {'config':34s} {'mediana':>8s}")
    for c, row in summ.head(15).iterrows():
        print(f"  {row['config']:34s} {row['mediana']:7.3f}%")
    print(f"\n  >>> analog-holidays (campeón)      {df.model_mape.median():7.3f}%")

    best = summ.index[0]
    print(f"\n=== PAREADO: campeón vs MEJOR Similar-Days ({summ.iloc[0]['config']}) ===")
    m = df.dropna(subset=["model_mape", best])
    _, p = wilcoxon(m.model_mape, m[best])
    d = m.model_mape - m[best]
    print(f"  n={len(m)} | SD={m[best].median():.3f}%  analog={m.model_mape.median():.3f}%")
    print(f"  analog gana {100*(d<0).mean():.0f}% de celdas | delta={d.median():+.3f} pp | "
          f"skill={1-m.model_mape.median()/m[best].median():+.3f} | p={p:.5f}")

    print("\n=== EFECTO DE LA RENORMALIZACIÓN DE NIVEL (docs §3.4) ===")
    for dist in DISTANCES:
        for k in (2, 3, 4):
            a = f"SD_{dist}_wmean_k{k}_mape"
            b = f"SD_{dist}_scaled_k{k}_mape"
            if a in df.columns and b in df.columns:
                print(f"  {dist:7s} k={k}  sin escalar {df[a].median():6.3f}%  ->  "
                      f"escalado {df[b].median():6.3f}%   ({df[b].median()-df[a].median():+.3f} pp)")

    print("\n=== MEJOR SD POR FAMILIA DE DISTANCIA (cualquier k/combinación) ===")
    for dist in DISTANCES:
        sub = summ[summ.index.str.startswith(f"SD_{dist}_")]
        if len(sub):
            print(f"  {dist:7s} {sub.iloc[0]['config']:30s} {sub.iloc[0]['mediana']:6.3f}%")

    print(f"\nCSV -> {OUT / CFG['out_csv']}")


if __name__ == "__main__":
    main()
