"""How much of the level-anchor gain survives using only issue-time information?"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/home/uriel/GIT/analog_holidays/experiments")
RAW = "mape_24_pct"
VARIANTS = {
    "head_ratio (NO valido)": "mape_24_head_ratio_pct",
    "exante_level": "mape_24_exante_level_pct",
    "exante_drop": "mape_24_exante_drop_pct",
}
CRITS = ["holiday_identity", "observance_tier", "seasonal_heat_cold"]


def latest(c: str) -> pd.DataFrame | None:
    ds = [d for d in sorted(glob.glob(str(EXP / f"experiment_*criterion_{c}"))) if "_kcap" not in d]
    for x in reversed(ds):
        df = pd.read_csv(Path(x) / "metrics.csv")
        if "mape_24_exante_drop_pct" in df.columns:
            return df
    return None


def main() -> None:
    runs = {c: latest(c) for c in CRITS}
    runs = {c: d for c, d in runs.items() if d is not None}
    if not runs:
        raise SystemExit("no ex-ante runs found yet")

    print("=== MEDIANA MAPE_24 POR VARIANTE ===")
    rows = []
    for c, d in runs.items():
        r = {"criterio": c, "crudo": d[RAW].median()}
        for name, col in VARIANTS.items():
            r[name] = d[col].median() if col in d.columns else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== PRUEBA PAREADA vs CRUDO ===")
    for c, d in runs.items():
        print(f"  {c}")
        for name, col in VARIANTS.items():
            if col not in d.columns:
                continue
            m = d.dropna(subset=[RAW, col])
            _, p = wilcoxon(m[RAW], m[col])
            diff = m[col] - m[RAW]
            flag = "MEJOR" if diff.median() < 0 and p < 0.05 else ("peor" if diff.median() > 0 else "")
            print(f"    {name:24s} delta={diff.median():+.3f} pp | gana {100*(diff<0).mean():3.0f}% | "
                  f"p={p:.5f}  {flag}")

    print("\n=== ¿CUANTO SOBREVIVE? (mejor criterio) ===")
    best = min(runs, key=lambda c: runs[c][RAW].median())
    d = runs[best]
    base = d[RAW].median()
    inval = d["mape_24_head_ratio_pct"].median()
    gain_inval = base - inval
    for name, col in VARIANTS.items():
        if col not in d.columns or name.startswith("head"):
            continue
        g = base - d[col].median()
        pct = 100 * g / gain_inval if gain_inval else np.nan
        print(f"  {name:16s} gana {g:+.3f} pp  =  {pct:5.1f}% de la ganancia no-valida ({gain_inval:.3f} pp)")

    print("\n=== FACTORES APLICADOS (mejor criterio) ===")
    for col in ["head_anchor_factor_ratio", "exante_level_factor", "exante_drop_factor"]:
        if col in d.columns:
            s = d[col]
            print(f"  {col:26s} mediana={s.median():.4f}  p10={s.quantile(.1):.4f}  "
                  f"p90={s.quantile(.9):.4f}  |desvio|medio={(s-1).abs().mean():.4f}")

    print("\n=== NIVEL RESIDUAL ===")
    for c, d2 in runs.items():
        parts = [f"crudo={d2.mpe_24_pct.abs().median():.3f}"]
        for name, col in VARIANTS.items():
            mcol = col.replace("mape_", "mpe_")
            if mcol in d2.columns:
                parts.append(f"{name.split(' ')[0]}={d2[mcol].abs().median():.3f}")
        print(f"  {c:22s} |MPE|: " + "  ".join(parts))


if __name__ == "__main__":
    main()
