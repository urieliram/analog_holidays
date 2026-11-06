"""Does the head-level anchor stack with the cluster criterion?"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/home/uriel/GIT/analog_holidays/experiments")
RAW = "mape_24_pct"
ANC = "mape_24_head_anchored_pct"
RAT = "mape_24_head_ratio_pct"


def latest(criterion: str) -> pd.DataFrame | None:
    ds = sorted(glob.glob(str(EXP / f"experiment_*criterion_{criterion}")))
    ds = [d for d in ds if "_kcap" not in Path(d).name]
    for d in reversed(ds):
        df = pd.read_csv(Path(d) / "metrics.csv")
        if ANC in df.columns:
            df["criterion"] = criterion
            return df
    return None


def main() -> None:
    crits = ["holiday_identity", "observance_tier", "seasonal_heat_cold"]
    runs = {c: latest(c) for c in crits}
    runs = {c: d for c, d in runs.items() if d is not None}
    if not runs:
        raise SystemExit("no anchored runs found")

    print("=== ¿SE SUMAN LOS DOS LEVERS? (mediana MAPE_24) ===")
    rows = []
    for c, d in runs.items():
        raw, anc, rat = d[RAW], d[ANC], d[RAT]
        rows.append({
            "criterio": c, "crudo": raw.median(), "anclado_mpe": anc.median(),
            "anclado_ratio": rat.median(),
            "mejora_pp": raw.median() - rat.median(),
            "mejora_%": 100 * (1 - rat.median() / raw.median()),
            "|MPE|_crudo": d.mpe_24_pct.abs().median(),
            "|MPE|_anclado": d["mpe_24_head_ratio_pct"].abs().median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== PRUEBA PAREADA crudo vs anclado(ratio), por criterio ===")
    for c, d in runs.items():
        m = d.dropna(subset=[RAW, RAT])
        _, p = wilcoxon(m[RAW], m[RAT])
        diff = m[RAT] - m[RAW]
        print(f"  {c:22s} n={len(m):3d} delta_mediana={diff.median():+.3f} pp | "
              f"anclado gana {100*(diff<0).mean():3.0f}% | p={p:.5f}{'  SIGNIF' if p<0.05 else ''}")

    print("\n=== COMPUERTA: anclar solo si el sesgo de cabeza supera un umbral ===")
    for c, d in runs.items():
        m = d.dropna(subset=[RAW, RAT]).copy()
        head_bias = m["mpe_14_pct"].abs()
        print(f"  {c}")
        for thr in [0.0, 1.0, 2.0, 3.0, 5.0]:
            gated = np.where(head_bias > thr, m[RAT], m[RAW])
            print(f"    umbral |mpe_14|>{thr:.0f}%  ->  mediana {np.median(gated):.3f}%  "
                  f"(ancla aplicada a {100*(head_bias>thr).mean():3.0f}% de celdas)")

    print("\n=== POR REGION (mediana, criterio ganador) ===")
    best = min(runs, key=lambda c: runs[c][RAT].median())
    d = runs[best]
    piv = d.groupby("unique_id").agg(crudo=(RAW, "median"), anclado=(RAT, "median"))
    piv["delta"] = piv.anclado - piv.crudo
    print(f"  criterio={best}")
    print(piv.round(2).sort_values("delta").to_string())

    print("\n=== POR FESTIVO (criterio ganador) ===")
    pivh = d.groupby("holiday_label").agg(crudo=(RAW, "median"), anclado=(RAT, "median"))
    pivh["delta"] = pivh.anclado - pivh.crudo
    print(pivh.round(2).sort_values("delta").to_string())

    print("\n=== descomposicion nivel/forma tras anclar ===")
    for c, d in runs.items():
        lvl = d["mpe_24_head_ratio_pct"].abs().median()
        tot = d[RAT].median()
        print(f"  {c:22s} MAPE={tot:.3f}%  nivel={lvl:.3f}%  forma={tot-lvl:.3f} pp")


if __name__ == "__main__":
    main()
