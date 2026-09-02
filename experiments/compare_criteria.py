"""Compare every analog-cluster criterion on the same 152-cell MX panel."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/home/uriel/GIT/analog_holidays/experiments")
KEY = ["unique_id", "target_date"]
M = "mape_24_pct"

POOL = {  # clusters, median pool per series (from gen_selectors)
    "seasonal_heat_cold": (2, 40), "shape_pearson_CDE_map_FGH": (3, 25),
    "seasonal_winter_sprint_fall": (3, 35), "observance_tier": (3, 34),
    "observance_tier_depth": (4, 18), "best_matching_weekday": (7, 9),
    "holiday_identity": (12, 6),
}


def load_all() -> dict[str, pd.DataFrame]:
    runs = {}
    for d in sorted(glob.glob(str(EXP / "experiment_*criterion_*"))):
        p = Path(d)
        crit = p.name.split("criterion_", 1)[1]
        df = pd.read_csv(p / "metrics.csv")
        df["target_date"] = pd.to_datetime(df["target_date"]).dt.strftime("%Y-%m-%d")
        df["criterion"] = crit
        runs[crit] = df
    return runs


def main() -> None:
    runs = load_all()
    if not runs:
        raise SystemExit("no criterion runs found")

    print("=== RESUMEN POR CRITERIO (panel MX 8x19) ===")
    rows = []
    for c, df in runs.items():
        ok = df[M].notna()
        nc, pool = POOL.get(c, (np.nan, np.nan))
        rows.append({
            "criterio": c, "clusters": nc, "pool_med": pool,
            "n_ok": int(ok.sum()), "fallos": int((~ok).sum()),
            "mediana": df[M].median(), "media": df[M].mean(),
            "p90": df[M].quantile(0.90),
            "|MPE|": df["mpe_24_pct"].abs().median(),
            "k_med": df["k"].mean(),
        })
    summary = pd.DataFrame(rows).sort_values("mediana").reset_index(drop=True)
    print(summary.round(2).to_string(index=False))

    best = summary.iloc[0]["criterio"]
    print(f"\n=== PAREADO vs mejor criterio ({best}) ===")
    for c in summary["criterio"]:
        if c == best:
            continue
        m = runs[best].merge(runs[c], on=KEY, suffixes=("_a", "_b")).dropna(subset=[f"{M}_a", f"{M}_b"])
        if len(m) < 10:
            print(f"  {c:32s} n={len(m)} insuficiente"); continue
        _, p = wilcoxon(m[f"{M}_a"], m[f"{M}_b"])
        d = m[f"{M}_b"] - m[f"{M}_a"]
        print(f"  {c:32s} n={len(m):3d} delta_mediana={d.median():+6.3f} pp | "
              f"{best} gana {100*(d>0).mean():3.0f}% | p={p:.4f}{'  SIGNIF' if p<0.05 else ''}")

    allr = pd.concat(runs.values())
    print("\n=== POR FESTIVO (mediana mape_24) ===")
    pivh = allr.pivot_table(index="holiday_label", columns="criterion", values=M, aggfunc="median")
    pivh = pivh[summary["criterio"].tolist()]
    pivh["MEJOR"] = pivh.idxmin(axis=1)
    print(pivh.round(2).to_string())

    print("\n=== POR REGION (mediana mape_24) ===")
    pivr = allr.pivot_table(index="unique_id", columns="criterion", values=M, aggfunc="median")
    pivr = pivr[summary["criterio"].tolist()]
    pivr["MEJOR"] = pivr.idxmin(axis=1)
    print(pivr.round(2).to_string())

    print("\n=== ORACULO: mejor criterio por festivo ===")
    per = allr.groupby(["holiday_label", "criterion"])[M].median().reset_index()
    win = per.loc[per.groupby("holiday_label")[M].idxmin()]
    print(f"  mediana del oraculo por festivo = {win[M].median():.3f}%  "
          f"vs mejor criterio unico = {summary.iloc[0]['mediana']:.3f}%")
    print(f"  criterios que ganan al menos un festivo: {win.criterion.nunique()} de {len(runs)}")
    print(win.criterion.value_counts().to_string())


if __name__ == "__main__":
    main()
