"""Seasonal-naive baselines on the exact champion panel, paired cell by cell.

Baseline definition (as specified): for each (region, target holiday), take the
SAME holiday in previous years and combine their 24-h profiles with a mean or a
median. Two matching rules are reported because they differ for movable feasts:

  by_date    same calendar month-day in prior years (literal "misma fecha")
  by_holiday same holiday name in prior years (follows Easter as it moves)

Scored exactly like the model: MAPE over the 24 holiday hours (00:00-23:00).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("/home/uriel/GIT/analog_holidays")
EXP = ROOT / "experiments"
DEMAND = ROOT / "holidays" / "holiday_demand_mx.csv"
SELECTOR = ROOT / "holidays" / "holiday_selector_features.csv"
OUT = Path(__file__).resolve().parents[1] / "docs"


def champion() -> pd.DataFrame:
    ds = [d for d in sorted(glob.glob(str(EXP / "experiment_*criterion_holiday_identity")))
          if "_kcap" not in d]
    for x in reversed(ds):
        df = pd.read_csv(Path(x) / "metrics.csv")
        if "mape_24_exante_drop_pct" in df.columns:
            df["target_date"] = pd.to_datetime(df["target_date"])
            df["_run_dir"] = Path(x).name
            return df
    raise SystemExit("champion run not found")


def hourly_matrix(col: str) -> pd.DataFrame:
    d = pd.read_csv(DEMAND, parse_dates=["ds"])
    d["date"] = d.ds.dt.normalize()
    d["h"] = d.ds.dt.hour
    w = d.pivot_table(index="date", columns="h", values=col)
    return w.dropna()


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    return float(np.nanmean(np.abs(actual - pred) / denom * 100.0))


def mpe(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    return float(np.nanmean((actual - pred) / denom * 100.0))


def main() -> None:
    ch = champion()
    sel = pd.read_csv(SELECTOR, parse_dates=["date"])
    sel["date"] = sel["date"].dt.normalize()
    # map date -> anchor holiday name (same for all regions)
    anchor = (sel.dropna(subset=["anchor_holiday_name"])
                 .drop_duplicates(subset=["date"])
                 .set_index("date")["anchor_holiday_name"].to_dict())

    rows = []
    for uid, g in ch.groupby("unique_id"):
        W = hourly_matrix(uid)
        idx = pd.DatetimeIndex(W.index)
        for _, r in g.iterrows():
            tdate = pd.Timestamp(r.target_date).normalize()
            if tdate not in W.index:
                continue
            actual = W.loc[tdate].to_numpy(dtype=np.float64)

            prior_date = [d for d in idx if d < tdate
                          and d.month == tdate.month and d.day == tdate.day]
            tgt_anchor = anchor.get(tdate)
            prior_hol = [d for d in idx if d < tdate and anchor.get(d) == tgt_anchor] if tgt_anchor else []

            rec = {"unique_id": uid, "target_date": tdate.strftime("%Y-%m-%d"),
                   "holiday_label": r.holiday_label,
                   "model_mape_24_pct": r.mape_24_pct,
                   "model_mpe_24_pct": r.mpe_24_pct,
                   "n_prior_by_date": len(prior_date), "n_prior_by_holiday": len(prior_hol)}

            for tag, prior in (("by_date", prior_date), ("by_holiday", prior_hol)):
                if not prior:
                    continue
                P = W.loc[prior].to_numpy(dtype=np.float64)
                rec[f"naive_{tag}_mean_mape"] = mape(actual, P.mean(axis=0))
                rec[f"naive_{tag}_median_mape"] = mape(actual, np.median(P, axis=0))
                rec[f"naive_{tag}_mean_mpe"] = mpe(actual, P.mean(axis=0))
                rec[f"naive_{tag}_median_mpe"] = mpe(actual, np.median(P, axis=0))
                rec[f"naive_{tag}_last_mape"] = mape(actual, P[-1])
                # last-k variants
                for k in (2, 3, 4):
                    if len(prior) >= k:
                        rec[f"naive_{tag}_mean{k}_mape"] = mape(actual, P[-k:].mean(axis=0))
                        rec[f"naive_{tag}_median{k}_mape"] = mape(actual, np.median(P[-k:], axis=0))
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seasonal_naive_ceiling.csv", index=False)
    print(f"celdas: {len(df)}  (panel del campeon: {len(ch)})")
    print(f"instancias previas disponibles -> by_date: med={df.n_prior_by_date.median():.0f} "
          f"min={df.n_prior_by_date.min()} max={df.n_prior_by_date.max()} | "
          f"by_holiday: med={df.n_prior_by_holiday.median():.0f} "
          f"min={df.n_prior_by_holiday.min()} max={df.n_prior_by_holiday.max()}")

    print("\n=== MEDIANA MAPE_24 (152 celdas) ===")
    cols = [c for c in df.columns if c.endswith("_mape")] + ["model_mape_24_pct"]
    summ = df[cols].median().sort_values()
    for c, v in summ.items():
        print(f"  {c:36s} {v:6.3f}%   (n={df[c].notna().sum()})")

    print("\n=== PAREADO: modelo campeon vs cada naive ===")
    for c in [c for c in df.columns if c.endswith("_mape")]:
        m = df.dropna(subset=["model_mape_24_pct", c])
        if len(m) < 20:
            continue
        _, p = wilcoxon(m.model_mape_24_pct, m[c])
        d = m.model_mape_24_pct - m[c]
        print(f"  vs {c:34s} n={len(m):3d} | naive={m[c].median():6.3f}% | "
              f"modelo gana {100*(d<0).mean():3.0f}% | delta={d.median():+6.3f} pp | "
              f"skill={1-m.model_mape_24_pct.median()/m[c].median():+.3f} | p={p:.5f}")

    print("\n=== POR REGION: modelo vs naive(by_holiday, mediana) ===")
    c = "naive_by_holiday_median_mape"
    piv = df.groupby("unique_id").agg(modelo=("model_mape_24_pct", "median"), naive=(c, "median"))
    piv["skill"] = 1 - piv.modelo / piv.naive
    print(piv.round(3).sort_values("skill", ascending=False).to_string())

    print("\n=== POR FESTIVO ===")
    pivh = df.groupby("holiday_label").agg(modelo=("model_mape_24_pct", "median"), naive=(c, "median"))
    pivh["skill"] = 1 - pivh.modelo / pivh.naive
    print(pivh.round(3).sort_values("skill", ascending=False).to_string())

    print("\nCSV -> " + str(OUT / "seasonal_naive_ceiling.csv"))


if __name__ == "__main__":
    main()
