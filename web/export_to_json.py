"""
export_to_json.py
=================
Convert audit_cache.parquet + audit_labels.parquet into static JSON
for consumption by the React SPA in web/.

Usage (from the repository root):
    python web/export_to_json.py

Output:
    web/public/data/audit_data.json
    web/public/data/audit_labels.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

# Paths relative to the repository root
_ROOT      = Path(__file__).resolve().parents[1]
_DATA_DIR  = _ROOT / "audit" / "data"
_OUT_DIR   = _ROOT / "web" / "public" / "data"

CACHE_PATH  = _DATA_DIR / "audit_cache.parquet"
LABELS_PATH = _DATA_DIR / "audit_labels.parquet"
HOUR_COLS   = [f"h_{h:02d}" for h in range(24)]


def _clean(val):
    """Convert NaN / NaT / numpy types to JSON-safe Python types."""
    if val is None:
        return None
    try:
        if math.isnan(float(val)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "isoformat"):
        return val.isoformat()
    if hasattr(val, "item"):          # numpy scalar → Python
        return val.item()
    return val


def export():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cache
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            "Run: python audit/build_cache.py"
        )
    df = pd.read_parquet(CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # Merge persisted labels
    if LABELS_PATH.exists():
        df_lbl = pd.read_parquet(LABELS_PATH)
        df_lbl["date"] = pd.to_datetime(df_lbl["date"])
        lookup = df_lbl.set_index(["unique_id", "date"])["label"].to_dict()
        df["label"] = df.apply(
            lambda r: lookup.get((r["unique_id"], r["date"]), r["label"]), axis=1
        )

    # Serialize rows
    records = []
    for _, row in df.iterrows():
        rec = {
            "unique_id":     str(row["unique_id"]),
            "date":          row["date"].strftime("%Y-%m-%d"),
            "label":         str(row.get("label", "normal_day")),
            "holiday_name":  _clean(row.get("holiday_name")),
            "holiday_type":  _clean(row.get("holiday_type")),
            "is_declared_holiday": bool(row.get("is_declared_holiday", False)),
            "is_outlier":    bool(row.get("is_outlier", False)),
            "outlier_score": _clean(row.get("outlier_score")),
            "year":          _clean(row.get("year")),
            "month":         _clean(row.get("month")),
            "dow":           _clean(row.get("dow")),
            "dow_name":      _clean(row.get("dow_name")),
            "event_description": _clean(row.get("event_description")),
            "hours":         [_clean(row.get(h)) for h in HOUR_COLS],
        }
        records.append(rec)

    out_path = _OUT_DIR / "audit_data.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] audit_data.json  →  {len(records):,} rows  →  {out_path}")

    # Export labels separately for save/load flows
    labels_out = []
    for _, row in df[["unique_id", "date", "label"]].iterrows():
        labels_out.append({
            "unique_id": str(row["unique_id"]),
            "date":      row["date"].strftime("%Y-%m-%d"),
            "label":     str(row["label"]),
        })
    lbl_path = _OUT_DIR / "audit_labels.json"
    lbl_path.write_text(json.dumps(labels_out, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] audit_labels.json  →  {len(labels_out):,} rows  →  {lbl_path}")


def cli() -> None:
    """Run the JSON export from the script entrypoint."""
    export()


if __name__ == "__main__":
    cli()
