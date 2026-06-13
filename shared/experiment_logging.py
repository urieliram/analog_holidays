"""Persist each analog-holiday run as a self-contained, comparable experiment folder.

Every meaningful run should be registered under ``experiments/`` so results stay reproducible,
traceable, and comparable across experiments. ``save_experiment_run`` creates one timestamped
folder per run:

    experiments/experiment_<YYYY_MM_DD_HH_MM>[_<slug>]/
        manifest.yaml   # conditions + provenance (the reproducibility contract)
        metrics.csv     # one row per (unique_id, target_date) -- concatenated batch results
        summary.csv     # aggregated metrics per series (+ an ALL row)
        plots/*.png      # figures (PNG; PDF/PKL/PARQUET are git-ignored)
        notes.md         # analyst log, seeded with the hypothesis prompts

The folder name uses the same ``%Y_%m_%d_%H_%M`` stamp convention as the rest of the package
(see ``analog/P_analog_pre_holidays.py``). See ``experiments/README.md`` for the full convention.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
EXPERIMENT_DIR_PREFIX = "experiment_"
EXPERIMENT_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M"
PRIMARY_METRIC_DEFAULT = "mape_24h_pct"
_SUMMARY_METRIC_CANDIDATES = (
    PRIMARY_METRIC_DEFAULT,
    "mape_window_pct",
    "mape_24_pct",
    "mae_24h",
    "mae_window",
    "bias_24",
    "bias_38",
)


def _run_git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(PACKAGE_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _safe_name(value: object) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "figure"


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pandas/Path objects into JSON/YAML-serializable Python primitives."""
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if value is pd.NA:
            return None
    except Exception:
        pass
    return value


def build_experiment_dir_name(
    timestamp: Optional[datetime] = None,
    slug: Optional[str] = None,
) -> str:
    """Return ``experiment_<YYYY_MM_DD_HH_MM>`` (optionally suffixed with a slug)."""
    stamp = (timestamp or datetime.now()).strftime(EXPERIMENT_TIMESTAMP_FORMAT)
    name = f"{EXPERIMENT_DIR_PREFIX}{stamp}"
    if slug:
        name = f"{name}_{_safe_name(slug)}"
    return name


def _resolve_unique_experiment_dir(base_dir: Path, dir_name: str) -> Path:
    candidate = base_dir / dir_name
    if not candidate.exists():
        return candidate
    # Minute-resolution names can collide on rapid reruns; never overwrite a past run.
    for suffix in range(2, 100):
        alt = base_dir / f"{dir_name}_{suffix:02d}"
        if not alt.exists():
            return alt
    raise RuntimeError(f"Could not find a free experiment directory for {dir_name!r}.")


def _extract_results_df(batch_result: Any) -> Optional[pd.DataFrame]:
    df = getattr(batch_result, "results_df", batch_result)
    if isinstance(df, pd.DataFrame):
        return df
    return None


def _consolidate_metrics(
    batch_results: Mapping[str, Any],
    experiment_id: str,
    selector_features_path: Path | str | None = None,
) -> pd.DataFrame:
    frames = []
    for unique_id, batch_result in batch_results.items():
        df = _extract_results_df(batch_result)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["unique_id"] = str(unique_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["experiment_id", "unique_id"])

    metrics = pd.concat(frames, ignore_index=True)
    metrics.insert(0, "experiment_id", experiment_id)

    # Move unique_id next to experiment_id for readability.
    cols = ["experiment_id", "unique_id"] + [
        c for c in metrics.columns if c not in ("experiment_id", "unique_id")
    ]
    metrics = metrics[cols]

    metrics = _attach_holiday_day_type(metrics, selector_features_path)
    return metrics


def _attach_holiday_day_type(
    metrics: pd.DataFrame,
    selector_features_path: Path | str | None,
) -> pd.DataFrame:
    if selector_features_path is None or "holiday_day_type" in metrics.columns:
        return metrics
    path = Path(selector_features_path)
    if not path.exists():
        return metrics
    try:
        selector = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return metrics
    needed = {"unique_id", "date", "holiday_day_type"}
    if not needed.issubset(selector.columns):
        return metrics

    lookup = selector[["unique_id", "date", "holiday_day_type"]].copy()
    lookup["unique_id"] = lookup["unique_id"].astype(str)
    lookup["date"] = pd.to_datetime(lookup["date"]).dt.normalize()

    merged = metrics.copy()
    merged["_date_key"] = pd.to_datetime(merged["target_date"], errors="coerce").dt.normalize()
    merged = merged.merge(
        lookup.rename(columns={"date": "_date_key"}),
        on=["unique_id", "_date_key"],
        how="left",
    ).drop(columns="_date_key")

    # Place holiday_day_type right after holiday_label when present.
    if "holiday_day_type" in merged.columns and "holiday_label" in merged.columns:
        cols = list(merged.columns)
        cols.remove("holiday_day_type")
        insert_at = cols.index("holiday_label") + 1
        cols.insert(insert_at, "holiday_day_type")
        merged = merged[cols]
    return merged


def _build_summary(metrics_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        c
        for c in dict.fromkeys((primary_metric, *_SUMMARY_METRIC_CANDIDATES))
        if c in metrics_df.columns
    ]

    def _summarize(group: pd.DataFrame, label: str) -> dict:
        fail = (
            group["fail"].fillna(False).astype(bool)
            if "fail" in group.columns
            else pd.Series(False, index=group.index)
        )
        ok = group[~fail]
        row: dict[str, Any] = {
            "unique_id": label,
            "n_targets": int(len(group)),
            "n_fail": int(fail.sum()),
            "fail_rate": round(float(fail.mean()), 4) if len(group) else np.nan,
        }
        for metric in metric_cols:
            vals = pd.to_numeric(ok[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
        return row

    rows = [_summarize(g, str(uid)) for uid, g in metrics_df.groupby("unique_id")]
    rows.append(_summarize(metrics_df, "ALL"))
    return pd.DataFrame(rows)


def _save_figures(figures: Mapping[str, Any], plots_dir: Path, dpi: int = 150) -> list[str]:
    saved: list[str] = []
    seen: dict[str, int] = {}
    for name, fig in figures.items():
        if fig is None or not hasattr(fig, "savefig"):
            continue
        stem = _safe_name(name)
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] > 1:
            stem = f"{stem}_{seen[stem]}"
        out_path = plots_dir / f"{stem}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved.append(out_path.name)
    return saved


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    serializable = _to_serializable(manifest)
    try:
        import yaml  # type: ignore

        manifest_path.write_text(
            yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception:
        manifest_path.with_suffix(".json").write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


_NOTES_TEMPLATE = """# Notes -- {experiment_id}

## Hypothesis
<!-- What question did this run test? -->

## Setup deltas vs baseline
<!-- Which axis changed relative to the comparison experiment? -->

## Observations
<!-- What metrics.csv / summary.csv / plots show. Cite the unique_id + target_date per number;
     prefer medians and report n. -->

## Conclusion
<!-- Did the hypothesis hold? For which series / day types / clusters? -->

## Recommended next experiments
1.
2.
3.
"""


def save_experiment_run(
    config: Mapping[str, Any],
    batch_results: Mapping[str, Any],
    *,
    figures: Optional[Mapping[str, Any]] = None,
    summary_df: Optional[pd.DataFrame] = None,
    notes: Optional[str] = None,
    slug: Optional[str] = None,
    base_dir: Path | str = EXPERIMENTS_DIR,
    timestamp: Optional[datetime] = None,
    selector_features_path: Path | str | None = None,
    primary_metric: str = PRIMARY_METRIC_DEFAULT,
    verbose: bool = True,
) -> Path:
    """Register one analog-holiday run as ``experiments/experiment_<timestamp>/``.

    Args:
        config: the experiment conditions (every mixed knob: cluster criterion, analog
            selection, regression, window, scope, tuning). Stored verbatim in ``manifest.yaml``.
        batch_results: mapping ``{unique_id: AnalogHolidayBatchResult}`` (or
            ``{unique_id: results_df}``). Their ``results_df`` frames are concatenated into
            ``metrics.csv`` with ``experiment_id`` + ``unique_id`` columns.
        figures: optional ``{name: matplotlib Figure}`` saved as ``plots/<name>.png``.
        summary_df: optional precomputed summary; otherwise aggregated from the metrics.
        notes: optional seed text for ``notes.md`` (defaults to the analyst-log template).
        slug: optional short label appended to the folder name.
        selector_features_path: optional selector CSV to attach ``holiday_day_type`` per row.

    Returns:
        The created experiment directory path.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    dir_name = build_experiment_dir_name(timestamp=timestamp, slug=slug)
    experiment_dir = _resolve_unique_experiment_dir(base, dir_name)
    experiment_dir.mkdir(parents=True)
    experiment_id = experiment_dir.name

    metrics_df = _consolidate_metrics(batch_results, experiment_id, selector_features_path)
    metrics_path = experiment_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if summary_df is None:
        summary_df = _build_summary(metrics_df, primary_metric)
    summary_path = experiment_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    saved_plots: list[str] = []
    if figures:
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        saved_plots = _save_figures(figures, plots_dir)

    target_dates = (
        sorted({str(d) for d in metrics_df["target_date"].dropna().tolist()})
        if "target_date" in metrics_df.columns
        else []
    )
    manifest = {
        "id": experiment_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "provenance": {
            "git_commit": _run_git("rev-parse", "HEAD"),
            "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_run_git("status", "--porcelain")),
        },
        "scope": {
            "unique_ids": [str(u) for u in batch_results.keys()],
            "n_series": len(batch_results),
            "n_metric_rows": int(len(metrics_df)),
            "target_dates": target_dates,
            "primary_metric": primary_metric,
        },
        "config": dict(config),
        "outputs": {
            "metrics_csv": metrics_path.name,
            "summary_csv": summary_path.name,
            "plots": saved_plots,
            "notes": "notes.md",
        },
    }
    _write_manifest(experiment_dir / "manifest.yaml", manifest)

    notes_text = notes if notes is not None else _NOTES_TEMPLATE.format(experiment_id=experiment_id)
    (experiment_dir / "notes.md").write_text(notes_text, encoding="utf-8")

    if verbose:
        print(f"Registered experiment: {experiment_dir}")
        print(f"  metrics.csv rows : {len(metrics_df)} ({len(batch_results)} series)")
        print(f"  plots saved      : {len(saved_plots)}")
    return experiment_dir
