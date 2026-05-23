"""Shared dataset configuration for holiday notebook and audit workflows.

Change ``ACTIVE_DATASET`` to switch the whole workflow between configured
datasets without editing paths or region lists in multiple files.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    demand_path: Path
    notebook_holidays_path: Path
    audit_holidays_path: Path
    region_prefix: str = ""
    default_regions: tuple[str, ...] = ()


ACTIVE_DATASET = "mx"


_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mx": DatasetConfig(
        key="mx",
        display_name="Mexico",
        demand_path=PROJ_ROOT / "data" / "demand_mx.csv",
        notebook_holidays_path=PROJ_ROOT / "analog_holidays" / "holidays_recognized.json",
        audit_holidays_path=PROJ_ROOT / "data" / "holidays_mx.json",
        region_prefix="SEN_demand_",
        default_regions=(
            "SEN_demand_SIN",
            "SEN_demand_CEL",
            "SEN_demand_OCC",
            "SEN_demand_NOR",
            "SEN_demand_PEN",
            "SEN_demand_ORI",
            "SEN_demand_NES",
            "SEN_demand_NTE",
        ),
    ),
    "us": DatasetConfig(
        key="us",
        display_name="United States",
        demand_path=PROJ_ROOT / "data" / "demand_us.csv",
        notebook_holidays_path=PROJ_ROOT / "analog_holidays" / "holidays_us_recognized.json",
        audit_holidays_path=PROJ_ROOT / "data" / "holidays_us.json",
    ),
}


def get_dataset_config(dataset_key: str | None = None) -> DatasetConfig:
    key = (dataset_key or ACTIVE_DATASET).strip().lower()
    try:
        return _DATASET_CONFIGS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {key!r}. Valid options: {valid}") from exc


ACTIVE_CONFIG = get_dataset_config()


def _read_csv_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return []

    return [col for col in header if col and not col.startswith("Unnamed:")]


def list_dataset_regions(dataset_key: str | None = None) -> list[str]:
    config = get_dataset_config(dataset_key)
    columns = _read_csv_columns(config.demand_path)
    if columns:
        return [col for col in columns if col != "ds"]
    return list(config.default_regions)


def format_region_label(region: str, dataset_key: str | None = None) -> str:
    config = get_dataset_config(dataset_key)
    prefix = config.region_prefix
    if prefix and region.startswith(prefix):
        return region[len(prefix):]
    return region