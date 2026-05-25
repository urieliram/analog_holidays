from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = REPO_ROOT.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


class AnalogPackageSmokeTests(unittest.TestCase):
    def test_analog_package_exports_core_models(self) -> None:
        import analog_holidays.analog as analog_pkg

        self.assertTrue(hasattr(analog_pkg, "AnalogKNN"))
        self.assertTrue(hasattr(analog_pkg, "AnalogSpecialDays"))
        self.assertTrue(hasattr(analog_pkg, "_REGRESSORS"))

    def test_analog_holidays_module_uses_repo_audit_data_path(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        self.assertEqual(
            analog_holidays_module.DEFAULT_SOURCE_PATH,
            REPO_ROOT / "audit" / "data",
        )

    def test_list_dataset_regions_excludes_helper_columns(self) -> None:
        from analog_holidays.shared.dataset_config import list_dataset_regions

        regions = list_dataset_regions("mx")

        self.assertIn("SEN_demand_SIN", regions)
        self.assertNotIn("SEN_demand_SIN_holiday", regions)
        self.assertNotIn("SIN_cluster", regions)

    def test_load_selector_cluster_lookup_and_pre_holiday_mask_filter_by_cluster(self) -> None:
        from analog_holidays.shared.identify_holidays import load_selector_cluster_lookup
        from analog_holidays.analog.P_analog_pre_holidays import _build_pre_holiday_mask

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selector_path = tmp_path / "holiday_selector_features.csv"
            selector_path.write_text(
                "date,analog_cluster\n"
                "2020-02-03,F\n"
                "2020-03-16,G\n",
                encoding="utf-8",
            )

            cluster_lookup = load_selector_cluster_lookup(selector_path)
            self.assertEqual(cluster_lookup[pd.Timestamp("2020-02-03")], "F")
            self.assertEqual(cluster_lookup[pd.Timestamp("2020-03-16")], "G")

            df_hist = pd.DataFrame(
                {
                    "ds": pd.to_datetime(
                        [
                            "2020-02-02 22:00:00",
                            "2020-02-02 23:00:00",
                            "2020-02-03 00:00:00",
                            "2020-02-03 01:00:00",
                            "2020-03-15 22:00:00",
                            "2020-03-15 23:00:00",
                            "2020-03-16 00:00:00",
                            "2020-03-16 01:00:00",
                        ]
                    ),
                    "SEN_demand_SIN_holiday": [0, 0, 1, 1, 0, 0, 1, 1],
                }
            )

            mask = _build_pre_holiday_mask(
                df_hist=df_hist,
                unique_id="SEN_demand_SIN",
                previously_w_hours=2,
                target_date="2020-03-16",
                selector_cluster_lookup=cluster_lookup,
                match_target_cluster=True,
            )

            self.assertEqual(mask.tolist(), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])

    def test_filter_dates_with_selector_cluster_skips_missing_dates(self) -> None:
        from analog_holidays.analog.P_analog_pre_holidays import _filter_dates_with_selector_cluster

        kept_dates, skipped_dates = _filter_dates_with_selector_cluster(
            dates=[
                pd.Timestamp("2023-12-12 00:00:00"),
                pd.Timestamp("2023-12-24 00:00:00"),
                pd.Timestamp("2023-12-31 00:00:00"),
            ],
            selector_cluster_lookup={
                pd.Timestamp("2023-12-24"): "F",
                pd.Timestamp("2023-12-31"): "G",
            },
        )

        self.assertEqual(
            [date_value.strftime("%Y-%m-%d") for date_value in kept_dates],
            ["2023-12-24", "2023-12-31"],
        )
        self.assertEqual(
            [date_value.strftime("%Y-%m-%d") for date_value in skipped_dates],
            ["2023-12-12"],
        )

    def test_pre_holiday_helpers_ignore_cluster_columns(self) -> None:
        from analog_holidays.analog.P_analog_pre_holidays import list_unique_ids
        from analog_holidays.analog.analog_holidays import load_audit_source

        df_source = pd.DataFrame(
            {
                "ds": ["2020-02-03 00:00:00", "2020-02-03 01:00:00"],
                "SEN_demand_SIN": [1.0, 2.0],
                "SEN_demand_SIN_holiday": [1, 1],
                "SIN_cluster": ["F", "F"],
            }
        )
        self.assertEqual(list_unique_ids(df_source), ["SEN_demand_SIN"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "holiday_demand_mx.csv"
            df_source.to_csv(source_path, index=False)

            df_daily = load_audit_source(source_path)
            self.assertEqual(df_daily["unique_id"].unique().tolist(), ["SEN_demand_SIN"])

    def test_pre_holiday_source_rejects_hourly_analog_cluster_csv(self) -> None:
        from analog_holidays.analog.P_analog_pre_holidays import load_pre_holiday_source

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "holiday_demand_mx_analog_cluster.csv"
            pd.DataFrame(
                {
                    "ds": ["2020-02-03 00:00:00"],
                    "SEN_demand_SIN": [1.0],
                    "SEN_demand_SIN_holiday": [1],
                    "SIN_cluster": ["F"],
                }
            ).to_csv(source_path, index=False)

            with self.assertRaisesRegex(ValueError, "no longer a valid historical demand source"):
                load_pre_holiday_source(source_path)

    def test_audit_source_rejects_hourly_analog_cluster_csv(self) -> None:
        from analog_holidays.analog.analog_holidays import load_audit_source

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "holiday_demand_mx_analog_cluster.csv"
            pd.DataFrame(
                {
                    "ds": ["2020-02-03 00:00:00"],
                    "SEN_demand_SIN": [1.0],
                    "SEN_demand_SIN_holiday": [1],
                    "SIN_cluster": ["F"],
                }
            ).to_csv(source_path, index=False)

            with self.assertRaisesRegex(ValueError, "no longer a valid historical demand source"):
                load_audit_source(source_path)

    def test_build_future_holiday_selector_features_feeds_selector_analog_cluster(self) -> None:
        from analog_holidays.shared.identify_holidays import (
            assign_holiday_selector_analog_clusters,
            build_future_holiday_selector_features,
            build_holiday_selector_priors,
        )

        df_selector_history = pd.DataFrame(
            {
                "holiday_name": ["Labor Day"],
                "anchor_holiday_name": ["Labor Day"],
                "date": [pd.Timestamp("2024-05-01")],
                "holiday_day_type": ["H2"],
                "weekday_name": ["Wednesday"],
                "day_class_code": [1],
                "day_class_name": ["Weekday"],
                "season": ["Spring"],
                "date_rule": ["fixed_date"],
                "is_fixed_date": [True],
                "is_observed_monday_rule": [False],
                "best_matching_weekday": ["Sunday"],
                "daily_profile_cluster": ["A"],
                "daily_profile_cluster_id": [0],
                "daily_profile_archetype": ["Sunday-like"],
                "event_profile_cluster": ["C"],
                "event_profile_cluster_id": [0],
            }
        )
        df_priors = build_holiday_selector_priors(df_selector_history)

        df_future = build_future_holiday_selector_features(
            df_holidays=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-05-01"), pd.Timestamp("2025-05-01")],
                    "holiday_name": ["Labor Day", "Labor Day"],
                }
            ),
            df_priors=df_priors,
            available_dates={pd.Timestamp("2024-05-01"), pd.Timestamp("2025-05-01")},
            start_date="2025-01-01",
            end_date="2025-12-31",
        )

        self.assertEqual(df_future["date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-05-01"])
        self.assertEqual(df_future["event_profile_cluster"].tolist(), ["C"])
        self.assertEqual(df_future["daily_profile_cluster"].tolist(), ["A"])
        self.assertEqual(df_future["best_matching_weekday"].tolist(), ["Sunday"])

        df_selector_all = pd.concat([df_selector_history, df_future], ignore_index=True)
        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector_all,
            df_priors=df_priors,
            criterion="event_profile_cluster",
        )
        df_selector_clusters = cluster_results["df_selector_clusters"]

        self.assertEqual(
            df_selector_clusters.loc[
                df_selector_clusters["date"] == pd.Timestamp("2025-05-01"),
                "analog_cluster",
            ].tolist(),
            ["F"],
        )


if __name__ == "__main__":
    unittest.main()