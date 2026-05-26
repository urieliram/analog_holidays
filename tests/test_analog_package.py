from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
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

    def test_date_cluster_formatter_appends_analog_cluster_when_available(self) -> None:
        from analog_holidays.shared.identify_holidays import (
            _build_date_value_lookup,
            _format_date_with_lookup,
        )

        lookup = _build_date_value_lookup(
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-12-25"), pd.Timestamp("2024-12-31")],
                    "analog_cluster": ["F", pd.NA],
                }
            ),
            "analog_cluster",
        )

        self.assertEqual(
            _format_date_with_lookup(pd.Timestamp("2024-12-25"), lookup),
            "25/12/24 [F]",
        )
        self.assertEqual(
            _format_date_with_lookup(pd.Timestamp("2024-12-31"), lookup),
            "31/12/24",
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

    def test_extract_hour_window_spans_preholiday_and_holiday(self) -> None:
        from analog_holidays.analog.analog_holidays import _extract_hour_window

        df_region = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-12-24", "2024-12-25"]),
                **{f"h_{hour:02d}": [float(hour), float(100 + hour)] for hour in range(24)},
            }
        )

        window = _extract_hour_window(
            df_region=df_region,
            window_start=pd.Timestamp("2024-12-24 10:00:00"),
            length_hours=38,
        )

        expected = np.asarray(
            list(range(10, 24)) + list(range(100, 124)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(window, expected)

    def test_run_analog_holidays_supports_offset_forecast_window(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        hour_columns = {
            f"h_{hour:02d}": [float(day_idx * 100 + hour) for day_idx in range(6)]
            for hour in range(24)
        }
        df_source = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 6,
                "date": pd.date_range("2024-12-20", periods=6, freq="D"),
                "label": [
                    "normal_day",
                    "normal_day",
                    "normal_day",
                    "holiday",
                    "normal_day",
                    "holiday",
                ],
                "holiday_name": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "Historic Holiday",
                    pd.NA,
                    "Target Holiday",
                ],
                "holiday_type": [pd.NA] * 6,
                "is_declared_holiday": [False, False, False, True, False, True],
                "is_outlier": [False] * 6,
                "outlier_score": [np.nan] * 6,
                **hour_columns,
            }
        )

        class DummyAnalogSpecialDays:
            def __init__(self, *args, **kwargs) -> None:
                self.kwargs = kwargs

            def fit(self, y, special_days):
                self.y = np.asarray(y, dtype=np.float64)
                self.special_days = np.asarray(special_days, dtype=np.float64)
                return self

            def predict(self, h, level):
                result = {"mean": np.full(h, 999.0, dtype=np.float64)}
                for lv in level:
                    result[f"lo-{lv}"] = np.full(h, 990.0, dtype=np.float64)
                    result[f"hi-{lv}"] = np.full(h, 1010.0, dtype=np.float64)
                return result

        def fake_core(**kwargs):
            vsele = int(kwargs["vsele"])
            return (
                np.full(vsele, 999.0, dtype=np.float64),
                0.01,
                0.02,
                False,
                [20],
                np.full((1, vsele), 777.0, dtype=np.float64),
            )

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "AnalogSpecialDays",
            DummyAnalogSpecialDays,
        ), patch.object(
            analog_holidays_module,
            "analog_special_days_core",
            side_effect=fake_core,
        ):
            run = analog_holidays_module.run_analog_holidays(
                unique_id="SEN_demand_SIN",
                target_date="2024-12-25",
                season_length=38,
                forecast_start_offset_hours=14,
                levels=[80, 95],
                special_labels=("holiday",),
                min_special_points=24,
            )

        self.assertEqual(run.forecast_start, pd.Timestamp("2024-12-24 10:00:00"))
        self.assertEqual(run.forecast_end, pd.Timestamp("2024-12-26 00:00:00"))
        self.assertEqual(run.forecast_start_offset_hours, 14)
        self.assertEqual(len(run.hourly_series), 106)
        self.assertEqual(len(run.previous_day_profile), 38)
        self.assertEqual(len(run.actual_profile), 38)
        expected_actual = np.asarray(
            list(range(410, 424)) + list(range(500, 524)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(run.actual_profile, expected_actual)
        self.assertEqual(
            run.selected_days_df["special_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-12-23"],
        )

    def test_run_analog_holidays_passes_regressor_params(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        hour_columns = {
            f"h_{hour:02d}": [float(day_idx * 100 + hour) for day_idx in range(4)]
            for hour in range(24)
        }
        df_source = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 4,
                "date": pd.date_range("2024-12-21", periods=4, freq="D"),
                "label": ["normal_day", "holiday", "normal_day", "holiday"],
                "holiday_name": [pd.NA, "Historic Holiday", pd.NA, "Target Holiday"],
                "holiday_type": [pd.NA] * 4,
                "is_declared_holiday": [False, True, False, True],
                "is_outlier": [False] * 4,
                "outlier_score": [np.nan] * 4,
                **hour_columns,
            }
        )

        init_kwargs = {}
        core_kwargs = {}

        class DummyAnalogSpecialDays:
            def __init__(self, *args, **kwargs) -> None:
                init_kwargs.update(kwargs)

            def fit(self, y, special_days):
                return self

            def predict(self, h, level):
                result = {"mean": np.full(h, 1.0, dtype=np.float64)}
                for lv in level:
                    result[f"lo-{lv}"] = np.full(h, 0.0, dtype=np.float64)
                    result[f"hi-{lv}"] = np.full(h, 2.0, dtype=np.float64)
                return result

        def fake_core(**kwargs):
            core_kwargs.update(kwargs)
            vsele = int(kwargs["vsele"])
            return (
                np.full(vsele, 1.0, dtype=np.float64),
                0.01,
                0.02,
                False,
                [0],
                np.full((1, vsele), 2.0, dtype=np.float64),
            )

        regressor_params = {"n_estimators": 250, "max_depth": 8, "min_samples_leaf": 2}

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "AnalogSpecialDays",
            DummyAnalogSpecialDays,
        ), patch.object(
            analog_holidays_module,
            "analog_special_days_core",
            side_effect=fake_core,
        ):
            analog_holidays_module.run_analog_holidays(
                unique_id="SEN_demand_SIN",
                target_date="2024-12-24",
                season_length=24,
                typedist="pearson",
                typereg="RF",
                regressor_params=regressor_params,
                special_labels=("holiday",),
            )

        self.assertEqual(init_kwargs["regressor_params"], regressor_params)
        self.assertEqual(core_kwargs["regressor_params"], regressor_params)

    def test_tune_analog_holidays_optuna_returns_rf_regressor_params(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        hour_columns = {
            f"h_{hour:02d}": [float(day_idx * 100 + hour) for day_idx in range(5)]
            for hour in range(24)
        }
        df_source = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 5,
                "date": pd.date_range("2023-12-20", periods=5, freq="D"),
                "label": ["holiday"] * 5,
                "holiday_name": [f"Holiday {idx}" for idx in range(5)],
                "holiday_type": [pd.NA] * 5,
                "is_declared_holiday": [True] * 5,
                "is_outlier": [False] * 5,
                "outlier_score": [np.nan] * 5,
                **hour_columns,
            }
        )

        def fake_special_mask(*args, **kwargs):
            return pd.Series([True] * len(df_source), index=df_source.index)

        def fake_evaluate_fold(**kwargs):
            return {
                "target_date": pd.Timestamp(kwargs["target_date"]).date().isoformat(),
                "mae": 1.0,
                "mape_pct": 2.0,
                "selected_analogs": 3,
                "fail": False,
                "train_days": 3,
                "train_special_days": 3,
            }

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "build_special_day_daily_mask",
            side_effect=fake_special_mask,
        ), patch.object(
            analog_holidays_module,
            "_evaluate_analog_holiday_fold",
            side_effect=fake_evaluate_fold,
        ):
            result = analog_holidays_module.tune_analog_holidays_optuna(
                unique_id="SEN_demand_SIN",
                train_end="2023-12-25",
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                typereg_choices=["RF"],
                random_seed=7,
                special_labels=("holiday",),
            )

        self.assertEqual(result.best_config["typereg"], "RF")
        self.assertEqual(set(result.best_config["regressor_params"].keys()), {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
        })
        self.assertIn(result.best_config["regressor_params"]["n_estimators"], {100, 200, 300})
        self.assertIn(result.best_config["regressor_params"]["max_depth"], {4, 8, 12})
        self.assertIn(result.best_config["regressor_params"]["min_samples_leaf"], {1, 2, 4})

    def test_tune_analog_holidays_optuna_caps_k_to_realizable_analog_pool(self) -> None:
        import optuna
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        hour_columns = {
            f"h_{hour:02d}": [float(day_idx * 100 + hour) for day_idx in range(5)]
            for hour in range(24)
        }
        df_source = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 5,
                "date": pd.date_range("2023-12-20", periods=5, freq="D"),
                "label": ["holiday"] * 5,
                "holiday_name": [f"Holiday {idx}" for idx in range(5)],
                "holiday_type": [pd.NA] * 5,
                "is_declared_holiday": [True] * 5,
                "is_outlier": [False] * 5,
                "outlier_score": [np.nan] * 5,
                **hour_columns,
            }
        )

        def fake_special_mask(df, *args, **kwargs):
            return pd.Series([True] * len(df), index=df.index)

        def fake_evaluate_fold(**kwargs):
            return {
                "target_date": pd.Timestamp(kwargs["target_date"]).date().isoformat(),
                "mae": 1.0,
                "mape_pct": 2.0,
                "selected_analogs": 4,
                "fail": False,
                "train_days": 3,
                "train_special_days": 3,
            }

        def fake_count_realizable_analog_positions(**kwargs):
            return 4

        k_bounds: list[tuple[int, int]] = []
        original_suggest_int = optuna.trial._trial.Trial.suggest_int

        def recording_suggest_int(self, name, low, high, *args, **kwargs):
            if name == "k":
                k_bounds.append((int(low), int(high)))
            return original_suggest_int(self, name, low, high, *args, **kwargs)

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "build_special_day_daily_mask",
            side_effect=fake_special_mask,
        ), patch.object(
            analog_holidays_module,
            "_evaluate_analog_holiday_fold",
            side_effect=fake_evaluate_fold,
        ), patch.object(
            analog_holidays_module,
            "_count_realizable_analog_positions",
            side_effect=fake_count_realizable_analog_positions,
        ), patch.object(
            optuna.trial._trial.Trial,
            "suggest_int",
            new=recording_suggest_int,
        ):
            result = analog_holidays_module.tune_analog_holidays_optuna(
                unique_id="SEN_demand_SIN",
                train_end="2023-12-25",
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                typereg_choices=["PCR"],
                random_seed=7,
                special_labels=("holiday",),
            )

        self.assertTrue(k_bounds)
        self.assertEqual(k_bounds[0], (3, 4))
        self.assertLessEqual(result.best_config["k"], 4)


if __name__ == "__main__":
    unittest.main()