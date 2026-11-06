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

    def test_load_selector_cluster_lookup_filters_mixed_series_exports_by_unique_id(self) -> None:
        from analog_holidays.shared.identify_holidays import load_selector_cluster_lookup

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selector_path = tmp_path / "holiday_selector_features.csv"
            selector_path.write_text(
                "unique_id,date,analog_cluster\n"
                "SEN_demand_SIN,2020-02-03,F\n"
                "OCC_demand_BAJ,2020-02-03,H\n"
                "SEN_demand_SIN,2020-03-16,G\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "multiple unique_id values"):
                load_selector_cluster_lookup(selector_path)

            cluster_lookup = load_selector_cluster_lookup(
                selector_path,
                unique_id="SEN_demand_SIN",
            )

            self.assertEqual(cluster_lookup[pd.Timestamp("2020-02-03")], "F")
            self.assertEqual(cluster_lookup[pd.Timestamp("2020-03-16")], "G")

    def test_resolve_selector_cluster_filter_label_prefers_exported_criterion(self) -> None:
        from analog_holidays.analog.analog_holidays import _resolve_selector_cluster_filter_label

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selector_path = tmp_path / "holiday_selector_features.csv"
            selector_path.write_text(
                "unique_id,date,analog_cluster,analog_cluster_criterion\n"
                "SEN_demand_SIN,2020-02-03,F,seasonal_heat_cold\n"
                "SEN_demand_SIN,2020-03-16,G,seasonal_heat_cold\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _resolve_selector_cluster_filter_label(
                    selector_path,
                    match_target_cluster=True,
                    unique_id="SEN_demand_SIN",
                ),
                "seasonal_heat_cold",
            )
            self.assertFalse(
                _resolve_selector_cluster_filter_label(
                    selector_path,
                    match_target_cluster=False,
                    unique_id="SEN_demand_SIN",
                )
            )

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

    def test_build_analog_cluster_38h_suptitle_includes_selection_criterion(self) -> None:
        from analog_holidays.shared.identify_holidays import _build_analog_cluster_38h_suptitle

        suptitle = _build_analog_cluster_38h_suptitle(
            ["F", "G", "H"],
            "SEN_demand_SIN",
            selection_criterion="best_matching_weekday",
        )

        self.assertEqual(
            suptitle,
            "38-h event profiles by analog cluster (F/G/H)  |  SEN_demand_SIN\n"
            "Selection criterion: best_matching_weekday",
        )

    def test_fit_hourly_bias_factor_model_uses_available_analogs_when_fewer_than_requested(self) -> None:
        from analog_holidays.analog.analog_holidays import fit_hourly_bias_factor_model

        neighbor_profiles = np.vstack(
            [
                np.arange(38, dtype=np.float64),
                np.arange(38, dtype=np.float64) + 10.0,
            ]
        )

        result = fit_hourly_bias_factor_model(
            neighbor_profiles=neighbor_profiles,
            window_hours=38,
            max_analogs=4,
        )

        self.assertEqual(result["requested_analogs"], 4)
        self.assertEqual(result["available_neighbor_profiles"], 2)
        self.assertEqual(result["selected_analogs"], 2)
        self.assertEqual(result["train_samples"], 2)
        self.assertEqual(result["window_hours"], 38)
        self.assertEqual(result["method"], "hourly_window_mean_factor_top_available_analogs")
        self.assertEqual(result["hourly_factors"].shape, (38,))

    def test_build_panel_config_row_formats_optuna_choice_and_pool(self) -> None:
        from analog_holidays.analog.analog_holidays import _build_panel_config_row

        row = pd.Series(
            {
                "k": 7,
                "optuna_k_max": 19,
                "scale_method": "standard",
                "typedist": "pearson",
                "typereg": "PCR",
                "n_components": 3,
                "filter_by_cluster": True,
            }
        )

        result = _build_panel_config_row(row)

        self.assertEqual(
            result,
            "k=7/19 | scale_method=standard |\n"
            "typedist=pearson |\n"
            "typereg=PCR | n_components=3\n"
            "cluster=True",
        )

    def test_build_panel_config_row_prefers_cluster_filter_label(self) -> None:
        from analog_holidays.analog.analog_holidays import _build_panel_config_row

        row = pd.Series(
            {
                "k": 7,
                "optuna_k_max": 19,
                "scale_method": "standard",
                "typedist": "pearson",
                "typereg": "PCR",
                "n_components": 3,
                "filter_by_cluster": True,
                "cluster_filter_label": "seasonal_heat_cold",
            }
        )

        result = _build_panel_config_row(row)

        self.assertEqual(
            result,
            "k=7/19 | scale_method=standard |\n"
            "typedist=pearson |\n"
            "typereg=PCR | n_components=3\n"
            "cluster=seasonal_heat_cold",
        )

    def test_plot_analog_pair_sequences_adds_recovery_day_actuals(self) -> None:
        import matplotlib.pyplot as plt
        from analog_holidays.analog.analog_holidays import AnalogHolidayRun, plot_analog_pair_sequences

        run = AnalogHolidayRun(
            unique_id="SEN_demand_SIN",
            target_date=pd.Timestamp("2025-05-01"),
            forecast_start=pd.Timestamp("2025-04-30 10:00:00"),
            forecast_end=pd.Timestamp("2025-05-02 00:00:00"),
            forecast_start_offset_hours=14,
            target_exists=True,
            target_has_complete_profile=True,
            typedist="pearson",
            typereg="PCR",
            scale_method=None,
            season_length=38,
            k=1,
            n_components=3,
            regressor_params={},
            levels=[80, 95],
            train_df=pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5, freq="D")}),
            target_row=None,
            special_day_daily_mask=pd.Series([True] * 5),
            hourly_series=np.arange(200, dtype=np.float64),
            special_day_hourly_mask=np.ones(200, dtype=np.float64),
            previous_day_profile=np.arange(38, dtype=np.float64),
            forecast_profile=np.arange(38, dtype=np.float64) + 100.0,
            interval_low={80: np.zeros(38, dtype=np.float64), 95: np.zeros(38, dtype=np.float64)},
            interval_high={80: np.ones(38, dtype=np.float64), 95: np.ones(38, dtype=np.float64)},
            actual_profile=np.arange(38, dtype=np.float64) + 200.0,
            positions=[10],
            neighbors2=np.array([np.arange(38, dtype=np.float64) + 50.0]),
            selected_days_df=pd.DataFrame({"special_date": [pd.Timestamp("2025-01-10")]}),
            fail=False,
            t_sel=0.1,
            t_reg=0.2,
            special_labels=("holiday",),
            include_declared_holidays=False,
            include_outliers=False,
            min_special_points=24,
            min_event_gap=24,
            max_events=None,
            recent_weekend_analogs=0,
            recent_weekend_like=None,
            recent_weekend_dates=[],
            label_column="label",
            post_holiday_actual_profile=np.arange(24, dtype=np.float64) + 300.0,
        )

        fig, ax = plot_analog_pair_sequences(run)

        labels = {line.get_label() for line in ax.get_lines()}
        self.assertIn("Historical recovery +24h", labels)
        self.assertIn("Actual recovery +24h", labels)
        self.assertEqual(ax.get_xlim(), (-52.0, 47.0))

        plt.close(fig)

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
                "unique_id": ["SEN_demand_SIN"],
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
            unique_id="SEN_demand_SIN",
        )

        self.assertEqual(df_future["date"].dt.strftime("%Y-%m-%d").tolist(), ["2025-05-01"])
        self.assertEqual(df_future["unique_id"].tolist(), ["SEN_demand_SIN"])
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

    def test_shape_pearson_analog_criterion_alias_maps_cde_to_fgh(self) -> None:
        from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

        df_selector = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "holiday_name": ["Holiday D", "Holiday C", "Holiday E"],
                "anchor_holiday_name": ["Holiday D", "Holiday C", "Holiday E"],
                "date": [
                    pd.Timestamp("2024-04-18"),
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-11-18"),
                ],
                "holiday_day_type": ["H2", "H2", "H2"],
                "event_profile_cluster": ["D", "C", "E"],
            }
        )
        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "anchor_holiday_name": ["Holiday D", "Holiday C", "Holiday E"],
                "holiday_day_type": ["H2", "H2", "H2"],
                "inferred_event_profile_cluster": ["D", "C", "E"],
            }
        )

        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector,
            df_priors=df_priors,
            criterion="shape_pearson_CDE_map_FGH",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        df_selector_clusters = cluster_results["df_selector_clusters"].sort_values("date").reset_index(drop=True)
        df_catalog = cluster_results["analog_cluster_catalog"]

        self.assertEqual(df_selector_clusters["analog_cluster"].tolist(), ["G", "F", "H"])
        self.assertEqual(df_catalog["analog_criterion_value"].tolist(), ["C", "D", "E"])
        self.assertEqual(df_catalog["analog_cluster"].tolist(), ["F", "G", "H"])

    def test_analog_cluster_criteria_catalog_lists_public_criteria(self) -> None:
        from analog_holidays.shared.identify_holidays import ANALOG_CLUSTER_CRITERIA_CATALOG

        self.assertEqual(
            set(ANALOG_CLUSTER_CRITERIA_CATALOG),
            {
                "shape_pearson_CDE_map_FGH",
                "seasonal_heat_cold",
                "seasonal_winter_sprint_fall",
                "best_matching_weekday",
                "observance_tier",
                "observance_tier_depth",
                "holiday_identity",
            },
        )

    def test_observance_tier_depth_criterion_splits_full_into_civic_and_deep(self) -> None:
        from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

        anchors = ["Labor Day", "Holy Saturday", "Independence Day", "Christmas Day"]
        df_selector = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 4,
                "holiday_name": anchors,
                "anchor_holiday_name": anchors,
                "date": [
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-03-30"),
                    pd.Timestamp("2024-09-16"),
                    pd.Timestamp("2024-12-25"),
                ],
                "holiday_day_type": ["H2", "H3", "H2", "H2"],
            }
        )
        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 4,
                "anchor_holiday_name": anchors,
                "holiday_day_type": ["H2", "H3", "H2", "H2"],
            }
        )

        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector,
            df_priors=df_priors,
            criterion="observance_tier_depth",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        df_selector_clusters = cluster_results["df_selector_clusters"]
        df_catalog = cluster_results["analog_cluster_catalog"]
        by_anchor = dict(
            zip(df_selector_clusters["anchor_holiday_name"], df_selector_clusters["analog_criterion_value"])
        )
        self.assertEqual(by_anchor["Labor Day"], "working")
        self.assertEqual(by_anchor["Holy Saturday"], "partial")
        self.assertEqual(by_anchor["Independence Day"], "full_civic")
        self.assertEqual(by_anchor["Christmas Day"], "full_deep")
        # four ordered tiers map to F/G/H/I
        self.assertEqual(
            df_catalog["analog_criterion_value"].tolist(),
            ["working", "partial", "full_civic", "full_deep"],
        )
        self.assertEqual(df_catalog["analog_cluster"].tolist(), ["F", "G", "H", "I"])

    def test_observance_tier_criterion_assigns_working_partial_full(self) -> None:
        from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

        df_selector = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 3,
                "holiday_name": ["Labor Day", "Holy Saturday", "Christmas Day"],
                "anchor_holiday_name": ["Labor Day", "Holy Saturday", "Christmas Day"],
                "date": [
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-03-30"),
                    pd.Timestamp("2024-12-25"),
                ],
                "holiday_day_type": ["H2", "H3", "H2"],
            }
        )
        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * 3,
                "anchor_holiday_name": ["Labor Day", "Holy Saturday", "Christmas Day"],
                "holiday_day_type": ["H2", "H3", "H2"],
            }
        )

        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector,
            df_priors=df_priors,
            criterion="observance_tier",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        df_selector_clusters = cluster_results["df_selector_clusters"].sort_values("date").reset_index(drop=True)
        df_catalog = cluster_results["analog_cluster_catalog"]

        by_anchor = dict(
            zip(df_selector_clusters["anchor_holiday_name"], df_selector_clusters["analog_criterion_value"])
        )
        self.assertEqual(by_anchor["Labor Day"], "working")
        self.assertEqual(by_anchor["Holy Saturday"], "partial")
        self.assertEqual(by_anchor["Christmas Day"], "full")
        # ordered_values place working->F, partial->G, full->H
        self.assertEqual(df_catalog["analog_criterion_value"].tolist(), ["working", "partial", "full"])
        self.assertEqual(df_catalog["analog_cluster"].tolist(), ["F", "G", "H"])

    def test_seasonal_heat_cold_criterion_assigns_binary_labels(self) -> None:
        from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

        df_selector = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "holiday_name": ["Spring Holiday", "Summer Holiday", "Winter Holiday"],
                "anchor_holiday_name": ["Spring Holiday", "Summer Holiday", "Winter Holiday"],
                "date": [
                    pd.Timestamp("2024-04-18"),
                    pd.Timestamp("2024-08-15"),
                    pd.Timestamp("2024-12-25"),
                ],
                "holiday_day_type": ["H2", "H2", "H2"],
                "season": ["Spring", "Summer", "Winter"],
            }
        )
        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "anchor_holiday_name": ["Spring Holiday", "Summer Holiday", "Winter Holiday"],
                "holiday_day_type": ["H2", "H2", "H2"],
            }
        )

        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector,
            df_priors=df_priors,
            criterion="seasonal_heat_cold",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        df_selector_clusters = cluster_results["df_selector_clusters"].sort_values("date").reset_index(drop=True)
        df_catalog = cluster_results["analog_cluster_catalog"]

        self.assertEqual(df_selector_clusters["analog_criterion_value"].tolist(), ["heat", "heat", "cold"])
        self.assertEqual(df_selector_clusters["analog_cluster"].tolist(), ["F", "F", "G"])
        self.assertEqual(df_catalog["analog_criterion_value"].tolist(), ["heat", "cold"])
        self.assertEqual(df_catalog["analog_cluster"].tolist(), ["F", "G"])

    def test_best_matching_weekday_criterion_assigns_stable_labels(self) -> None:
        from analog_holidays.shared.identify_holidays import assign_holiday_selector_analog_clusters

        df_selector = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "holiday_name": ["Holiday Friday", "Holiday Saturday", "Holiday Sunday"],
                "anchor_holiday_name": ["Holiday Friday", "Holiday Saturday", "Holiday Sunday"],
                "date": [
                    pd.Timestamp("2024-04-18"),
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-11-18"),
                ],
                "holiday_day_type": ["H2", "H2", "H2"],
                "best_matching_weekday": ["Friday", "Saturday", "Sunday"],
            }
        )
        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN", "SEN_demand_SIN", "SEN_demand_SIN"],
                "anchor_holiday_name": ["Holiday Friday", "Holiday Saturday", "Holiday Sunday"],
                "holiday_day_type": ["H2", "H2", "H2"],
                "inferred_best_matching_weekday": ["Friday", "Saturday", "Sunday"],
            }
        )

        cluster_results = assign_holiday_selector_analog_clusters(
            df_selector=df_selector,
            df_priors=df_priors,
            criterion="best_matching_weekday",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        df_selector_clusters = cluster_results["df_selector_clusters"].sort_values("date").reset_index(drop=True)
        df_catalog = cluster_results["analog_cluster_catalog"]

        self.assertEqual(df_selector_clusters["analog_cluster"].tolist(), ["F", "G", "H"])
        self.assertEqual(df_catalog["analog_criterion_value"].tolist(), ["Friday", "Saturday", "Sunday"])
        self.assertEqual(df_catalog["analog_cluster"].tolist(), ["F", "G", "H"])

    def test_identify_future_holiday_analog_cluster_supports_seasonal_criterion(self) -> None:
        from analog_holidays.shared.identify_holidays import identify_future_holiday_analog_cluster

        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"],
                "anchor_holiday_name": ["Labor Day"],
                "holiday_day_type": ["H2"],
                "history_rows": [4],
            }
        )

        candidate_info = identify_future_holiday_analog_cluster(
            candidate={
                "unique_id": "SEN_demand_SIN",
                "holiday_name": "Labor Day",
                "anchor_holiday_name": "Labor Day",
                "holiday_day_type": "H2",
                "date": pd.Timestamp("2025-05-01"),
            },
            df_priors=df_priors,
            criterion="seasonal_winter_sprint_fall",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        self.assertEqual(candidate_info["analog_criterion_value"], "spring")
        self.assertEqual(candidate_info["analog_cluster"], "G")

    def test_identify_future_holiday_analog_cluster_supports_heat_cold_criterion(self) -> None:
        from analog_holidays.shared.identify_holidays import identify_future_holiday_analog_cluster

        df_priors = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"],
                "anchor_holiday_name": ["Labor Day"],
                "holiday_day_type": ["H2"],
                "history_rows": [4],
            }
        )

        candidate_info = identify_future_holiday_analog_cluster(
            candidate={
                "unique_id": "SEN_demand_SIN",
                "holiday_name": "Labor Day",
                "anchor_holiday_name": "Labor Day",
                "holiday_day_type": "H2",
                "date": pd.Timestamp("2025-05-01"),
            },
            df_priors=df_priors,
            criterion="seasonal_heat_cold",
            group_cols=("unique_id", "anchor_holiday_name", "holiday_day_type"),
        )

        self.assertEqual(candidate_info["analog_criterion_value"], "heat")
        self.assertEqual(candidate_info["analog_cluster"], "F")

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
                match_target_cluster=False,
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
        scale_method = "standard"

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
                scale_method=scale_method,
                regressor_params=regressor_params,
                special_labels=("holiday",),
                match_target_cluster=False,
            )

        self.assertEqual(init_kwargs["regressor_params"], regressor_params)
        self.assertEqual(core_kwargs["regressor_params"], regressor_params)
        self.assertEqual(init_kwargs["scale_method"], scale_method)
        self.assertEqual(core_kwargs["scale_method"], scale_method)

    def test_run_analog_holidays_enables_cluster_prefilter_by_default(self) -> None:
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

        captured: dict[str, object] = {}

        class DummyAnalogSpecialDays:
            def __init__(self, *args, **kwargs) -> None:
                self.kwargs = kwargs

            def fit(self, y, special_days):
                return self

            def predict(self, h, level):
                result = {"mean": np.full(h, 1.0, dtype=np.float64)}
                for lv in level:
                    result[f"lo-{lv}"] = np.full(h, 0.0, dtype=np.float64)
                    result[f"hi-{lv}"] = np.full(h, 2.0, dtype=np.float64)
                return result

        def fake_build_candidate_mask(*args, **kwargs):
            captured["match_target_cluster"] = kwargs["match_target_cluster"]
            captured["selector_cluster_lookup"] = kwargs["selector_cluster_lookup"]
            train_df = args[0]
            return pd.Series([False] * len(train_df), index=train_df.index, dtype=bool), None, []

        def fake_core(**kwargs):
            vsele = int(kwargs["vsele"])
            return (
                np.full(vsele, 1.0, dtype=np.float64),
                0.01,
                0.02,
                False,
                [],
                np.empty((0, vsele), dtype=np.float64),
            )

        cluster_lookup = {
            pd.Timestamp("2024-12-22"): "F",
            pd.Timestamp("2024-12-24"): "F",
        }

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "_resolve_selector_cluster_lookup",
            return_value=cluster_lookup,
        ) as resolve_lookup_mock, patch.object(
            analog_holidays_module,
            "_build_analog_candidate_daily_mask",
            side_effect=fake_build_candidate_mask,
        ), patch.object(
            analog_holidays_module,
            "AnalogSpecialDays",
            DummyAnalogSpecialDays,
        ), patch.object(
            analog_holidays_module,
            "analog_special_days_core",
            side_effect=fake_core,
        ), patch.object(
            analog_holidays_module,
            "build_selected_days_table",
            return_value=pd.DataFrame(),
        ):
            analog_holidays_module.run_analog_holidays(
                unique_id="SEN_demand_SIN",
                target_date="2024-12-24",
                season_length=24,
                special_labels=("holiday",),
            )

        resolve_lookup_mock.assert_called_once()
        self.assertTrue(captured["match_target_cluster"])
        self.assertEqual(captured["selector_cluster_lookup"], cluster_lookup)

    def test_run_analog_holidays_adds_recent_weekend_candidates_outside_cluster_filter(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        dates = pd.date_range("2024-12-01", periods=25, freq="D")
        hour_columns = {
            f"h_{hour:02d}": [float(day_idx * 100 + hour) for day_idx in range(len(dates))]
            for hour in range(24)
        }
        labels = ["normal_day"] * len(dates)
        holiday_names = [pd.NA] * len(dates)
        declared_flags = [False] * len(dates)

        historic_idx = list(dates).index(pd.Timestamp("2024-12-18"))
        target_idx = list(dates).index(pd.Timestamp("2024-12-25"))
        labels[historic_idx] = "holiday"
        holiday_names[historic_idx] = "Historic Holiday"
        declared_flags[historic_idx] = True
        labels[target_idx] = "holiday"
        holiday_names[target_idx] = "Target Holiday"
        declared_flags[target_idx] = True

        df_source = pd.DataFrame(
            {
                "unique_id": ["SEN_demand_SIN"] * len(dates),
                "date": dates,
                "label": labels,
                "holiday_name": holiday_names,
                "holiday_type": [pd.NA] * len(dates),
                "is_declared_holiday": declared_flags,
                "is_outlier": [False] * len(dates),
                "outlier_score": [np.nan] * len(dates),
                **hour_columns,
            }
        )

        selector_df = pd.DataFrame(
            {
                "holiday_name": ["Historic Holiday", "Target Holiday"],
                "anchor_holiday_name": ["Historic Holiday", "Target Holiday"],
                "date": pd.to_datetime(["2024-12-18", "2024-12-25"]),
                "holiday_day_type": ["H2", "H2"],
                "weekday_name": ["Wednesday", "Wednesday"],
                "day_class_code": ["1", "1"],
                "day_class_name": ["Weekday", "Weekday"],
                "season": ["Winter", "Winter"],
                "date_rule": ["fixed_date", "fixed_date"],
                "is_fixed_date": [True, True],
                "is_observed_monday_rule": [False, False],
                "best_matching_weekday": [pd.NA, "Sunday"],
                "daily_profile_cluster": [pd.NA, pd.NA],
                "daily_profile_cluster_id": [pd.NA, pd.NA],
                "daily_profile_archetype": [pd.NA, "Sunday-like"],
                "event_profile_cluster": [pd.NA, pd.NA],
                "event_profile_cluster_id": [pd.NA, pd.NA],
                "analog_cluster": ["G", "H"],
            }
        )

        class DummyAnalogSpecialDays:
            def __init__(self, *args, **kwargs) -> None:
                # The test only needs the constructor to accept the production signature.
                pass

            def fit(self, y, special_days):
                self.y = np.asarray(y, dtype=np.float64)
                self.special_days = np.asarray(special_days, dtype=np.float64)
                return self

            def predict(self, h, level):
                result = {"mean": np.full(h, 123.0, dtype=np.float64)}
                for lv in level:
                    result[f"lo-{lv}"] = np.full(h, 120.0, dtype=np.float64)
                    result[f"hi-{lv}"] = np.full(h, 126.0, dtype=np.float64)
                return result

        def fake_core(**kwargs):
            vsele = int(kwargs["vsele"])
            return (
                np.full(vsele, 123.0, dtype=np.float64),
                0.01,
                0.02,
                False,
                [0],
                np.full((1, vsele), 123.0, dtype=np.float64),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            selector_path = Path(tmp_dir) / "holiday_selector_features.csv"
            selector_df.to_csv(selector_path, index=False)

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
                    levels=[80],
                    special_labels=("holiday",),
                    min_special_points=24,
                    selector_features_path=selector_path,
                    match_target_cluster=True,
                    recent_weekend_analogs=3,
                )

        self.assertEqual(run.recent_weekend_like, "Sunday")
        self.assertEqual(
            [date_value.date().isoformat() for date_value in run.recent_weekend_dates],
            ["2024-12-08", "2024-12-15", "2024-12-22"],
        )
        self.assertEqual(int(run.special_day_daily_mask.sum()), 3)

    def test_run_analog_pre_holiday_enables_cluster_prefilter_by_default(self) -> None:
        from analog_holidays.analog import P_analog_pre_holidays as pre_holidays_module

        ds = pd.date_range("2024-12-20 00:00:00", periods=96, freq="H")
        df_source = pd.DataFrame(
            {
                "ds": ds,
                "SEN_demand_SIN": np.arange(len(ds), dtype=float),
                "SEN_demand_SIN_holiday": [0] * 48 + [1] * 24 + [0] * 24,
            }
        )

        captured: dict[str, object] = {}

        class DummyAnalogSpecialDays:
            def __init__(self, *args, **kwargs) -> None:
                self.kwargs = kwargs

            def fit(self, y, special_days):
                return self

            def predict(self, h):
                return {"mean": np.full(h, 5.0, dtype=np.float64)}

        def fake_build_pre_holiday_mask(**kwargs):
            captured["match_target_cluster"] = kwargs["match_target_cluster"]
            captured["selector_cluster_lookup"] = kwargs["selector_cluster_lookup"]
            df_hist = kwargs["df_hist"]
            return np.zeros(len(df_hist), dtype=np.float64)

        cluster_lookup = {
            pd.Timestamp("2024-12-22"): "G",
            pd.Timestamp("2024-12-24"): "G",
        }

        with patch.object(
            pre_holidays_module,
            "_resolve_selector_cluster_lookup",
            return_value=cluster_lookup,
        ) as resolve_lookup_mock, patch.object(
            pre_holidays_module,
            "_build_pre_holiday_mask",
            side_effect=fake_build_pre_holiday_mask,
        ), patch.object(
            pre_holidays_module,
            "AnalogSpecialDays",
            DummyAnalogSpecialDays,
        ):
            pre_holidays_module.run_analog_pre_holiday(
                unique_id="SEN_demand_SIN",
                target_date="2024-12-24",
                df_source=df_source,
                previously_w_hours=14,
                season_length=24,
            )

        resolve_lookup_mock.assert_called_once()
        self.assertTrue(captured["match_target_cluster"])
        self.assertEqual(captured["selector_cluster_lookup"], cluster_lookup)

    def test_analog_special_days_core_standard_scaling_returns_original_scale(self) -> None:
        from analog_holidays.analog.analog_special_days import analog_special_days_core

        serie = np.asarray([10.0, 12.0, 14.0, 20.0, 22.0, 24.0, 30.0, 32.0, 34.0])
        special_days = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

        pred_raw, _, _, fail_raw, positions_raw, neighbors2_raw = analog_special_days_core(
            serie=serie,
            special_days=special_days,
            vsele=3,
            k=1,
            typedist="pearson",
            typereg="LinearReg",
        )
        pred_scaled, _, _, fail_scaled, positions_scaled, neighbors2_scaled = analog_special_days_core(
            serie=serie,
            special_days=special_days,
            vsele=3,
            k=1,
            typedist="pearson",
            typereg="LinearReg",
            scale_method="standard",
        )

        self.assertFalse(fail_raw)
        self.assertFalse(fail_scaled)
        self.assertEqual(positions_scaled, positions_raw)
        np.testing.assert_allclose(neighbors2_scaled, neighbors2_raw)
        np.testing.assert_allclose(pred_scaled, pred_raw, atol=1e-8)

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
            "_resolve_selector_weekend_like_lookup",
            return_value={},
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

    def test_tune_analog_holidays_optuna_passes_recent_weekend_analogs_to_fold_builders(self) -> None:
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

        received_recent_weekends: list[int] = []

        def fake_special_mask(*args, **kwargs):
            return pd.Series([True] * len(df_source), index=df_source.index)

        def fake_evaluate_fold(**kwargs):
            received_recent_weekends.append(int(kwargs["recent_weekend_analogs"]))
            return {
                "target_date": pd.Timestamp(kwargs["target_date"]).date().isoformat(),
                "mae": 1.0,
                "mape_pct": 2.0,
                "selected_analogs": 3,
                "fail": False,
                "train_days": 3,
                "train_special_days": 3,
            }

        def fake_count_realizable_analog_positions(**kwargs):
            received_recent_weekends.append(int(kwargs["recent_weekend_analogs"]))
            return 4

        with patch.object(analog_holidays_module, "load_audit_source", return_value=df_source), patch.object(
            analog_holidays_module,
            "build_special_day_daily_mask",
            side_effect=fake_special_mask,
        ), patch.object(
            analog_holidays_module,
            "_resolve_selector_weekend_like_lookup",
            return_value={},
        ), patch.object(
            analog_holidays_module,
            "_evaluate_analog_holiday_fold",
            side_effect=fake_evaluate_fold,
        ), patch.object(
            analog_holidays_module,
            "_count_realizable_analog_positions",
            side_effect=fake_count_realizable_analog_positions,
        ):
            analog_holidays_module.tune_analog_holidays_optuna(
                unique_id="SEN_demand_SIN",
                train_end="2023-12-25",
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                typereg_choices=["PCR"],
                random_seed=7,
                special_labels=("holiday",),
                recent_weekend_analogs=3,
            )

        self.assertTrue(received_recent_weekends)
        self.assertTrue(all(value == 3 for value in received_recent_weekends))

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

    def test_tune_analog_holidays_optuna_honors_configurable_min_k(self) -> None:
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
                optuna_min_k=2,
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                typereg_choices=["PCR"],
                random_seed=7,
                special_labels=("holiday",),
            )

        self.assertTrue(k_bounds)
        self.assertEqual(k_bounds[0], (2, 4))
        self.assertGreaterEqual(result.best_config["k"], 2)
        self.assertLessEqual(result.best_config["k"], 4)

    def test_tune_analog_holidays_optuna_can_tune_scale_method(self) -> None:
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

        used_scale_methods: list[object] = []

        def fake_evaluate_fold(**kwargs):
            used_scale_methods.append(kwargs["scale_method"])
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

        scale_choices: list[tuple[object, ...]] = []
        original_suggest_categorical = optuna.trial._trial.Trial.suggest_categorical

        def recording_suggest_categorical(self, name, choices, *args, **kwargs):
            if name == "scale_method":
                scale_choices.append(tuple(choices))
                return "minmax"
            return original_suggest_categorical(self, name, choices, *args, **kwargs)

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
            "suggest_categorical",
            new=recording_suggest_categorical,
        ):
            result = analog_holidays_module.tune_analog_holidays_optuna(
                unique_id="SEN_demand_SIN",
                train_end="2023-12-25",
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                typereg_choices=["PCR"],
                scale_method_choices=[None, "standard", "minmax"],
                random_seed=7,
                special_labels=("holiday",),
            )

        self.assertEqual(scale_choices[0], (None, "standard", "minmax"))
        self.assertEqual(result.best_config["scale_method"], "minmax")
        self.assertIn("minmax", used_scale_methods)

    def test_tune_analog_holidays_optuna_excludes_lgbm_from_default_grid(self) -> None:
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

        typereg_choices_seen: list[tuple[object, ...]] = []
        original_suggest_categorical = optuna.trial._trial.Trial.suggest_categorical

        def recording_suggest_categorical(self, name, choices, *args, **kwargs):
            if name == "typereg":
                typereg_choices_seen.append(tuple(choices))
            return original_suggest_categorical(self, name, choices, *args, **kwargs)

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
            analog_holidays_module.importlib.util,
            "find_spec",
            return_value=object(),
        ), patch.object(
            optuna.trial._trial.Trial,
            "suggest_categorical",
            new=recording_suggest_categorical,
        ):
            analog_holidays_module.tune_analog_holidays_optuna(
                unique_id="SEN_demand_SIN",
                train_end="2023-12-25",
                n_trials=1,
                timeout_sec=60,
                max_eval_dates=2,
                typedist_choices=["pearson"],
                random_seed=7,
                special_labels=("holiday",),
            )

        self.assertTrue(typereg_choices_seen)
        self.assertEqual(typereg_choices_seen[0], ("PCR", "PLS"))

    def test_save_experiment_run_registers_timestamped_folder(self) -> None:
        import types
        from datetime import datetime
        from analog_holidays.shared.experiment_logging import save_experiment_run

        def _batch(unique_id: str, mape: float) -> object:
            results_df = pd.DataFrame(
                {
                    "target_date": ["2026-05-01", "2026-09-16"],
                    "holiday_label": ["Labor Day", "Independence Day"],
                    "analog_cluster": ["F", "G"],
                    "k": [100, 100],
                    "selected_analogs": [8, 5],
                    "typereg": ["PCR", "PCR"],
                    "mae_24h": [120.0, 90.0],
                    "mape_24h_pct": [mape, mape + 1.0],
                    "fail": [False, False],
                    "error": [None, None],
                }
            )
            return types.SimpleNamespace(results_df=results_df)

        config = {
            "window": {"season_length": 38, "forecast_start_offset_hours": 14},
            "cluster": {"use_cluster": True, "analog_cluster_criterion": "shape_pearson_CDE_map_FGH"},
            "k_np": np.int64(100),  # exercises numpy serialization
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment_dir = save_experiment_run(
                config=config,
                batch_results={
                    "SEN_demand_SIN": _batch("SEN_demand_SIN", 4.2),
                    "SEN_demand_PEN": _batch("SEN_demand_PEN", 6.7),
                },
                base_dir=tmp_dir,
                slug="unit test",
                timestamp=datetime(2026, 6, 13, 9, 6),
                verbose=False,
            )

            self.assertEqual(experiment_dir.name, "experiment_2026_06_13_09_06_unit_test")
            for fname in ("manifest.yaml", "metrics.csv", "summary.csv", "notes.md"):
                self.assertTrue((experiment_dir / fname).exists(), fname)

            metrics = pd.read_csv(experiment_dir / "metrics.csv")
            self.assertEqual(list(metrics.columns[:2]), ["experiment_id", "unique_id"])
            self.assertEqual(len(metrics), 4)
            self.assertEqual(
                set(metrics["unique_id"]), {"SEN_demand_SIN", "SEN_demand_PEN"}
            )
            self.assertTrue((metrics["experiment_id"] == experiment_dir.name).all())

            summary = pd.read_csv(experiment_dir / "summary.csv")
            self.assertIn("ALL", set(summary["unique_id"]))
            self.assertIn("mape_24h_pct_median", summary.columns)

            manifest_text = (experiment_dir / "manifest.yaml").read_text(encoding="utf-8")
            self.assertIn("shape_pearson_CDE_map_FGH", manifest_text)
            self.assertIn(experiment_dir.name, manifest_text)

            # Reruns at the same minute must never overwrite a prior experiment folder.
            experiment_dir_2 = save_experiment_run(
                config=config,
                batch_results={"SEN_demand_SIN": _batch("SEN_demand_SIN", 4.2)},
                base_dir=tmp_dir,
                slug="unit test",
                timestamp=datetime(2026, 6, 13, 9, 6),
                verbose=False,
            )
            self.assertNotEqual(experiment_dir, experiment_dir_2)
            self.assertTrue(experiment_dir_2.name.startswith(experiment_dir.name))


if __name__ == "__main__":
    unittest.main()