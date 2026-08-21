"""Integration tests cho report, hình, exporter và runner độc lập."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.reports import create_classification_report_dataframe
from evaluation.visualizations import plot_per_class_metrics
from experiments.run_classification_evaluation import (
    run_classification_evaluation,
)


class ClassificationReportTests(unittest.TestCase):
    """Kiểm tra report chỉ tổ chức metric đã có."""

    def test_report_contains_class_and_summary_rows(self) -> None:
        per_class = [
            {
                "class_label": 0,
                "class_name": "Negative",
                "precision": 0.75,
                "recall": 0.60,
                "f1_score": 2 * 0.75 * 0.60 / (0.75 + 0.60),
                "support": 5,
            },
            {
                "class_label": 1,
                "class_name": "Positive",
                "precision": 0.80,
                "recall": 1.00,
                "f1_score": 2 * 0.80 / 1.80,
                "support": 4,
            },
        ]
        aggregate = {
            "precision_macro": 0.775,
            "recall_macro": 0.80,
            "f1_macro": 0.7777777777777778,
            "precision_weighted": 0.7722222222222223,
            "recall_weighted": 0.7777777777777778,
            "f1_weighted": 0.7703703703703704,
        }

        report = create_classification_report_dataframe(
            per_class,
            aggregate,
            accuracy=7 / 9,
        )

        self.assertEqual(
            report["label"].tolist(),
            ["Negative", "Positive", "accuracy", "macro avg", "weighted avg"],
        )
        self.assertEqual(report["support"].tolist(), [5, 4, 9, 9, 9])
        self.assertEqual(
            report["row_type"].tolist(),
            ["class", "class", "summary", "summary", "summary"],
        )
        accuracy_row = report.loc[report["label"] == "accuracy"].iloc[0]
        self.assertAlmostEqual(accuracy_row["f1_score"], 7 / 9)
        self.assertTrue(np.isnan(accuracy_row["precision"]))

    def test_reserved_summary_text_is_allowed_as_a_class_name(self) -> None:
        per_class = [
            {
                "class_label": 0,
                "class_name": "accuracy",
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "support": 1,
            },
            {
                "class_label": 1,
                "class_name": "macro avg",
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "support": 1,
            },
        ]
        aggregate = {
            "precision_macro": 1.0,
            "recall_macro": 1.0,
            "f1_macro": 1.0,
            "precision_weighted": 1.0,
            "recall_weighted": 1.0,
            "f1_weighted": 1.0,
        }
        report = create_classification_report_dataframe(per_class, aggregate, 1.0)

        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "per_class.png"
            plot_per_class_metrics(report, output_path, dpi=100)
            self.assertTrue(output_path.is_file())


class StandaloneRunnerIntegrationTests(unittest.TestCase):
    """Chạy end-to-end từ prediction arrays tới CSV và PNG."""

    def test_binary_runner_creates_all_expected_outputs(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        y_proba = np.array(
            [0.100000123456, 0.400000234567, 0.350000345678, 0.800000456789]
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            result = run_classification_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                classes=[0, 1],
                class_names=["Negative", "Positive"],
                positive_label=1,
                task_type="binary",
                output_dir=output_dir,
            )

            self.assertAlmostEqual(result["metrics"]["accuracy"], 0.75)
            self.assertAlmostEqual(result["metrics"]["roc_auc"], 0.75)
            self._assert_all_paths_exist(result)

            expected_tables = {
                "metrics_summary.csv",
                "classification_report.csv",
                "confusion_matrix_counts.csv",
                "confusion_matrix_normalized.csv",
                "roc_points.csv",
            }
            self.assertEqual(
                {path.name for path in (output_dir / "tables").iterdir()},
                expected_tables,
            )
            self.assertTrue((output_dir / "predictions" / "predictions.csv").is_file())

            prediction_frame = pd.read_csv(
                output_dir / "predictions" / "predictions.csv"
            )
            self.assertEqual(
                prediction_frame.columns.tolist(),
                [
                    "actual_label",
                    "predicted_label",
                    "probability_positive",
                    "is_correct",
                ],
            )
            self.assertEqual(len(prediction_frame), len(y_true))
            np.testing.assert_allclose(
                prediction_frame["probability_positive"].to_numpy(),
                y_proba,
                rtol=0.0,
                atol=5e-12,
            )
            self.assertEqual(plt.get_fignums(), [])

    def test_binary_two_column_probability_uses_positive_label_column(self) -> None:
        y_true = np.array(["normal", "normal", "fraud", "fraud"])
        y_pred = np.array(["normal", "normal", "normal", "fraud"])
        y_proba = np.array(
            [
                [0.10, 0.90],
                [0.40, 0.60],
                [0.35, 0.65],
                [0.80, 0.20],
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            result = run_classification_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                classes=["fraud", "normal"],
                class_names=["Fraud", "Normal"],
                positive_label="fraud",
                task_type="binary",
                output_dir=output_dir,
                save_dpi=100,
            )

            self.assertEqual(result["metadata"]["positive_index"], 0)
            self.assertAlmostEqual(result["metrics"]["roc_auc"], 0.75)
            predictions = pd.read_csv(
                output_dir / "predictions" / "predictions.csv"
            )
            np.testing.assert_allclose(
                predictions["probability_fraud"].to_numpy(),
                y_proba[:, 0],
            )

    def test_multiclass_runner_creates_ovr_roc_files(self) -> None:
        y_true = np.array(["a", "b", "c", "a", "b", "c"])
        y_pred = np.array(["a", "b", "c", "b", "b", "a"])
        y_proba = np.array(
            [
                [0.85, 0.10, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.15, 0.80],
                [0.35, 0.55, 0.10],
                [0.15, 0.70, 0.15],
                [0.50, 0.20, 0.30],
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            result = run_classification_evaluation(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                classes=["a", "b", "c"],
                class_names=["A-B", "A B", "!"],
                task_type="multiclass",
                output_dir=output_dir,
                save_dpi=100,
            )

            self.assertEqual(result["metadata"]["task_type"], "multiclass")
            self.assertIsNotNone(result["roc_results"]["macro_auc"])
            self.assertEqual(len(result["table_paths"]["roc_points"]), 3)
            self.assertEqual(
                result["figure_paths"]["roc_curve"].name,
                "roc_ovr_multiclass.png",
            )
            self._assert_all_paths_exist(result)
            probability_columns = [
                column
                for column in pd.read_csv(
                    output_dir / "predictions" / "predictions.csv"
                ).columns
                if column.startswith("probability_")
            ]
            self.assertEqual(
                probability_columns,
                ["probability_a_b", "probability_a_b_2", "probability_class"],
            )
            confusion_frame = pd.read_csv(
                output_dir / "tables" / "confusion_matrix_counts.csv"
            )
            self.assertEqual(
                confusion_frame.columns.tolist(),
                [
                    "true_label",
                    "predicted_a_b",
                    "predicted_a_b_2",
                    "predicted_class",
                ],
            )
            self.assertEqual(plt.get_fignums(), [])

    def test_multiclass_runner_keeps_outputs_when_all_roc_curves_are_undefined(
        self,
    ) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        y_proba = np.array(
            [
                [0.90, 0.07, 0.03],
                [0.80, 0.15, 0.05],
                [0.85, 0.10, 0.05],
            ]
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            with self.assertWarns(RuntimeWarning):
                result = run_classification_evaluation(
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    classes=[0, 1, 2],
                    class_names=["Zero", "One", "Two"],
                    task_type="multiclass",
                    output_dir=output_dir,
                    save_dpi=100,
                )

            self.assertIsNone(result["metrics"]["roc_auc_macro"])
            self.assertEqual(len(result["roc_results"]["undefined_classes"]), 3)
            self.assertEqual(len(result["table_paths"]["roc_points"]), 3)
            self._assert_all_paths_exist(result)
            summary = pd.read_csv(output_dir / "tables" / "metrics_summary.csv")
            auc_row = summary.loc[summary["metric"] == "roc_auc_macro"].iloc[0]
            self.assertTrue(np.isnan(auc_row["value"]))
            self.assertEqual(plt.get_fignums(), [])

    def test_reusing_output_dir_removes_only_stale_pipeline_artifacts(self) -> None:
        binary_arguments = {
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 0, 1],
            "y_proba": [0.10, 0.40, 0.35, 0.80],
            "classes": [0, 1],
            "class_names": ["Negative", "Positive"],
            "positive_label": 1,
            "task_type": "binary",
        }
        multiclass_arguments = {
            "y_true": [0, 1, 2],
            "y_pred": [0, 1, 2],
            "y_proba": np.eye(3),
            "classes": [0, 1, 2],
            "class_names": ["Zero", "One", "Two"],
            "task_type": "multiclass",
        }

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "outputs"
            run_classification_evaluation(
                **binary_arguments,
                output_dir=output_dir,
                save_dpi=100,
            )
            user_file = output_dir / "keep_me.txt"
            user_file.write_text("user-owned", encoding="utf-8")
            previous_roc_path = output_dir / "tables" / "roc_points.csv"
            previous_manifest_path = output_dir / "evaluation_manifest.json"
            with self.assertRaisesRegex(ValueError, "save_dpi"):
                run_classification_evaluation(
                    **binary_arguments,
                    output_dir=output_dir,
                    save_dpi=0,
                )
            self.assertTrue(previous_roc_path.is_file())
            self.assertTrue(previous_manifest_path.is_file())

            run_classification_evaluation(
                **multiclass_arguments,
                output_dir=output_dir,
                save_dpi=100,
            )
            self.assertFalse((output_dir / "tables" / "roc_points.csv").exists())
            self.assertFalse((output_dir / "figures" / "roc_curve.png").exists())
            self.assertEqual(
                len(list((output_dir / "tables").glob("roc_points_class_*.csv"))),
                3,
            )
            self.assertTrue(user_file.is_file())

            run_classification_evaluation(
                **binary_arguments,
                output_dir=output_dir,
                save_dpi=100,
            )
            self.assertEqual(
                list((output_dir / "tables").glob("roc_points_class_*.csv")),
                [],
            )
            self.assertFalse(
                (output_dir / "figures" / "roc_ovr_multiclass.png").exists()
            )
            self.assertTrue((output_dir / "tables" / "roc_points.csv").is_file())
            self.assertTrue(user_file.is_file())

    def _assert_all_paths_exist(self, result: dict[str, object]) -> None:
        """Duyệt cấu trúc đường dẫn trả về và kiểm tra tệp không rỗng."""

        def assert_path_value(value: object) -> None:
            if isinstance(value, Path):
                self.assertTrue(value.is_file(), f"Thiếu output: {value}")
                self.assertGreater(value.stat().st_size, 0, f"Output rỗng: {value}")
            elif isinstance(value, dict):
                for nested_value in value.values():
                    assert_path_value(nested_value)

        assert_path_value(result["table_paths"])
        assert_path_value(result["figure_paths"])
        assert_path_value(result["manifest_path"])


if __name__ == "__main__":
    unittest.main()
