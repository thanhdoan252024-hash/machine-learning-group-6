"""Unit tests for the manually implemented classification metrics."""

import unittest

import numpy as np

from evaluation.manual_metrics import (
    build_confusion_matrix,
    calculate_accuracy_from_confusion_matrix,
    calculate_aggregate_metrics,
    calculate_f1_score,
    calculate_ovr_counts,
    calculate_per_class_metrics,
    calculate_precision,
    calculate_recall,
    evaluate_classification,
    normalize_confusion_matrix,
    safe_divide,
)


class TestSafeDivideAndScalarMetrics(unittest.TestCase):
    """Verify scalar formulas and their zero-denominator behavior."""

    def test_safe_divide_returns_value_and_optional_status(self) -> None:
        self.assertAlmostEqual(safe_divide(3, 4), 0.75)
        self.assertEqual(safe_divide(2, 0), 0.0)
        self.assertEqual(
            safe_divide(2, 0, undefined_value=-1.0, return_status=True),
            (-1.0, True),
        )
        self.assertEqual(safe_divide(2, 4, return_status=True), (0.5, False))

    def test_precision_recall_and_f1(self) -> None:
        self.assertAlmostEqual(calculate_precision(2, 1), 2.0 / 3.0)
        self.assertAlmostEqual(calculate_recall(2, 1), 2.0 / 3.0)
        self.assertAlmostEqual(calculate_f1_score(2.0 / 3.0, 2.0 / 3.0), 2.0 / 3.0)

    def test_scalar_metrics_return_zero_when_undefined(self) -> None:
        self.assertEqual(calculate_precision(0, 0), 0.0)
        self.assertEqual(calculate_recall(0, 0), 0.0)
        self.assertEqual(calculate_f1_score(0.0, 0.0), 0.0)

    def test_scalar_metric_validation(self) -> None:
        with self.assertRaises(ValueError):
            safe_divide(np.inf, 1)
        with self.assertRaises(ValueError):
            safe_divide(1, 0, undefined_value=np.nan)
        with self.assertRaises(ValueError):
            calculate_precision(-1, 2)
        with self.assertRaises(ValueError):
            calculate_f1_score(1.1, 0.5)


class TestConfusionMatrix(unittest.TestCase):
    """Verify count construction, normalization, and one-vs-rest counts."""

    def test_binary_confusion_matrix_and_counts(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])

        matrix = build_confusion_matrix(y_true, y_pred, [0, 1])

        np.testing.assert_array_equal(matrix, np.array([[1, 1], [1, 2]]))
        self.assertEqual(
            calculate_ovr_counts(matrix, 1),
            {"tp": 2, "tn": 1, "fp": 1, "fn": 1},
        )
        self.assertAlmostEqual(calculate_accuracy_from_confusion_matrix(matrix), 0.6)

    def test_string_labels_preserve_class_order(self) -> None:
        y_true = np.array(["normal", "fraud", "normal"])
        y_pred = np.array(["normal", "normal", "normal"])

        matrix = build_confusion_matrix(
            y_true,
            y_pred,
            ["normal", "fraud"],
        )

        np.testing.assert_array_equal(matrix, np.array([[2, 0], [1, 0]]))

    def test_multiclass_confusion_matrix(self) -> None:
        y_true = np.array([0, 1, 2, 2])
        y_pred = np.array([0, 2, 2, 1])
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]])

        np.testing.assert_array_equal(
            build_confusion_matrix(y_true, y_pred, [0, 1, 2]),
            expected,
        )

    def test_normalization_leaves_an_empty_row_at_zero(self) -> None:
        normalized = normalize_confusion_matrix(np.array([[2, 2], [0, 0]]))
        np.testing.assert_allclose(normalized, np.array([[0.5, 0.5], [0.0, 0.0]]))

    def test_confusion_matrix_errors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_confusion_matrix([], [], [0, 1])
        with self.assertRaises(ValueError):
            build_confusion_matrix([0, 1], [0], [0, 1])
        with self.assertRaises(ValueError):
            build_confusion_matrix([0, 2], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            normalize_confusion_matrix([[1, 0]], mode="true")
        with self.assertRaises(ValueError):
            normalize_confusion_matrix([[1, 0], [0, 1]], mode="pred")
        with self.assertRaises(ValueError):
            calculate_ovr_counts([[1, 0], [0, 1]], 2)
        with self.assertRaises(ValueError):
            calculate_accuracy_from_confusion_matrix([[0, 0], [0, 0]])


class TestPerClassAndAggregateMetrics(unittest.TestCase):
    """Verify per-class, macro, and support-weighted calculations."""

    def test_multiclass_per_class_metrics(self) -> None:
        matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]])

        metrics = calculate_per_class_metrics(
            matrix,
            classes=[0, 1, 2],
            class_names=["zero", "one", "two"],
        )

        self.assertEqual([item["support"] for item in metrics], [1, 1, 2])
        self.assertEqual(metrics[0]["class_name"], "zero")
        self.assertAlmostEqual(metrics[0]["precision"], 1.0)
        self.assertAlmostEqual(metrics[0]["recall"], 1.0)
        self.assertAlmostEqual(metrics[1]["precision"], 0.0)
        self.assertAlmostEqual(metrics[2]["precision"], 0.5)
        self.assertAlmostEqual(metrics[2]["recall"], 0.5)
        self.assertAlmostEqual(metrics[2]["f1_score"], 0.5)

    def test_macro_and_weighted_averages_are_computed_by_hand(self) -> None:
        metrics = [
            {"precision": 0.5, "recall": 0.25, "f1_score": 1.0 / 3.0, "support": 1},
            {"precision": 1.0, "recall": 0.5, "f1_score": 2.0 / 3.0, "support": 3},
        ]

        result = calculate_aggregate_metrics(metrics)

        self.assertAlmostEqual(result["precision_macro"], 0.75)
        self.assertAlmostEqual(result["recall_macro"], 0.375)
        self.assertAlmostEqual(result["f1_macro"], 0.5)
        self.assertAlmostEqual(result["precision_weighted"], 0.875)
        self.assertAlmostEqual(result["recall_weighted"], 0.4375)
        self.assertAlmostEqual(result["f1_weighted"], 7.0 / 12.0)

    def test_no_positive_prediction_records_undefined_status(self) -> None:
        matrix = build_confusion_matrix(
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 0]),
            [0, 1],
        )

        positive_metrics = calculate_per_class_metrics(matrix, [0, 1])[1]

        self.assertEqual(positive_metrics["tp"], 0)
        self.assertEqual(positive_metrics["fp"], 0)
        self.assertEqual(positive_metrics["precision"], 0.0)
        self.assertEqual(positive_metrics["recall"], 0.0)
        self.assertEqual(positive_metrics["f1_score"], 0.0)
        self.assertTrue(positive_metrics["precision_undefined"])
        self.assertFalse(positive_metrics["recall_undefined"])
        self.assertTrue(positive_metrics["f1_undefined"])

    def test_aggregate_errors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_aggregate_metrics([])
        with self.assertRaises(ValueError):
            calculate_aggregate_metrics(
                [{"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}]
            )


class TestEvaluateClassification(unittest.TestCase):
    """Verify the end-to-end label-metric result dictionary."""

    def test_binary_metrics_match_hand_calculation(self) -> None:
        result = evaluate_classification(
            y_true=np.array([0, 0, 1, 1, 1]),
            y_pred=np.array([0, 1, 1, 0, 1]),
            classes=[0, 1],
            class_names=["negative", "positive"],
            positive_label=1,
        )

        self.assertEqual(result["task_type"], "binary")
        self.assertEqual(result["positive_label"], 1)
        self.assertAlmostEqual(result["accuracy"], 3.0 / 5.0)
        self.assertAlmostEqual(result["precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(result["f1_score"], 2.0 / 3.0)
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            np.array([[1, 1], [1, 2]]),
        )

    def test_perfect_binary_predictions(self) -> None:
        result = evaluate_classification(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1],
            positive_label=1,
        )

        for metric_name in ("accuracy", "precision", "recall", "f1_score"):
            self.assertAlmostEqual(result[metric_name], 1.0)

    def test_binary_string_labels(self) -> None:
        result = evaluate_classification(
            ["normal", "fraud", "normal"],
            ["normal", "normal", "normal"],
            ["normal", "fraud"],
            class_names=["Normal", "Fraud"],
            positive_label="fraud",
        )

        self.assertEqual(result["positive_label"], "fraud")
        self.assertEqual(result["precision"], 0.0)
        self.assertTrue(result["precision_undefined"])

    def test_multiclass_evaluation(self) -> None:
        result = evaluate_classification(
            [0, 1, 2, 2],
            [0, 2, 2, 1],
            [0, 1, 2],
        )

        self.assertEqual(result["task_type"], "multiclass")
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["precision_macro"], 0.5)
        self.assertAlmostEqual(result["recall_macro"], 0.5)
        self.assertAlmostEqual(result["f1_macro"], 0.5)
        self.assertAlmostEqual(result["precision_weighted"], 0.5)
        self.assertNotIn("positive_label", result)

    def test_binary_requires_an_explicit_valid_positive_label(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_classification([0, 1], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            evaluate_classification([0, 1], [0, 1], [0, 1], positive_label=2)

    def test_invalid_task_configurations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_classification([0], [0], [0], positive_label=0)
        with self.assertRaises(ValueError):
            evaluate_classification(
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
                positive_label=1,
            )


if __name__ == "__main__":
    unittest.main()
