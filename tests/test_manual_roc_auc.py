"""Unit tests for the manual ROC and AUC implementation."""

import unittest

import numpy as np

from evaluation.manual_roc_auc import (
    calculate_auc_trapezoid,
    calculate_binary_roc_curve,
    calculate_multiclass_roc_ovr,
    convert_to_binary_targets,
    extract_positive_scores,
)


class BinaryRocAucTests(unittest.TestCase):
    """Verify binary target conversion, curve construction and integration."""

    def test_convert_to_binary_targets_supports_string_labels(self) -> None:
        labels = np.array(["normal", "fraud", "normal", "fraud"])

        result = convert_to_binary_targets(labels, "fraud")

        np.testing.assert_array_equal(result, np.array([0, 1, 0, 1]))
        self.assertEqual(result.dtype, np.dtype(np.int64))

    def test_convert_to_binary_targets_rejects_non_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            convert_to_binary_targets(np.array([[0, 1]]), 1)

    def test_extract_positive_scores_accepts_one_dimensional_input(self) -> None:
        probabilities = np.array([0.15, 0.8, 0.4])

        result = extract_positive_scores(probabilities, ["no", "yes"], "yes")

        np.testing.assert_allclose(result, probabilities)
        self.assertIsNot(result, probabilities)

    def test_extract_positive_scores_uses_class_column_order(self) -> None:
        probabilities = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )

        result = extract_positive_scores(
            probabilities,
            classes=["middle", "negative", "positive"],
            positive_label="positive",
        )

        np.testing.assert_allclose(result, np.array([0.1, 0.6]))

    def test_extract_positive_scores_rejects_unknown_label_and_bad_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            extract_positive_scores(np.array([0.2, 0.8]), [0, 1], 2)
        with self.assertRaisesRegex(ValueError, "column count"):
            extract_positive_scores(np.ones((2, 3)) / 3.0, [0, 1], 1)

    def test_known_ranking_has_auc_point_seven_five(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        result = calculate_binary_roc_curve(y_true, y_score, positive_label=1)

        self.assertAlmostEqual(float(result["auc"]), 0.75, places=12)
        self.assertTrue(np.isinf(result["thresholds"][0]))
        self.assertEqual(float(result["fpr"][0]), 0.0)
        self.assertEqual(float(result["tpr"][0]), 0.0)
        self.assertEqual(float(result["fpr"][-1]), 1.0)
        self.assertEqual(float(result["tpr"][-1]), 1.0)

    def test_all_tied_scores_form_one_group_and_auc_is_half(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_score = np.full(4, 0.5)

        result = calculate_binary_roc_curve(y_true, y_score, positive_label=1)
        permuted = calculate_binary_roc_curve(
            y_true[[1, 3, 0, 2]],
            y_score[[1, 3, 0, 2]],
            positive_label=1,
        )

        np.testing.assert_allclose(result["fpr"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(result["tpr"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(result["thresholds"][1:], np.array([0.5]))
        np.testing.assert_allclose(permuted["fpr"], result["fpr"])
        np.testing.assert_allclose(permuted["tpr"], result["tpr"])
        self.assertAlmostEqual(float(result["auc"]), 0.5, places=12)

    def test_perfect_ranking_has_unit_auc(self) -> None:
        result = calculate_binary_roc_curve(
            y_true=np.array([0, 0, 1, 1]),
            y_score=np.array([0.1, 0.2, 0.8, 0.9]),
            positive_label=1,
        )

        self.assertAlmostEqual(float(result["auc"]), 1.0, places=12)

    def test_binary_curve_rejects_missing_target_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive"):
            calculate_binary_roc_curve([0, 0], [0.1, 0.2], positive_label=1)
        with self.assertRaisesRegex(ValueError, "no negative"):
            calculate_binary_roc_curve([1, 1], [0.8, 0.9], positive_label=1)

    def test_binary_curve_rejects_mismatched_or_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number"):
            calculate_binary_roc_curve([0, 1], [0.2], positive_label=1)
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            calculate_binary_roc_curve([0, 1], [0.2, np.nan], positive_label=1)

    def test_auc_loop_validates_curve_arrays(self) -> None:
        self.assertAlmostEqual(
            calculate_auc_trapezoid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
            0.5,
            places=12,
        )
        with self.assertRaisesRegex(ValueError, "same number"):
            calculate_auc_trapezoid([0.0, 1.0], [0.0])
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            calculate_auc_trapezoid([0.0, 0.8, 0.7], [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            calculate_auc_trapezoid([0.0, 1.1], [0.0, 1.0])


class MulticlassRocAucTests(unittest.TestCase):
    """Verify one-vs-rest behavior, including undefined classes."""

    def test_multiclass_perfect_probabilities(self) -> None:
        y_true = np.array(["a", "b", "c", "a", "b", "c"])
        y_proba = np.array(
            [
                [0.90, 0.08, 0.02],
                [0.10, 0.80, 0.10],
                [0.05, 0.10, 0.85],
                [0.70, 0.20, 0.10],
                [0.20, 0.65, 0.15],
                [0.10, 0.20, 0.70],
            ]
        )

        result = calculate_multiclass_roc_ovr(
            y_true,
            y_proba,
            classes=["a", "b", "c"],
            class_names=["Class A", "Class B", "Class C"],
        )

        self.assertEqual(result["undefined_classes"], [])
        self.assertAlmostEqual(result["macro_auc"], 1.0, places=12)
        self.assertEqual(set(result["per_class"]), {"a", "b", "c"})
        for class_result in result["per_class"].values():
            self.assertTrue(class_result["defined"])
            self.assertAlmostEqual(class_result["auc"], 1.0, places=12)

    def test_absent_class_is_warned_and_excluded_from_macro(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array(
            [
                [0.90, 0.09, 0.01],
                [0.10, 0.85, 0.05],
                [0.80, 0.15, 0.05],
                [0.20, 0.75, 0.05],
            ]
        )

        with self.assertWarnsRegex(RuntimeWarning, "class 2.*no positive"):
            result = calculate_multiclass_roc_ovr(
                y_true,
                y_proba,
                classes=[0, 1, 2],
            )

        self.assertEqual(result["undefined_classes"], [2])
        self.assertFalse(result["per_class"][2]["defined"])
        self.assertIsNone(result["per_class"][2]["auc"])
        self.assertEqual(result["per_class"][2]["fpr"].size, 0)
        self.assertAlmostEqual(result["macro_auc"], 1.0, places=12)

    def test_no_defined_ovr_curve_produces_none_macro(self) -> None:
        y_true = np.array([0, 0])
        y_proba = np.array([[0.9, 0.08, 0.02], [0.8, 0.15, 0.05]])

        with self.assertWarns(RuntimeWarning):
            result = calculate_multiclass_roc_ovr(
                y_true,
                y_proba,
                classes=[0, 1, 2],
            )

        self.assertIsNone(result["macro_auc"])
        self.assertEqual(result["undefined_classes"], [0, 1, 2])

    def test_multiclass_rejects_invalid_probability_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least three classes"):
            calculate_multiclass_roc_ovr(
                y_true=[0, 1],
                y_proba=np.array([[0.8, 0.2], [0.2, 0.8]]),
                classes=[0, 1],
            )
        with self.assertRaisesRegex(ValueError, "column count"):
            calculate_multiclass_roc_ovr(
                y_true=[0, 1],
                y_proba=np.array([[0.8, 0.2], [0.2, 0.8]]),
                classes=[0, 1, 2],
            )
        with self.assertRaisesRegex(ValueError, "sum approximately"):
            calculate_multiclass_roc_ovr(
                y_true=[0, 1],
                y_proba=np.array([[0.8, 0.4, 0.1], [0.2, 0.7, 0.2]]),
                classes=[0, 1, 2],
            )


if __name__ == "__main__":
    unittest.main()
