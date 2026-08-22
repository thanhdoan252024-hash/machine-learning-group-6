"""Unit tests for classification input validation."""

from __future__ import annotations

import unittest

import numpy as np

from classification.evaluation.input_validation import (
    validate_binary_configuration,
    validate_classes,
    validate_class_names,
    validate_evaluation_inputs,
    validate_label_arrays,
    validate_probability_array,
)


class LabelArrayValidationTests(unittest.TestCase):
    """Tests for true and predicted label vectors."""

    def test_accepts_numeric_and_string_labels_without_reordering(self) -> None:
        y_true, y_pred = validate_label_arrays(
            ["normal", "fraud", "normal"],
            ["normal", "normal", "normal"],
        )

        np.testing.assert_array_equal(y_true, ["normal", "fraud", "normal"])
        np.testing.assert_array_equal(y_pred, ["normal", "normal", "normal"])

    def test_accepts_an_arbitrarily_large_integer_label(self) -> None:
        large_label = 10**400

        y_true, y_pred = validate_label_arrays(
            [large_label, 0],
            [0, large_label],
        )

        self.assertEqual(y_true[0], large_label)
        self.assertEqual(y_pred[1], large_label)

    def test_rejects_non_one_dimensional_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "y_true must be one-dimensional"):
            validate_label_arrays([[0], [1]], [0, 1])

    def test_rejects_empty_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            validate_label_arrays([], [])

    def test_rejects_different_sample_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            validate_label_arrays([0, 1], [0])

    def test_rejects_nan_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "y_true contains NaN at index 1"):
            validate_label_arrays([0, np.nan], [0, 1])

    def test_rejects_infinite_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "y_pred contains infinity at index 1"):
            validate_label_arrays([0, 1], [0, np.inf])

    def test_rejects_non_numeric_non_string_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite numeric or string labels"):
            validate_label_arrays([0, None], [0, 1])


class ClassValidationTests(unittest.TestCase):
    """Tests for ordered classes and display names."""

    def test_preserves_supplied_class_order(self) -> None:
        classes = validate_classes(
            ["fraud", "normal"],
            y_true=["normal", "fraud"],
            y_pred=["fraud", "fraud"],
        )

        np.testing.assert_array_equal(classes, ["fraud", "normal"])

    def test_rejects_empty_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "classes must not be empty"):
            validate_classes([])

    def test_rejects_duplicate_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique labels"):
            validate_classes([0, 1, 0])

    def test_rejects_observed_true_label_not_in_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "y_true contains label 2"):
            validate_classes([0, 1], y_true=[0, 2], y_pred=[0, 1])

    def test_rejects_observed_prediction_not_in_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "y_pred contains label 'unknown'"):
            validate_classes(
                ["normal", "fraud"],
                y_true=["normal"],
                y_pred=["unknown"],
            )

    def test_derives_default_class_names(self) -> None:
        names = validate_class_names(None, [10, 20])

        np.testing.assert_array_equal(names, ["10", "20"])

    def test_accepts_explicit_class_names(self) -> None:
        names = validate_class_names(["Negative", "Positive"], [0, 1])

        np.testing.assert_array_equal(names, ["Negative", "Positive"])

    def test_rejects_wrong_class_name_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "one name for each class"):
            validate_class_names(["Only one"], [0, 1])

    def test_rejects_blank_class_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty name at index 1"):
            validate_class_names(["Negative", "  "], [0, 1])

    def test_rejects_non_string_class_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "only strings"):
            validate_class_names(["Negative", 1], [0, 1])


class ProbabilityValidationTests(unittest.TestCase):
    """Tests for binary and multiclass probability arrays."""

    def test_accepts_one_dimensional_binary_positive_probabilities(self) -> None:
        values = np.array([0.05, 0.75, 1.0])

        result = validate_probability_array(values, 3, 2, "binary")

        self.assertIs(result, values)

    def test_accepts_two_column_binary_matrix(self) -> None:
        values = np.array([[0.9, 0.1], [0.25, 0.75]])

        result = validate_probability_array(values, 2, 2, "binary")

        np.testing.assert_array_equal(result, values)

    def test_accepts_multiclass_matrix(self) -> None:
        values = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )

        result = validate_probability_array(values, 2, 3, "multiclass")

        np.testing.assert_array_equal(result, values)

    def test_rejects_probability_outside_unit_interval_without_clipping(self) -> None:
        values = np.array([[1.2, -0.2], [0.3, 0.7]])
        original = values.copy()

        with self.assertRaisesRegex(ValueError, "inclusive interval"):
            validate_probability_array(values, 2, 2, "binary")

        np.testing.assert_array_equal(values, original)

    def test_rejects_nan_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains NaN"):
            validate_probability_array([0.2, np.nan], 2, 2, "binary")

    def test_rejects_infinite_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "contains infinity"):
            validate_probability_array([0.2, np.inf], 2, 2, "binary")

    def test_rejects_non_numeric_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "real numeric probabilities"):
            validate_probability_array(["0.2", "0.8"], 2, 2, "binary")

    def test_rejects_wrong_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "one row/value per sample"):
            validate_probability_array([0.2], 2, 2, "binary")

    def test_rejects_wrong_binary_column_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 2 columns"):
            validate_probability_array(np.ones((2, 1)), 2, 2, "binary")

    def test_rejects_one_dimensional_multiclass_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional matrix"):
            validate_probability_array([0.1, 0.2], 2, 3, "multiclass")

    def test_rejects_wrong_multiclass_column_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "column count must match"):
            validate_probability_array(
                np.array([[0.4, 0.6], [0.2, 0.8]]),
                2,
                3,
                "multiclass",
            )

    def test_rejects_matrix_row_that_does_not_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "row 1 sums to"):
            validate_probability_array(
                np.array([[0.4, 0.6], [0.2, 0.7]]),
                2,
                2,
                "binary",
            )

    def test_accepts_matrix_row_sum_within_allclose_tolerance(self) -> None:
        values = np.array([[0.4, 0.600000001], [0.2, 0.8]])

        result = validate_probability_array(values, 2, 2, "binary")

        np.testing.assert_array_equal(result, values)


class BinaryConfigurationTests(unittest.TestCase):
    """Tests for explicit binary positive-label configuration."""

    def test_returns_positive_column_index(self) -> None:
        self.assertEqual(
            validate_binary_configuration(["positive", "negative"], "positive"),
            0,
        )

    def test_rejects_missing_positive_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive_label is required"):
            validate_binary_configuration([0, 1], None)

    def test_rejects_positive_label_outside_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not present in classes"):
            validate_binary_configuration([0, 1], 2)

    def test_rejects_non_binary_class_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            validate_binary_configuration([0, 1, 2], 1)


class CompleteInputValidationTests(unittest.TestCase):
    """Tests for the consolidated evaluation-input validator."""

    def test_returns_normalized_binary_metadata(self) -> None:
        metadata = validate_evaluation_inputs(
            y_true=["no", "yes", "yes"],
            y_pred=["no", "no", "yes"],
            y_proba=[0.1, 0.4, 0.9],
            classes=["no", "yes"],
            class_names=["Negative", "Positive"],
            positive_label="yes",
        )

        self.assertEqual(metadata["n_samples"], 3)
        self.assertEqual(metadata["n_classes"], 2)
        self.assertEqual(metadata["task_type"], "binary")
        self.assertEqual(metadata["positive_index"], 1)
        self.assertEqual(metadata["positive_label"], "yes")
        np.testing.assert_array_equal(metadata["classes"], ["no", "yes"])
        np.testing.assert_array_equal(
            metadata["class_names"], ["Negative", "Positive"]
        )
        np.testing.assert_array_equal(metadata["y_true"], ["no", "yes", "yes"])
        np.testing.assert_array_equal(metadata["y_pred"], ["no", "no", "yes"])
        np.testing.assert_array_equal(metadata["y_proba"], [0.1, 0.4, 0.9])

    def test_infers_multiclass_and_default_names(self) -> None:
        metadata = validate_evaluation_inputs(
            y_true=[2, 0, 1],
            y_pred=[2, 1, 1],
            y_proba=[
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.1, 0.8, 0.1],
            ],
            classes=[0, 1, 2],
        )

        self.assertEqual(metadata["task_type"], "multiclass")
        self.assertIsNone(metadata["positive_index"])
        np.testing.assert_array_equal(metadata["class_names"], ["0", "1", "2"])

    def test_binary_requires_explicit_positive_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive_label is required"):
            validate_evaluation_inputs(
                y_true=[0, 1],
                y_pred=[0, 1],
                y_proba=[0.1, 0.9],
                classes=[0, 1],
            )

    def test_rejects_explicit_binary_task_with_three_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires exactly two classes"):
            validate_evaluation_inputs(
                y_true=[0, 1, 2],
                y_pred=[0, 1, 2],
                y_proba=[
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                classes=[0, 1, 2],
                positive_label=1,
                task_type="binary",
            )

    def test_rejects_explicit_multiclass_task_with_two_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least three classes"):
            validate_evaluation_inputs(
                y_true=[0, 1],
                y_pred=[0, 1],
                y_proba=[[0.9, 0.1], [0.1, 0.9]],
                classes=[0, 1],
                task_type="multiclass",
            )

    def test_multiclass_rejects_positive_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "only valid for binary"):
            validate_evaluation_inputs(
                y_true=[0, 1, 2],
                y_pred=[0, 1, 2],
                y_proba=np.eye(3),
                classes=[0, 1, 2],
                positive_label=1,
                task_type="multiclass",
            )

    def test_rejects_unknown_task_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_type must be None"):
            validate_evaluation_inputs(
                y_true=[0, 1],
                y_pred=[0, 1],
                y_proba=[0.1, 0.9],
                classes=[0, 1],
                positive_label=1,
                task_type="classification",
            )


if __name__ == "__main__":
    unittest.main()
