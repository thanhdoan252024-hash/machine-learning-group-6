"""Kiểm thử adapter dành riêng cho pipeline Machine failure của repo."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np

from classification.evaluation.adapter import (
    CLASS_NAME_BY_LABEL,
    POSITIVE_LABEL,
    evaluate_classification_outputs,
    evaluate_fitted_classifier,
)
from classification.lightgbm_classification import LightGBMClassification


class ClassificationOutputAdapterTests(unittest.TestCase):
    """Kiểm tra metadata repo được chuyển nguyên vẹn vào core runner."""

    @patch(
        "classification.evaluation.adapter."
        "run_classification_evaluation"
    )
    def test_uses_model_classes_and_label_based_names(self, runner: Mock) -> None:
        model = Mock()
        model.classes_ = np.array([0, 1])
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 1])
        y_proba = np.array(
            [
                [0.90, 0.10],
                [0.60, 0.40],
                [0.20, 0.80],
            ]
        )
        expected_result = {"metrics": {"accuracy": 2 / 3}}
        runner.return_value = expected_result

        result = evaluate_classification_outputs(
            model,
            y_true,
            y_pred,
            y_proba,
            Path("classification") / "evaluation" / "outputs",
            save_dpi=120,
        )

        self.assertIs(result, expected_result)
        runner.assert_called_once_with(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=[0, 1],
            class_names=[CLASS_NAME_BY_LABEL[0], CLASS_NAME_BY_LABEL[1]],
            positive_label=POSITIVE_LABEL,
            task_type="binary",
            output_dir=Path("classification") / "evaluation" / "outputs",
            save_dpi=120,
        )

    @patch(
        "classification.evaluation.adapter."
        "run_classification_evaluation"
    )
    def test_rejects_classes_with_positive_label_in_wrong_column(
        self,
        runner: Mock,
    ) -> None:
        model = Mock()
        model.classes_ = np.array([1, 0])

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            evaluate_classification_outputs(
                model,
                [0, 1],
                [0, 1],
                [[0.8, 0.2], [0.2, 0.8]],
                "classification/evaluation/outputs",
            )

        runner.assert_not_called()

    def test_rejects_model_without_fitted_classes(self) -> None:
        class UnfittedModel:
            pass

        with self.assertRaisesRegex(ValueError, "classes_"):
            evaluate_classification_outputs(
                UnfittedModel(),  # type: ignore[arg-type]
                [0, 1],
                [0, 1],
                [[0.8, 0.2], [0.2, 0.8]],
                "classification/evaluation/outputs",
            )

    @patch(
        "classification.evaluation.adapter."
        "evaluate_classification_outputs"
    )
    def test_fitted_model_api_calls_predict_and_predict_proba_once(
        self,
        output_adapter: Mock,
    ) -> None:
        model = Mock()
        model.classes_ = np.array([0, 1])
        model.predict.return_value = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        X = np.array([[1.0], [2.0]])
        y_true = np.array([0, 1])
        expected_result = {"metrics": {"accuracy": 1.0}}
        output_adapter.return_value = expected_result

        result = evaluate_fitted_classifier(
            model,
            X,
            y_true,
            "classification/evaluation/outputs",
            save_dpi=96,
        )

        self.assertIs(result, expected_result)
        model.predict.assert_called_once_with(X)
        model.predict_proba.assert_called_once_with(X)
        output_adapter.assert_called_once_with(
            model=model,
            y_true=y_true,
            y_pred=model.predict.return_value,
            y_proba=model.predict_proba.return_value,
            output_dir="classification/evaluation/outputs",
            save_dpi=96,
        )


class RealModelAdapterIntegrationTests(unittest.TestCase):
    """Chạy adapter bằng đúng model NumPy của repository."""

    def test_real_model_outputs_create_evaluation_artifacts(self) -> None:
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = np.array([0] * 6 + [1] * 6)
        model = LightGBMClassification(
            n_estimators=3,
            learning_rate=0.2,
            num_leaves=3,
            max_depth=2,
            max_bins=6,
            min_child_samples=1,
            top_rate=0.5,
            other_rate=0.5,
            random_state=42,
        ).fit(X, y)

        y_proba = model.predict_proba(X)
        y_pred = model.predict(X)
        self.assertEqual(y_proba.shape, (len(X), 2))
        np.testing.assert_allclose(y_proba.sum(axis=1), np.ones(len(X)))
        expected_prediction = model.classes_[
            (y_proba[:, 1] >= model.threshold).astype(int)
        ]
        np.testing.assert_array_equal(y_pred, expected_prediction)

        with TemporaryDirectory() as temporary_directory:
            output_dir = (
                Path(temporary_directory)
                / "classification"
                / "evaluation"
                / "outputs"
            )
            result = evaluate_fitted_classifier(
                model,
                X,
                y,
                output_dir,
                save_dpi=72,
            )

            self.assertEqual(result["metadata"]["classes"].tolist(), [0, 1])
            self.assertEqual(
                result["metadata"]["class_names"].tolist(),
                [CLASS_NAME_BY_LABEL[0], CLASS_NAME_BY_LABEL[1]],
            )
            self.assertEqual(result["metadata"]["positive_label"], POSITIVE_LABEL)
            self.assertTrue(result["manifest_path"].is_file())
            self.assertTrue(
                (output_dir / "predictions" / "predictions.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
