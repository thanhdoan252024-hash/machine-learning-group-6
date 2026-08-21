"""Kiểm thử hợp đồng dữ liệu và split của pipeline Machine failure."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd

import classification.evaluation.run_machine_failure_evaluation as pipeline_module

from classification.evaluation.run_machine_failure_evaluation import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    evaluate_machine_failure_splits,
    load_machine_failure_dataset,
    run_machine_failure_pipeline,
    split_machine_failure_dataset,
)


class MachineFailureDatasetTests(unittest.TestCase):
    """Giữ preprocessing của script đồng bộ với notebook và dataset thật."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.features, cls.target = load_machine_failure_dataset(DEFAULT_DATA_PATH)

    def test_dataset_contract(self) -> None:
        self.assertEqual(self.features.shape, (10_000, 6))
        self.assertEqual(tuple(self.features.columns), FEATURE_COLUMNS)
        self.assertFalse(self.features.isna().any().any())
        self.assertEqual(
            self.target.value_counts().sort_index().to_dict(),
            {0: 9_661, 1: 339},
        )

    def test_stratified_split_is_reproducible(self) -> None:
        first = split_machine_failure_dataset(self.features, self.target)
        second = split_machine_failure_dataset(self.features, self.target)

        X_train, X_test, y_train, y_test = first
        self.assertEqual((len(X_train), len(X_test)), (8_000, 2_000))
        self.assertEqual((len(y_train), len(y_test)), (8_000, 2_000))
        self.assertEqual(
            y_test.value_counts().sort_index().to_dict(),
            {0: 1_932, 1: 68},
        )
        self.assertTrue(X_train.index.equals(second[0].index))
        self.assertTrue(X_test.index.equals(second[1].index))
        self.assertTrue(y_train.index.equals(second[2].index))
        self.assertTrue(y_test.index.equals(second[3].index))


class MachineFailurePipelineTests(unittest.TestCase):
    """Kiểm tra orchestration train/test và lifecycle manifest gốc."""

    @staticmethod
    def _split_data() -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        X_train = pd.DataFrame({"feature": [10, 20, 30]}, index=[0, 1, 2])
        X_test = pd.DataFrame({"feature": [40, 50]}, index=[3, 4])
        y_train = pd.Series([0, 1, 0], index=X_train.index)
        y_test = pd.Series([1, 0], index=X_test.index)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _fake_evaluation(**kwargs: object) -> dict[str, object]:
        output_dir = Path(kwargs["output_dir"])
        split_name = output_dir.name
        generated_files = [
            "figures/summary.png",
            "tables/metrics.csv",
        ]
        for relative_path in generated_files:
            artifact_path = output_dir / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(split_name, encoding="utf-8")

        manifest_path = output_dir / "evaluation_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_type": "binary",
                    "generated_files": generated_files,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        accuracy = 0.75 if split_name == "train" else 0.5
        return {
            "metrics": {
                "accuracy": accuracy,
                "precision": accuracy,
                "recall": accuracy,
                "f1_score": accuracy,
                "roc_auc": accuracy,
            },
            "classification_report": f"{split_name}-report",
            "manifest_path": manifest_path,
        }

    @staticmethod
    def _mock_model() -> mock.Mock:
        model = mock.Mock()
        model.threshold = 0.5
        model.predict.side_effect = [
            np.array([0, 1, 0]),
            np.array([1, 0]),
        ]
        model.predict_proba.side_effect = [
            np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
            np.array([[0.1, 0.9], [0.6, 0.4]]),
        ]
        return model

    def test_pipeline_fits_once_and_evaluates_both_splits(self) -> None:
        X_train, X_test, y_train, y_test = self._split_data()
        features = pd.concat([X_train, X_test])
        target = pd.concat([y_train, y_test])
        model = self._mock_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            with (
                mock.patch.object(
                    pipeline_module,
                    "load_machine_failure_dataset",
                    return_value=(features, target),
                ),
                mock.patch.object(
                    pipeline_module,
                    "split_machine_failure_dataset",
                    return_value=(X_train, X_test, y_train, y_test),
                ),
                mock.patch.object(
                    pipeline_module,
                    "build_machine_failure_classifier",
                    return_value=model,
                ),
                mock.patch.object(
                    pipeline_module,
                    "evaluate_classification_outputs",
                    side_effect=self._fake_evaluation,
                ) as evaluate_mock,
            ):
                result = run_machine_failure_pipeline(output_dir=output_root)

            model.fit.assert_called_once_with(X_train, y_train)
            self.assertEqual(model.predict.call_count, 2)
            self.assertIs(model.predict.call_args_list[0].args[0], X_train)
            self.assertIs(model.predict.call_args_list[1].args[0], X_test)
            self.assertEqual(model.predict_proba.call_count, 2)
            self.assertIs(model.predict_proba.call_args_list[0].args[0], X_train)
            self.assertIs(model.predict_proba.call_args_list[1].args[0], X_test)

            self.assertEqual(evaluate_mock.call_count, 2)
            train_call, test_call = evaluate_mock.call_args_list
            self.assertEqual(train_call.kwargs["output_dir"], output_root / "train")
            self.assertEqual(test_call.kwargs["output_dir"], output_root / "test")
            np.testing.assert_array_equal(train_call.kwargs["y_true"], y_train)
            np.testing.assert_array_equal(test_call.kwargs["y_true"], y_test)
            np.testing.assert_array_equal(
                train_call.kwargs["y_pred"],
                np.array([0, 1, 0]),
            )
            np.testing.assert_array_equal(
                test_call.kwargs["y_pred"],
                np.array([1, 0]),
            )

            self.assertEqual(set(result["split_results"]), {"train", "test"})
            self.assertIs(
                result["metrics"],
                result["split_results"]["test"]["metrics"],
            )
            self.assertEqual(result["primary_reporting_split"], "test")
            self.assertEqual(
                result["manifest_path"],
                output_root / "evaluation_manifest.json",
            )
            self.assertEqual(
                result["split_results"]["test"]["manifest_path"],
                output_root / "test" / "evaluation_manifest.json",
            )

            manifest = json.loads(
                result["manifest_path"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["task_type"], "binary")
            self.assertEqual(manifest["model_fit_split"], "train")
            self.assertEqual(manifest["primary_reporting_split"], "test")
            self.assertEqual(manifest["model_threshold"], 0.5)
            self.assertEqual(manifest["splits"]["train"]["n_samples"], 3)
            self.assertEqual(manifest["splits"]["test"]["n_samples"], 2)
            self.assertEqual(
                manifest["splits"]["train"]["class_counts"],
                {"0": 2, "1": 1},
            )
            self.assertEqual(
                manifest["splits"]["test"]["class_counts"],
                {"0": 1, "1": 1},
            )
            self.assertEqual(
                manifest["splits"]["train"]["manifest_path"],
                "train/evaluation_manifest.json",
            )
            self.assertEqual(
                manifest["splits"]["test"]["manifest_path"],
                "test/evaluation_manifest.json",
            )
            self.assertEqual(
                manifest["generated_files"],
                sorted(
                    [
                        "train/evaluation_manifest.json",
                        "train/figures/summary.png",
                        "train/tables/metrics.csv",
                        "test/evaluation_manifest.json",
                        "test/figures/summary.png",
                        "test/tables/metrics.csv",
                    ]
                ),
            )

    def test_legacy_cleanup_only_deletes_valid_manifest_owned_files(self) -> None:
        X_train, X_test, y_train, y_test = self._split_data()
        model = self._mock_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_root = temp_root / "outputs"
            owned_figure = output_root / "figures" / "old.png"
            owned_table = output_root / "tables" / "old.csv"
            unowned_file = output_root / "figures" / "keep.png"
            outside_file = temp_root / "outside.txt"
            for path in (owned_figure, owned_table, unowned_file, outside_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep-or-delete", encoding="utf-8")

            legacy_manifest = {
                "schema_version": 1,
                "task_type": "binary",
                "generated_files": [
                    "figures/old.png",
                    "tables/old.csv",
                    "../outside.txt",
                    "train/evaluation_manifest.json",
                ],
            }
            (output_root / "evaluation_manifest.json").write_text(
                json.dumps(legacy_manifest, indent=2) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                pipeline_module,
                "evaluate_classification_outputs",
                side_effect=self._fake_evaluation,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = evaluate_machine_failure_splits(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        output_dir=output_root,
                    )

            self.assertFalse(owned_figure.exists())
            self.assertFalse(owned_table.exists())
            self.assertTrue(unowned_file.is_file())
            self.assertTrue(outside_file.is_file())
            self.assertTrue(
                (output_root / "train" / "evaluation_manifest.json").is_file()
            )
            self.assertTrue(any("không an toàn" in str(item.message) for item in caught))
            root_manifest = json.loads(
                result["manifest_path"].read_text(encoding="utf-8")
            )
            self.assertEqual(root_manifest["schema_version"], 2)
            self.assertNotIn("figures/keep.png", root_manifest["generated_files"])

    def test_child_failure_preserves_legacy_manifest_and_files(self) -> None:
        X_train, X_test, y_train, y_test = self._split_data()
        model = self._mock_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            legacy_file = output_root / "figures" / "legacy.png"
            legacy_file.parent.mkdir(parents=True, exist_ok=True)
            legacy_file.write_text("legacy", encoding="utf-8")
            legacy_manifest_path = output_root / "evaluation_manifest.json"
            legacy_manifest_text = json.dumps(
                {
                    "schema_version": 1,
                    "task_type": "binary",
                    "generated_files": ["figures/legacy.png"],
                },
                indent=2,
            ) + "\n"
            legacy_manifest_path.write_text(
                legacy_manifest_text,
                encoding="utf-8",
            )

            def fail_on_test(**kwargs: object) -> dict[str, object]:
                if Path(kwargs["output_dir"]).name == "test":
                    raise RuntimeError("test evaluation failed")
                return self._fake_evaluation(**kwargs)

            with mock.patch.object(
                pipeline_module,
                "evaluate_classification_outputs",
                side_effect=fail_on_test,
            ):
                with self.assertRaisesRegex(RuntimeError, "test evaluation failed"):
                    evaluate_machine_failure_splits(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        output_dir=output_root,
                    )

            self.assertEqual(
                legacy_manifest_path.read_text(encoding="utf-8"),
                legacy_manifest_text,
            )
            self.assertTrue(legacy_file.is_file())

    def test_invalid_or_incomplete_child_manifest_never_replaces_root(self) -> None:
        X_train, X_test, y_train, y_test = self._split_data()

        for failure_mode in ("not-a-mapping", "missing-artifact"):
            with self.subTest(failure_mode=failure_mode):
                model = self._mock_model()
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_root = Path(temp_dir) / "outputs"
                    root_manifest = output_root / "evaluation_manifest.json"
                    root_manifest.parent.mkdir(parents=True, exist_ok=True)
                    legacy_text = json.dumps(
                        {
                            "schema_version": 1,
                            "task_type": "binary",
                            "generated_files": [],
                        },
                        indent=2,
                    ) + "\n"
                    root_manifest.write_text(legacy_text, encoding="utf-8")

                    def invalid_evaluation(**kwargs: object) -> dict[str, object]:
                        split_root = Path(kwargs["output_dir"])
                        split_root.mkdir(parents=True, exist_ok=True)
                        child_manifest = split_root / "evaluation_manifest.json"
                        if failure_mode == "not-a-mapping":
                            child_manifest.write_text("[]\n", encoding="utf-8")
                        else:
                            child_manifest.write_text(
                                json.dumps(
                                    {
                                        "schema_version": 1,
                                        "task_type": "binary",
                                        "generated_files": ["missing.csv"],
                                    }
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                        return {
                            "metrics": {},
                            "manifest_path": child_manifest,
                        }

                    with mock.patch.object(
                        pipeline_module,
                        "evaluate_classification_outputs",
                        side_effect=invalid_evaluation,
                    ):
                        with self.assertRaises(ValueError):
                            evaluate_machine_failure_splits(
                                model,
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                output_dir=output_root,
                            )

                    self.assertEqual(
                        root_manifest.read_text(encoding="utf-8"),
                        legacy_text,
                    )

    def test_cleanup_error_is_nonfatal_after_atomic_manifest_replace(self) -> None:
        X_train, X_test, y_train, y_test = self._split_data()
        model = self._mock_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "outputs"
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "task_type": "binary",
                        "generated_files": ["legacy.csv"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    pipeline_module,
                    "evaluate_classification_outputs",
                    side_effect=self._fake_evaluation,
                ),
                mock.patch.object(
                    pipeline_module,
                    "_cleanup_legacy_generated_files",
                    side_effect=OSError("locked legacy file"),
                ),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                result = evaluate_machine_failure_splits(
                    model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    output_dir=output_root,
                )

            manifest = json.loads(
                result["manifest_path"].read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue(
                any("cleanup legacy thất bại" in str(item.message) for item in caught)
            )

    def test_cli_prints_train_and_test_metrics(self) -> None:
        metrics = {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "roc_auc": 0.95,
        }
        result = {
            "split_results": {
                "train": {"metrics": metrics},
                "test": {"metrics": metrics},
            },
            "pipeline_metadata": {
                "n_train": 8_000,
                "n_test": 2_000,
                "output_dir": Path("classification/evaluation/outputs"),
            },
        }
        output = StringIO()
        with (
            mock.patch.object(
                pipeline_module,
                "run_machine_failure_pipeline",
                return_value=result,
            ),
            redirect_stdout(output),
        ):
            pipeline_module.main()

        printed = output.getvalue()
        self.assertIn("Train: 8000 samples", printed)
        self.assertIn("Test: 2000 samples", printed)


if __name__ == "__main__":
    unittest.main()
