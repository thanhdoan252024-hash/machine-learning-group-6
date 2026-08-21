"""Kiểm thử hợp đồng dữ liệu và split của pipeline Machine failure."""

import unittest

from classification.evaluation.run_machine_failure_evaluation import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    load_machine_failure_dataset,
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


if __name__ == "__main__":
    unittest.main()
