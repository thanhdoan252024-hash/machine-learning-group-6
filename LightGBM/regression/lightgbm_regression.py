import numpy as np

from lightgbm_from_scratch.core.binning import create_quantile_bins, bin_matrix
from lightgbm_from_scratch.core.efb import build_efb_plan
from lightgbm_from_scratch.core.goss import goss_sample
from lightgbm_from_scratch.core.histogram import build_feature_histograms
from lightgbm_from_scratch.core.split import find_best_histogram_split, split_rows_by_bin
from lightgbm_from_scratch.core.tree import build_leafwise_tree, predict_tree_dict
from lightgbm_from_scratch.core.regularization import (
    leaf_value as core_leaf_value,
    regularized_score as core_regularized_score,
    split_gain as core_split_gain,
)
from lightgbm_from_scratch.objectives.regression import squared_error_gradient_hessian
from lightgbm_from_scratch.metrics import rmse as core_rmse


class LightGBMRegression:
    """Educational LightGBM-style regressor built without LightGBM itself."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        max_bins=32,
        top_rate=0.2,
        other_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_gain_to_split=0.0,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        use_goss=True,
        use_efb=True,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = max(2, int(num_leaves))
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_bins = max(2, int(max_bins))
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        self.min_data_in_leaf = max(1, int(min_data_in_leaf))
        self.feature_fraction = float(feature_fraction)
        self.use_goss = bool(use_goss)
        self.use_efb = bool(use_efb)
        self.base_prediction = 0.0
        self.trees = []
        self.feature_bins = None
        self.bin_thresholds = None
        self.feature_importances_ = None
        self.efb_bundles = []
        self.efb_applied = False
        self.efb_offsets = {}

    @staticmethod
    def _compute_gradient_hessian(y_true, predictions):
        """Tính gradient và hessian cho regression loss 0.5 * (pred - y)^2.

        Với loss bình phương, gradient = pred - y và hessian = 1.
        """
        return squared_error_gradient_hessian(y_true, predictions)

    def _prepare_bins(self, X):
        self.bin_thresholds = create_quantile_bins(
            X, self.max_bins, quantile_style="boundaries"
        )
        self.feature_bins = bin_matrix(
            X, self.bin_thresholds, missing_bin="last", max_bins=self.max_bins
        )
        if self.use_efb:
            self.efb_plan_ = build_efb_plan(
                X, self.feature_bins, missing_bin="last", max_bins=self.max_bins
            )
            self.efb_bundles = [list(group) for group in self.efb_plan_.groups]
            self.efb_applied = self.efb_plan_.applied
            self.efb_offsets = dict(self.efb_plan_.offsets)
        else:
            self.efb_plan_ = None
            self.efb_bundles = [[i] for i in range(X.shape[1])]
            self.efb_applied = False
            self.efb_offsets = {}

    def _find_efb_bundles(self, X):
        active = (~np.isnan(X)) & (X != 0)
        bundles = []
        for feature_index in range(X.shape[1]):
            placed = False
            for bundle in bundles:
                conflict = np.count_nonzero(
                    np.any(active[:, bundle], axis=1) & active[:, feature_index]
                )
                if conflict == 0:
                    bundle.append(feature_index)
                    placed = True
                    break
            if not placed:
                bundles.append([feature_index])
        return bundles

    def _histogram(self, row_indices, gradients, hessians):
        """Compatibility wrapper over the shared histogram engine."""
        row_indices = np.asarray(row_indices, dtype=int)
        features = np.arange(self.feature_bins.shape[1])
        histograms = build_feature_histograms(
            self.feature_bins, gradients, hessians, row_indices,
            np.ones(row_indices.size, dtype=float), features,
            efb_plan=self.efb_plan_, include_count=False, missing_bin=self.max_bins,
        )
        histogram = np.zeros(
            (self.feature_bins.shape[1], self.max_bins + 1, 2), dtype=float
        )
        for feature_index, feature_histogram in histograms.items():
            histogram[feature_index] = feature_histogram.T
        return histogram

    def _leaf_value(self, gradient_sum, hessian_sum):
        return core_leaf_value(
            gradient_sum, hessian_sum, self.reg_alpha, self.reg_lambda
        )

    def _regularized_score(self, gradient_sum, hessian_sum):
        return core_regularized_score(
            gradient_sum, hessian_sum, self.reg_alpha, self.reg_lambda
        )

    def _split_gain(self, parent_gradient, parent_hessian, left_gradient, left_hessian):
        right_gradient = parent_gradient - left_gradient
        right_hessian = parent_hessian - left_hessian
        return core_split_gain(
            left_gradient,
            left_hessian,
            right_gradient,
            right_hessian,
            self.reg_alpha,
            self.reg_lambda,
        )

    def _best_split(self, histogram, gradient_sum, hessian_sum):
        """Compatibility wrapper over the shared split search.

        The historical method reports gain after subtracting
        ``min_gain_to_split``; keep that contract for existing tests.
        """
        histograms = {
            feature: np.asarray(histogram[feature], dtype=float).T
            for feature in range(histogram.shape[0])
        }
        split = find_best_histogram_split(
            histograms,
            range(histogram.shape[0]),
            missing_bin=self.max_bins,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_gain=-np.inf,
            min_child_samples=1,
            min_child_weight=0.0,
        )
        if split is None:
            return None
        split = dict(split)
        split["gain"] -= self.min_gain_to_split
        return split

    def _split_rows(self, row_indices, split):
        left, right, _ = split_rows_by_bin(
            self.feature_bins, row_indices, split, missing_bin=self.max_bins
        )
        return left, right

    def _best_first_tree(self, rows, gradients, hessians, features=None):
        """Build a tree through the shared best-first tree engine."""
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(
                self.feature_bins.shape[1], dtype=float
            )
        if features is None:
            features = np.arange(self.feature_bins.shape[1])
        root, importance = build_leafwise_tree(
            self.feature_bins,
            gradients,
            hessians,
            np.asarray(rows, dtype=int),
            np.ones(len(rows), dtype=float),
            features=np.asarray(features, dtype=int),
            missing_bin=self.max_bins,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_data_in_leaf,
            min_child_weight=0.0,
            min_gain=self.min_gain_to_split,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            efb_plan=self.efb_plan_,
        )
        self._last_tree_importance = importance
        self.feature_importances_ += importance
        return root

    def _goss_sample(self, gradients, estimator_index):
        rng = np.random.default_rng(self.random_state + estimator_index)
        return goss_sample(
            gradients,
            self.top_rate,
            self.other_rate,
            rng,
            count_mode="floor",
            weight_mode="rate_ratio",
            full_weight_array=True,
        )

    def _bins_for_data(self, X):
        return bin_matrix(
            X, self.bin_thresholds, missing_bin="last", max_bins=self.max_bins
        )

    def _predict_tree_row(self, row_bins, node):
        if "split" not in node:
            return node["value"]
        split = node["split"]
        value = row_bins[split["feature"]]
        goes_left = value <= split["threshold_bin"]
        if value == self.max_bins:
            goes_left = split["missing_left"]
        child = node["left"] if goes_left else node["right"]
        return self._predict_tree_row(row_bins, child)

    def _predict_tree_binned(self, bins, tree):
        return predict_tree_dict(tree, bins, missing_bin=self.max_bins)

    def _predict_tree(self, X, tree):
        return self._predict_tree_binned(self._bins_for_data(X), tree)

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        early_stopping_rounds=None,
    ):
        """Fit the regressor with optional validation-based early stopping."""
        print("Đang tiến hành huấn luyện (fit)...")
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X_train phải là ma trận và có cùng số dòng với y_train.")
        if self.use_goss:
            if not 0 < self.top_rate < 1 or not 0 < self.other_rate < 1:
                raise ValueError("top_rate và other_rate phải thuộc khoảng (0, 1).")
            if self.top_rate + self.other_rate > 1:
                raise ValueError("top_rate + other_rate không được vượt quá 1.")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction phải thuộc (0, 1].")
        if self.reg_alpha < 0 or self.reg_lambda < 0:
            raise ValueError("reg_alpha và reg_lambda không được âm.")
        if self.min_data_in_leaf < 1:
            raise ValueError("min_data_in_leaf phải lớn hơn hoặc bằng 1.")
        if early_stopping_rounds is not None:
            early_stopping_rounds = int(early_stopping_rounds)
            if early_stopping_rounds < 1:
                raise ValueError("early_stopping_rounds phải >= 1.")
            if eval_set is None:
                raise ValueError("early stopping yêu cầu eval_set.")

        self._prepare_bins(X)
        self.base_prediction = float(y.mean())
        predictions = np.full(y.shape, self.base_prediction, dtype=float)
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        tree_importances = []
        self.evals_result_ = {"training": {"rmse": []}}

        X_val = y_val = val_bins = val_predictions = None
        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise ValueError("eval_set phải là tuple (X_val, y_val).")
            X_val = np.asarray(eval_set[0], dtype=float)
            y_val = np.asarray(eval_set[1], dtype=float).reshape(-1)
            if (
                X_val.ndim != 2
                or X_val.shape[1] != X.shape[1]
                or X_val.shape[0] != y_val.shape[0]
            ):
                raise ValueError("eval_set không khớp số feature/số dòng.")
            val_bins = self._bins_for_data(X_val)
            val_predictions = np.full(y_val.shape, self.base_prediction, dtype=float)
            self.evals_result_["validation"] = {"rmse": []}

        best_score = np.inf
        best_iteration = 0
        no_improvement = 0

        for estimator_index in range(self.n_estimators):
            gradients, hessians = self._compute_gradient_hessian(y, predictions)
            if self.use_goss:
                selected_rows, weights = self._goss_sample(gradients, estimator_index)
            else:
                selected_rows = np.arange(len(y), dtype=int)
                weights = np.ones(len(y), dtype=float)
            sampled_gradients = np.zeros_like(gradients)
            sampled_hessians = np.zeros_like(hessians)
            sampled_gradients[selected_rows] = gradients[selected_rows] * weights[selected_rows]
            sampled_hessians[selected_rows] = hessians[selected_rows] * weights[selected_rows]
            n_features = max(1, int(np.ceil(self.feature_fraction * X.shape[1])))
            feature_rng = np.random.default_rng(self.random_state + 100000 + estimator_index)
            features = feature_rng.choice(X.shape[1], n_features, replace=False)
            tree = self._best_first_tree(
                selected_rows, sampled_gradients, sampled_hessians, features=features
            )
            self.trees.append(tree)
            tree_importances.append(self._last_tree_importance.copy())
            predictions += self.learning_rate * self._predict_tree_binned(
                self.feature_bins, tree
            )
            self.evals_result_["training"]["rmse"].append(core_rmse(y, predictions))

            if X_val is not None:
                val_predictions += self.learning_rate * self._predict_tree_binned(
                    val_bins, tree
                )
                score = core_rmse(y_val, val_predictions)
                self.evals_result_["validation"]["rmse"].append(score)
                if score < best_score - 1e-12:
                    best_score = score
                    best_iteration = len(self.trees)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if (
                        early_stopping_rounds is not None
                        and no_improvement >= early_stopping_rounds
                    ):
                        break

        if X_val is None:
            best_iteration = len(self.trees)
            best_score = self.evals_result_["training"]["rmse"][-1]
        elif best_iteration == 0:
            best_iteration = len(self.trees)
            best_score = self.evals_result_["validation"]["rmse"][-1]

        if early_stopping_rounds is not None and best_iteration < len(self.trees):
            self.trees = self.trees[:best_iteration]
            self.feature_importances_ = np.sum(
                np.asarray(tree_importances[:best_iteration]), axis=0
            )

        self.best_iteration_ = int(best_iteration)
        self.best_score_ = float(best_score)
        self.n_estimators_ = len(self.trees)
        print("Huấn luyện hoàn tất!")
        return self

    def predict(self, X_test):
        print("Đang tiến hành dự đoán (predict)...")
        if self.bin_thresholds is None:
            raise RuntimeError("Cần gọi fit() trước khi predict().")
        X = np.asarray(X_test, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.bin_thresholds):
            raise ValueError("X_test phải có cùng số cột với dữ liệu huấn luyện.")
        predictions = np.full(X.shape[0], self.base_prediction, dtype=float)
        bins = self._bins_for_data(X)
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree_binned(bins, tree)
        return predictions

    def get_feature_importance(self):
        return self.feature_importances_.copy()
