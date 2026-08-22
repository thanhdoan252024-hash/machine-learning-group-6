import heapq

import numpy as np


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
        self.base_prediction = 0.0
        self.trees = []
        self.feature_bins = None
        self.bin_thresholds = None
        self.feature_importances_ = None
        self.efb_bundles = []
        self.efb_applied = False
        self.efb_offsets = {}
        self.feature_missing = None

    def _prepare_bins(self, X):
        self.bin_thresholds = []
        bin_matrix = np.full(X.shape, self.max_bins, dtype=np.int32)
        for feature_index in range(X.shape[1]):
            values = X[:, feature_index]
            observed = values[~np.isnan(values)]
            if observed.size == 0:
                thresholds = np.array([], dtype=float)
            else:
                quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
                thresholds = np.unique(np.quantile(observed, quantiles))
            self.bin_thresholds.append(thresholds)
            valid = ~np.isnan(values)
            bin_matrix[valid, feature_index] = np.searchsorted(
                thresholds, values[valid], side="right"
            )
        self.feature_bins = bin_matrix
        self.efb_bundles = self._find_efb_bundles(X)
        self.efb_applied = any(len(bundle) > 1 for bundle in self.efb_bundles)
        self.feature_missing = np.isnan(X) | (X == 0) if self.efb_applied else np.isnan(X)
        self.efb_offsets = {
            feature_index: position * (self.max_bins + 1)
            for bundle in self.efb_bundles
            for position, feature_index in enumerate(bundle)
        }

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
        histogram = np.zeros((self.feature_bins.shape[1], self.max_bins + 1, 2))
        if self.efb_applied:
            for bundle in self.efb_bundles:
                bundle_width = len(bundle) * (self.max_bins + 1) + 1
                bundled_bins = np.zeros(row_indices.size, dtype=np.int32)
                for feature_index in bundle:
                    bins = self.feature_bins[row_indices, feature_index]
                    active = bins != self.max_bins
                    bundled_bins[active] = (
                        self.efb_offsets[feature_index] + bins[active] + 1
                    )
                bundled_histogram = np.zeros((bundle_width, 2))
                np.add.at(
                    bundled_histogram[:, 0],
                    bundled_bins,
                    gradients[row_indices],
                )
                np.add.at(
                    bundled_histogram[:, 1],
                    bundled_bins,
                    hessians[row_indices],
                )
                for feature_index in bundle:
                    offset = self.efb_offsets[feature_index]
                    histogram[feature_index, 0, :] = np.array([
                        gradients[row_indices][self.feature_missing[row_indices, feature_index]].sum(),
                        hessians[row_indices][self.feature_missing[row_indices, feature_index]].sum(),
                    ])
                    histogram[feature_index, 1:, :] = bundled_histogram[
                        offset + 1:offset + self.max_bins + 1, :
                    ]
            return histogram
        for feature_index in range(self.feature_bins.shape[1]):
            bins = self.feature_bins[row_indices, feature_index]
            np.add.at(histogram[feature_index, :, 0], bins, gradients[row_indices])
            np.add.at(histogram[feature_index, :, 1], bins, hessians[row_indices])
        return histogram

    def _leaf_value(self, gradient_sum, hessian_sum):
        gradient_sum = np.sign(gradient_sum) * max(
            abs(gradient_sum) - self.reg_alpha, 0.0
        )
        return -gradient_sum / (hessian_sum + self.reg_lambda)

    def _regularized_score(self, gradient_sum, hessian_sum):
        gradient_sum = np.sign(gradient_sum) * max(
            abs(gradient_sum) - self.reg_alpha, 0.0
        )
        return gradient_sum**2 / (hessian_sum + self.reg_lambda)

    def _split_gain(self, parent_gradient, parent_hessian, left_gradient, left_hessian):
        right_gradient = parent_gradient - left_gradient
        right_hessian = parent_hessian - left_hessian
        return 0.5 * (
            self._regularized_score(left_gradient, left_hessian)
            + self._regularized_score(right_gradient, right_hessian)
            - self._regularized_score(parent_gradient, parent_hessian)
        )

    def _best_split(self, histogram, gradient_sum, hessian_sum):
        best_split = None
        for feature_index in range(histogram.shape[0]):
            feature_histogram = histogram[feature_index]
            missing_gradient, missing_hessian = feature_histogram[self.max_bins]
            cumulative_gradient = 0.0
            cumulative_hessian = 0.0
            for threshold_bin in range(self.max_bins):
                cumulative_gradient += feature_histogram[threshold_bin, 0]
                cumulative_hessian += feature_histogram[threshold_bin, 1]
                if cumulative_hessian <= 0:
                    continue

                for missing_left in (True, False):
                    left_gradient = cumulative_gradient
                    left_hessian = cumulative_hessian
                    if missing_left:
                        left_gradient += missing_gradient
                        left_hessian += missing_hessian
                    if left_hessian >= hessian_sum:
                        continue
                    gain = self._split_gain(
                        gradient_sum, hessian_sum, left_gradient, left_hessian
                    ) - self.min_gain_to_split
                    if best_split is None or gain > best_split["gain"]:
                        best_split = {
                            "feature": feature_index,
                            "threshold_bin": threshold_bin,
                            "missing_left": missing_left,
                            "gain": gain,
                        }
        return best_split

    def _split_rows(self, row_indices, split):
        bins = self.feature_bins[row_indices, split["feature"]]
        is_missing = self.feature_missing[row_indices, split["feature"]]
        goes_left = bins <= split["threshold_bin"]
        if split["missing_left"]:
            goes_left = goes_left | is_missing
        else:
            goes_left = goes_left & ~is_missing
        return row_indices[goes_left], row_indices[~goes_left]

    def _best_first_tree(self, rows, gradients, hessians):
        root_histogram = self._histogram(rows, gradients, hessians)
        root_gradient = gradients[rows].sum()
        root_hessian = hessians[rows].sum()
        root = {
            "value": self._leaf_value(root_gradient, root_hessian),
            "rows": rows,
            "histogram": root_histogram,
            "gradient_sum": root_gradient,
            "hessian_sum": root_hessian,
            "depth": 0,
        }
        leaves = [root]
        queue = []
        counter = 0

        def add_candidate(leaf):
            nonlocal counter
            if self.max_depth != -1 and leaf["depth"] >= self.max_depth:
                return
            split = self._best_split(
                leaf["histogram"], leaf["gradient_sum"], leaf["hessian_sum"]
            )
            if split is not None and split["gain"] > 0:
                heapq.heappush(queue, (-split["gain"], counter, leaf, split))
                counter += 1

        add_candidate(root)
        while queue and len(leaves) < self.num_leaves:
            _, _, leaf, split = heapq.heappop(queue)
            if "split" in leaf:
                continue
            left_rows, right_rows = self._split_rows(leaf["rows"], split)
            if left_rows.size == 0 or right_rows.size == 0:
                continue

            if left_rows.size <= right_rows.size:
                left_histogram = self._histogram(left_rows, gradients, hessians)
                right_histogram = leaf["histogram"] - left_histogram
            else:
                right_histogram = self._histogram(right_rows, gradients, hessians)
                left_histogram = leaf["histogram"] - right_histogram

            left_gradient = gradients[left_rows].sum()
            left_hessian = hessians[left_rows].sum()
            right_gradient = gradients[right_rows].sum()
            right_hessian = hessians[right_rows].sum()
            left = {
                "value": self._leaf_value(left_gradient, left_hessian),
                "rows": left_rows,
                "histogram": left_histogram,
                "gradient_sum": left_gradient,
                "hessian_sum": left_hessian,
                "depth": leaf["depth"] + 1,
            }
            right = {
                "value": self._leaf_value(right_gradient, right_hessian),
                "rows": right_rows,
                "histogram": right_histogram,
                "gradient_sum": right_gradient,
                "hessian_sum": right_hessian,
                "depth": leaf["depth"] + 1,
            }
            leaf["split"] = split
            leaf["left"] = left
            leaf["right"] = right
            leaves.remove(leaf)
            leaves.extend((left, right))
            self.feature_importances_[split["feature"]] += split["gain"]
            add_candidate(left)
            add_candidate(right)

        return root

    def _goss_sample(self, gradients, estimator_index):
        row_count = gradients.size
        top_count = max(1, int(row_count * self.top_rate))
        other_count = max(1, int(row_count * self.other_rate))
        order = np.argsort(np.abs(gradients))[::-1]
        top_rows = order[:top_count]
        remaining = order[top_count:]
        rng = np.random.default_rng(self.random_state + estimator_index)
        if remaining.size > other_count:
            other_rows = rng.choice(remaining, size=other_count, replace=False)
        else:
            other_rows = remaining
        selected = np.concatenate((top_rows, other_rows))
        weights = np.ones(row_count, dtype=float)
        if other_rows.size:
            weights[other_rows] = (1.0 - self.top_rate) / max(self.other_rate, 1e-12)
        return selected, weights

    def _bins_for_data(self, X):
        bins = np.full(X.shape, self.max_bins, dtype=np.int32)
        for feature_index, thresholds in enumerate(self.bin_thresholds):
            valid = ~np.isnan(X[:, feature_index])
            bins[valid, feature_index] = np.searchsorted(
                thresholds, X[valid, feature_index], side="right"
            )
        if self.efb_applied:
            bins[self.feature_missing_for_data(X)] = self.max_bins
        return bins

    def feature_missing_for_data(self, X):
        return np.isnan(X) | (X == 0)

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

    def _predict_tree(self, X, tree):
        bins = self._bins_for_data(X)
        return np.array([self._predict_tree_row(row, tree) for row in bins])

    def fit(self, X_train, y_train):
        print("Đang tiến hành huấn luyện (fit)...")
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X_train phải là ma trận và có cùng số dòng với y_train.")
        if not 0 < self.top_rate < 1 or not 0 < self.other_rate < 1:
            raise ValueError("top_rate và other_rate phải thuộc khoảng (0, 1).")
        if self.top_rate + self.other_rate > 1:
            raise ValueError("top_rate + other_rate không được vượt quá 1.")
        if self.reg_alpha < 0 or self.reg_lambda < 0:
            raise ValueError("reg_alpha và reg_lambda không được âm.")
        self._prepare_bins(X)
        self.base_prediction = float(y.mean())
        predictions = np.full(y.shape, self.base_prediction, dtype=float)
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)

        for estimator_index in range(self.n_estimators):
            gradients = predictions - y
            hessians = np.ones_like(gradients)
            selected_rows, weights = self._goss_sample(gradients, estimator_index)
            sampled_gradients = np.zeros_like(gradients)
            sampled_hessians = np.zeros_like(hessians)
            sampled_gradients[selected_rows] = gradients[selected_rows] * weights[selected_rows]
            sampled_hessians[selected_rows] = hessians[selected_rows] * weights[selected_rows]
            tree = self._best_first_tree(
                selected_rows, sampled_gradients, sampled_hessians
            )
            self.trees.append(tree)
            predictions += self.learning_rate * self._predict_tree(X, tree)
        print("Huấn luyện hoàn tất!")

    def predict(self, X_test):
        print("Đang tiến hành dự đoán (predict)...")
        if self.bin_thresholds is None:
            raise RuntimeError("Cần gọi fit() trước khi predict().")
        X = np.asarray(X_test, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.bin_thresholds):
            raise ValueError("X_test phải có cùng số cột với dữ liệu huấn luyện.")
        predictions = np.full(X.shape[0], self.base_prediction, dtype=float)
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(X, tree)
        return predictions

    def get_feature_importance(self):
        return self.feature_importances_.copy()
