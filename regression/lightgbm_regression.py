import numpy as np

class LightGBMRegression:
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1, random_state=42, max_bins=32):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_bins = max_bins
        self.base_prediction = 0.0
        self.trees = []
        self.feature_medians = None

    def _best_split(self, X, residuals, row_indices):
        best_split = None
        parent_error = np.sum(residuals[row_indices] ** 2)

        for feature_index in range(X.shape[1]):
            values = X[row_indices, feature_index]
            unique_values = np.unique(values)
            if unique_values.size < 2:
                continue
            if unique_values.size > self.max_bins:
                positions = np.linspace(0, unique_values.size - 1, self.max_bins, dtype=int)
                unique_values = unique_values[positions]
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = values <= threshold
                left_indices = row_indices[left_mask]
                right_indices = row_indices[~left_mask]
                if left_indices.size == 0 or right_indices.size == 0:
                    continue
                left_error = np.sum((residuals[left_indices] - residuals[left_indices].mean()) ** 2)
                right_error = np.sum((residuals[right_indices] - residuals[right_indices].mean()) ** 2)
                gain = parent_error - left_error - right_error
                if best_split is None or gain > best_split["gain"]:
                    best_split = {
                        "feature": feature_index,
                        "threshold": threshold,
                        "gain": gain,
                        "left": left_indices,
                        "right": right_indices,
                    }
        return best_split

    def _build_tree(self, X, residuals, row_indices, depth, leaf_count):
        node_value = float(residuals[row_indices].mean())
        node = {"value": node_value}
        if leaf_count[0] >= self.num_leaves:
            return node
        if self.max_depth != -1 and depth >= self.max_depth:
            return node
        if row_indices.size < 4:
            return node

        split = self._best_split(X, residuals, row_indices)
        if split is None or split["gain"] <= 0:
            return node

        leaf_count[0] += 1
        node.update({
            "feature": split["feature"],
            "threshold": split["threshold"],
            "left": self._build_tree(X, residuals, split["left"], depth + 1, leaf_count),
            "right": self._build_tree(X, residuals, split["right"], depth + 1, leaf_count),
        })
        return node

    def _predict_tree_row(self, row, node):
        if "feature" not in node:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_tree_row(row, node["left"])
        return self._predict_tree_row(row, node["right"])

    def _predict_tree(self, X, tree):
        return np.array([self._predict_tree_row(row, tree) for row in X])

    def fit(self, X_train, y_train):
        print("Đang tiến hành huấn luyện (fit)...")
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X_train phải là ma trận và có cùng số dòng với y_train.")
        self.feature_medians = np.nanmedian(X, axis=0)
        missing_values = np.isnan(X)
        X[missing_values] = np.take(self.feature_medians, np.where(missing_values)[1])
        self.base_prediction = float(y.mean())
        predictions = np.full(y.shape, self.base_prediction, dtype=float)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = self._build_tree(X, residuals, np.arange(X.shape[0]), 0, [1])
            self.trees.append(tree)
            predictions += self.learning_rate * self._predict_tree(X, tree)
        print("Huấn luyện hoàn tất!")

    def predict(self, X_test):
        print("Đang tiến hành dự đoán (predict)...")
        if self.feature_medians is None:
            raise RuntimeError("Cần gọi fit() trước khi predict().")
        X = np.asarray(X_test, dtype=float).copy()
        if X.ndim != 2 or X.shape[1] != self.feature_medians.shape[0]:
            raise ValueError("X_test phải có cùng số cột với dữ liệu huấn luyện.")
        missing_values = np.isnan(X)
        X[missing_values] = np.take(self.feature_medians, np.where(missing_values)[1])
        y_pred = np.full(X.shape[0], self.base_prediction, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * self._predict_tree(X, tree)
        return y_pred
