
from dataclasses import dataclass
import numpy as np

from lightgbm_from_scratch.core.binning import create_quantile_bins, bin_matrix
from lightgbm_from_scratch.core.efb import build_efb_plan
from lightgbm_from_scratch.core.goss import goss_sample as core_goss_sample
from lightgbm_from_scratch.core.histogram import (
    build_feature_histograms as core_build_feature_histograms,
    histogram_subtraction as core_histogram_subtraction,
)
from lightgbm_from_scratch.core.split import (
    find_best_histogram_split,
    split_rows_by_bin,
)
from lightgbm_from_scratch.core.tree import build_leafwise_tree
from lightgbm_from_scratch.core.regularization import (
    leaf_value as core_leaf_value,
    soft_threshold as core_soft_threshold,
    split_gain as core_split_gain,
)
from lightgbm_from_scratch.metrics import binary_logloss
from lightgbm_from_scratch.objectives.binary import (
    binary_logloss_gradient_hessian,
    sigmoid as core_sigmoid,
)


def sigmoid(x):
    """Chuyển raw score thành xác suất."""
    return core_sigmoid(x)


def calculate_gradients(y, raw_score):
    """Tính gradient và Hessian của binary log-loss."""
    return binary_logloss_gradient_hessian(y, raw_score)


def encode_categorical_features(X, categorical_features, category_maps=None):
    """Mã hóa feature phân loại thành số; category chưa gặp được gán NaN.

    category_maps bằng None khi fit để học ánh xạ, và được truyền lại khi
    predict để bảo đảm train/test dùng cùng một mã category.
    """
    X = np.asarray(X, dtype=object)
    if X.ndim != 2:
        raise ValueError("X phải là ma trận hai chiều.")
    categorical_features = set(categorical_features or [])
    fitting = category_maps is None
    maps = {} if fitting else category_maps
    encoded = np.empty(X.shape, dtype=float)

    for j in range(X.shape[1]):
        if j not in categorical_features:
            encoded[:, j] = X[:, j].astype(float)
            continue
        keys = [
            None if (value is None or
                     isinstance(value, (float, np.floating)) and np.isnan(value))
            else (type(value).__name__, str(value))
            for value in X[:, j]
        ]
        if fitting:
            unique_keys = dict.fromkeys(key for key in keys if key is not None)
            maps[j] = {key: code for code, key in enumerate(unique_keys)}
        encoded[:, j] = [maps[j].get(key, np.nan) for key in keys]
    return encoded, maps


def create_bins(X, max_bins):
    """Tạo các ngưỡng quantile; NaN được dành riêng bin số 0."""
    return create_quantile_bins(X, max_bins, quantile_style="centers")


def bin_data(X, cuts):
    """Nén đặc trưng thành bin; bin 0 chỉ dành riêng cho missing value."""
    return bin_matrix(X, cuts, missing_bin="zero")


def exclusive_feature_bundling(X, max_conflict_rate=0.0, bundles=None):
    """Gộp các feature thưa gần như loại trừ nhau theo kỹ thuật EFB.

    Mỗi feature trong bundle dùng một vùng bin riêng nên không bị lẫn giá trị.
    Khi predict, truyền lại bundles đã học ở fit để biến đổi giống nhau.
    """
    if bundles is None:
        groups = []
        active = X != 0
        for feature in range(X.shape[1]):
            for group in groups:
                occupied = np.any(active[:, group], axis=1)
                conflict = np.mean(occupied & active[:, feature])
                if conflict <= max_conflict_rate:
                    group.append(feature)
                    break
            else:
                groups.append([feature])

        bundles = []
        for group in groups:
            offset, definition = 0, []
            for feature in group:
                definition.append((feature, offset))
                offset += int(X[:, feature].max())
            bundles.append(definition)

    bundled = np.zeros((len(X), len(bundles)), dtype=np.int32)
    for bundle_index, definition in enumerate(bundles):
        for feature, offset in definition:
            values = X[:, feature]
            bundled[:, bundle_index] += np.where(values > 0, values + offset, 0)
    return bundled, bundles


def histogram_subtraction(parent_histogram, child_histogram):
    """Compatibility wrapper over the shared histogram subtraction core."""
    return core_histogram_subtraction(parent_histogram, child_histogram)


def build_histograms(
    X, gradients, hessians, rows, weights, features, efb_plan=None
):
    """Dựng histogram [gradient, Hessian, count] qua shared core."""
    return core_build_feature_histograms(
        X, gradients, hessians, rows, weights, features,
        efb_plan=efb_plan, include_count=True, missing_bin=0,
    )


def goss_sample(gradients, top_rate, other_rate, rng):
    """Giữ gradient lớn, lấy mẫu gradient nhỏ và bù trọng số (GOSS)."""
    return core_goss_sample(
        gradients,
        top_rate,
        other_rate,
        rng,
        count_mode="ceil",
        weight_mode="sample_ratio",
        full_weight_array=False,
    )


def soft_threshold(g, alpha):
    """Áp dụng regularization L1 lên tổng gradient."""
    return core_soft_threshold(g, alpha)


def leaf_value(g, h, alpha, reg_lambda):
    """Giá trị leaf tối ưu theo bước Newton."""
    return core_leaf_value(g, h, alpha, reg_lambda)


def split_gain(lg, lh, rg, rh, alpha, reg_lambda):
    """Mức giảm loss khi chia một node thành hai leaf."""
    return core_split_gain(lg, lh, rg, rh, alpha, reg_lambda)


@dataclass
class TreeNode:
    value: float
    feature: int = None
    threshold: int = None
    default_left: bool = True
    left: object = None
    right: object = None


def find_best_split(X, rows, histograms, params):
    """Compatibility wrapper over the shared split-search core."""
    split = find_best_histogram_split(
        histograms,
        params["features"],
        missing_bin=0,
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_gain=params["min_split_gain"],
        min_child_samples=params["min_child_samples"],
        min_child_weight=params["min_child_weight"],
    )
    if split is None:
        return None, params["min_split_gain"]
    _, _, mask = split_rows_by_bin(X, rows, split, missing_bin=0)
    legacy = (
        int(split["feature"]),
        int(split["threshold_bin"]),
        bool(split["missing_left"]),
        mask,
    )
    return legacy, float(split["gain"])


def _dict_to_tree_node(node):
    result = TreeNode(value=float(node["value"]))
    if "split" in node:
        split = node["split"]
        result.feature = int(split["feature"])
        result.threshold = int(split["threshold_bin"])
        result.default_left = bool(split["missing_left"])
        result.left = _dict_to_tree_node(node["left"])
        result.right = _dict_to_tree_node(node["right"])
    return result


def _build_tree_and_importance(X, gradients, hessians, rows, weights, params):
    root, importance = build_leafwise_tree(
        X, gradients, hessians, rows, weights,
        features=params["features"],
        missing_bin=0,
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        min_child_weight=params["min_child_weight"],
        min_gain=params["min_split_gain"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        efb_plan=params.get("efb_plan"),
    )
    return _dict_to_tree_node(root), importance


def build_tree(X, gradients, hessians, rows, weights, params):
    """Xây cây leaf-wise bằng shared core và giữ API TreeNode cũ."""
    tree, _ = _build_tree_and_importance(
        X, gradients, hessians, rows, weights, params
    )
    return tree


def predict_tree(tree, X):
    """Dự đoán raw score bằng vectorized node-wise routing."""
    X = np.asarray(X)
    result = np.empty(len(X), dtype=float)
    stack = [(tree, np.arange(len(X), dtype=int))]
    while stack:
        node, rows = stack.pop()
        if rows.size == 0:
            continue
        if node.feature is None:
            result[rows] = node.value
            continue
        values = X[rows, node.feature]
        missing = values == 0
        threshold_mask = values <= node.threshold
        if node.default_left:
            left_mask = missing | (~missing & threshold_mask)
        else:
            left_mask = (~missing) & threshold_mask
        stack.append((node.right, rows[~left_mask]))
        stack.append((node.left, rows[left_mask]))
    return result


class LightGBMClassification:

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        max_bins=255,
        min_child_samples=20,
        min_child_weight=1e-3,
        min_split_gain=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        top_rate=0.2,
        other_rate=0.1,
        feature_fraction=1.0,
        categorical_features=None,
        max_conflict_rate=0.0,
        threshold=0.5,
        random_state=None,
        use_goss=True,
        use_efb=True,
        scale_pos_weight=1.0,
    ):
        for name, value in locals().copy().items():
            if name != "self":
                setattr(self, name, value)
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Huấn luyện ensemble với optional validation-based early stopping."""
        X, self.category_maps_ = encode_categorical_features(
            X, self.categorical_features)
        original_y = np.asarray(y)
        if (
            X.ndim != 2
            or original_y.ndim != 1
            or len(X) != len(original_y)
            or np.isinf(X).any()
        ):
            raise ValueError("X hoặc y không hợp lệ.")
        self.classes_, y = np.unique(original_y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Mô hình chỉ hỗ trợ đúng hai lớp.")
        if self.use_goss and not (0 < self.top_rate < 1 and 0 < self.other_rate < 1
                and self.top_rate + self.other_rate <= 1):
            raise ValueError("Tỉ lệ GOSS không hợp lệ.")
        if self.scale_pos_weight <= 0:
            raise ValueError("scale_pos_weight phải > 0.")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction phải thuộc (0, 1].")
        if early_stopping_rounds is not None:
            early_stopping_rounds = int(early_stopping_rounds)
            if early_stopping_rounds < 1:
                raise ValueError("early_stopping_rounds phải >= 1.")
            if eval_set is None:
                raise ValueError("early stopping yêu cầu eval_set.")

        self.n_features_in_ = X.shape[1]
        self.bin_thresholds_ = create_bins(X, self.max_bins)
        X_bin = bin_data(X, self.bin_thresholds_)
        if self.use_efb:
            self.efb_plan_ = build_efb_plan(X, X_bin, missing_bin="zero")
            self.feature_bundles_ = [list(group) for group in self.efb_plan_.groups]
            self.efb_applied_ = self.efb_plan_.applied
        else:
            self.efb_plan_ = None
            self.feature_bundles_ = [[i] for i in range(X.shape[1])]
            self.efb_applied_ = False
        positive_rate = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        self.init_score_ = np.log(positive_rate / (1 - positive_rate))
        raw_score = np.full(len(y), self.init_score_, dtype=float)
        self.trees_, rng = [], np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        tree_importances = []
        self.evals_result_ = {"training": {"binary_logloss": []}}

        X_val_bin = y_val = val_score_raw = None
        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise ValueError("eval_set phải là tuple (X_val, y_val).")
            X_val, _ = encode_categorical_features(
                eval_set[0], self.categorical_features, self.category_maps_
            )
            raw_y_val = np.asarray(eval_set[1])
            if (
                X_val.ndim != 2
                or X_val.shape[1] != X.shape[1]
                or raw_y_val.ndim != 1
                or len(X_val) != len(raw_y_val)
                or np.isinf(X_val).any()
            ):
                raise ValueError("eval_set không hợp lệ.")
            class_to_index = {label: index for index, label in enumerate(self.classes_)}
            try:
                y_val = np.asarray([class_to_index[label] for label in raw_y_val], dtype=int)
            except KeyError as exc:
                raise ValueError("eval_set chứa nhãn chưa xuất hiện trong train.") from exc
            X_val_bin = bin_data(X_val, self.bin_thresholds_)
            val_score_raw = np.full(len(y_val), self.init_score_, dtype=float)
            self.evals_result_["validation"] = {"binary_logloss": []}

        best_score = np.inf
        best_iteration = 0
        no_improvement = 0

        for _ in range(self.n_estimators):
            gradients, hessians = calculate_gradients(y, raw_score)
            if self.scale_pos_weight != 1.0:
                positive_mask = y == 1
                gradients = gradients.copy()
                hessians = hessians.copy()
                gradients[positive_mask] *= self.scale_pos_weight
                hessians[positive_mask] *= self.scale_pos_weight
            if self.use_goss:
                rows, weights = goss_sample(
                    gradients, self.top_rate, self.other_rate, rng)
            else:
                rows = np.arange(len(y), dtype=int)
                weights = np.ones(len(y), dtype=float)
            n_features = max(1, int(np.ceil(self.feature_fraction * X_bin.shape[1])))
            params = vars(self) | {
                "features": rng.choice(X_bin.shape[1], n_features, replace=False),
                "efb_plan": self.efb_plan_,
            }
            tree, importance = _build_tree_and_importance(
                X_bin, gradients, hessians, rows, weights, params
            )
            raw_score += self.learning_rate * predict_tree(tree, X_bin)
            self.trees_.append(tree)
            tree_importances.append(importance.copy())
            self.feature_importances_ += importance
            train_loss = binary_logloss(y, sigmoid(raw_score))
            self.evals_result_["training"]["binary_logloss"].append(train_loss)

            if X_val_bin is not None:
                val_score_raw += self.learning_rate * predict_tree(tree, X_val_bin)
                score = binary_logloss(y_val, sigmoid(val_score_raw))
                self.evals_result_["validation"]["binary_logloss"].append(score)
                if score < best_score - 1e-12:
                    best_score = score
                    best_iteration = len(self.trees_)
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if (
                        early_stopping_rounds is not None
                        and no_improvement >= early_stopping_rounds
                    ):
                        break

        if X_val_bin is None:
            best_iteration = len(self.trees_)
            best_score = self.evals_result_["training"]["binary_logloss"][-1]
        elif best_iteration == 0:
            best_iteration = len(self.trees_)
            best_score = self.evals_result_["validation"]["binary_logloss"][-1]

        if early_stopping_rounds is not None and best_iteration < len(self.trees_):
            self.trees_ = self.trees_[:best_iteration]
            self.feature_importances_ = np.sum(
                np.asarray(tree_importances[:best_iteration]), axis=0
            )

        self.best_iteration_ = int(best_iteration)
        self.best_score_ = float(best_score)
        self.n_estimators_ = len(self.trees_)
        return self

    def predict_proba(self, X):
        """Trả ma trận xác suất [P(lớp 0), P(lớp 1)] từ 0 đến 1."""

        if not hasattr(self, "trees_"):
            raise RuntimeError("Cần gọi fit trước khi dự đoán.")

        X, _ = encode_categorical_features(
            X,
            self.categorical_features,
            self.category_maps_
        )

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Số đặc trưng không phù hợp.")

        # Chuyển dữ liệu về bin giống quá trình training
        X_bin = bin_data(
            X,
            self.bin_thresholds_
        )

        # Khởi tạo raw score từ F0
        score = np.full(
            len(X),
            self.init_score_,
            dtype=float
        )

        # Cộng kết quả của toàn bộ cây
        for tree in self.trees_:
            score += (
                self.learning_rate
                * predict_tree(tree, X_bin)
            )

        # Raw score -> xác suất lớp 1
        positive = sigmoid(score)

        # Đảm bảo xác suất nằm trong [0, 1]
        positive = np.clip(
            positive,
            0.0,
            1.0
        )

        # Xác suất lớp 0
        negative = 1.0 - positive

        # Ma trận:
        # cột 0 = P(class 0)
        # cột 1 = P(class 1)
        probabilities = np.column_stack(
            (negative, positive)
        )

        return probabilities

    def predict(self, X):
        """Dự đoán nhãn 0/1 với ngưỡng truyền vào ."""
        class_index = (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
        return self.classes_[class_index]

    def get_feature_importance(self):
        """Return gain-based feature importance accumulated over kept trees."""
        if not hasattr(self, "feature_importances_"):
            raise RuntimeError("Cần gọi fit trước khi lấy feature importance.")
        return self.feature_importances_.copy()


__all__ = [
    "LightGBMClassification", "TreeNode", "sigmoid", "calculate_gradients",
    "encode_categorical_features", "create_bins", "bin_data",
    "exclusive_feature_bundling", "goss_sample", "histogram_subtraction",
    "build_histograms", "leaf_value", "split_gain", "find_best_split", "build_tree",
    "predict_tree"
]
