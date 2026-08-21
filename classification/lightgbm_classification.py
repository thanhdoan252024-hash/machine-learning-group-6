
from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    """Chuyển raw score thành xác suất."""
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def calculate_gradients(y, raw_score):
    """Tính gradient và Hessian của binary log-loss."""
    probability = sigmoid(raw_score)
    gradient = probability - y
    hessian = np.maximum(probability * (1 - probability), 1e-12)
    return gradient, hessian


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
    q = np.linspace(0, 1, max_bins)[1:-1]
    cuts = []
    for column in X.T:
        valid = column[np.isfinite(column)]
        cuts.append(np.unique(np.quantile(valid, q))
                    if len(valid) > 1 and np.ptp(valid) > 0 else np.array([]))
    return cuts


def bin_data(X, cuts):
    """Nén đặc trưng thành bin; bin 0 chỉ dành riêng cho missing value."""
    result = np.empty(X.shape, dtype=np.int32)
    for j, thresholds in enumerate(cuts):
        result[:, j] = np.where(
            np.isnan(X[:, j]), 0,
            np.searchsorted(thresholds, X[:, j], side="right") + 1)
    return result


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
    """Tạo histogram sibling bằng histogram cha trừ histogram một child."""
    return {
        feature: parent_histogram[feature] - child_histogram[feature]
        for feature in parent_histogram
    }


def build_histograms(X, gradients, hessians, rows, weights, features):
    """Dựng histogram [gradient, Hessian, count] cho từng feature."""
    histograms = {}
    g, h = gradients[rows] * weights, hessians[rows] * weights
    for feature in features:
        bins = X[rows, feature]
        n_bins = int(X[:, feature].max()) + 1
        histograms[int(feature)] = np.vstack([
            np.bincount(bins, weights=g, minlength=n_bins),
            np.bincount(bins, weights=h, minlength=n_bins),
            np.bincount(bins, minlength=n_bins)
        ])
    return histograms


def goss_sample(gradients, top_rate, other_rate, rng):
    """Giữ gradient lớn, lấy mẫu gradient nhỏ và bù trọng số (GOSS)."""
    n = len(gradients)
    n_top = max(1, int(np.ceil(top_rate * n)))
    order = np.argsort(np.abs(gradients))[::-1]
    large, pool = order[:n_top], order[n_top:]
    n_other = min(len(pool), max(1, int(np.ceil(other_rate * n))))
    small = rng.choice(pool, n_other, replace=False) if len(pool) else np.array([], int)
    rows = np.r_[large, small]
    weights = np.ones(len(rows))
    if len(small):
        weights[len(large):] = len(pool) / len(small)
    return rows, weights


def soft_threshold(g, alpha):
    """Áp dụng regularization L1 lên tổng gradient."""
    return np.sign(g) * max(abs(g) - alpha, 0)


def leaf_value(g, h, alpha, reg_lambda):
    """Giá trị leaf tối ưu theo bước Newton."""
    return -soft_threshold(g, alpha) / (h + reg_lambda)


def split_gain(lg, lh, rg, rh, alpha, reg_lambda):
    """Mức giảm loss khi chia một node thành hai leaf."""
    score = lambda g, h: soft_threshold(g, alpha) ** 2 / (h + reg_lambda)
    return 0.5 * (score(lg, lh) + score(rg, rh) - score(lg + rg, lh + rh))


@dataclass
class TreeNode:
    value: float
    feature: int = None
    threshold: int = None
    default_left: bool = True
    left: object = None
    right: object = None


def find_best_split(X, rows, histograms, params):
    """Tìm split tốt nhất và thử đưa missing sang cả trái lẫn phải."""
    best, best_gain = None, params["min_split_gain"]
    node_X = X[rows]

    for feature in params["features"]:
        bins = node_X[:, feature]
        histogram = histograms[int(feature)]
        n_bins = histogram.shape[1]
        if n_bins < 2:
            continue
        missing = histogram[:, 0]
        # Bin 0 là missing; tổng tích luỹ dưới đây chỉ gồm bin không thiếu.
        cumulative = np.c_[np.zeros(3), np.cumsum(histogram[:, 1:], axis=1)]
        total = histogram.sum(axis=1)

        for threshold in range(n_bins):
            for default_left in (True, False):
                left = cumulative[:, threshold].copy()
                if default_left:
                    left += missing
                right = total - left
                if (left[2] < params["min_child_samples"] or
                        right[2] < params["min_child_samples"] or
                        left[1] < params["min_child_weight"] or
                        right[1] < params["min_child_weight"]):
                    continue
                gain = split_gain(
                    left[0], left[1], right[0], right[1],
                    params["reg_alpha"], params["reg_lambda"])
                if gain > best_gain:
                    best_gain = gain
                    mask = (bins != 0) & (bins <= threshold)
                    if default_left:
                        mask |= bins == 0
                    best = (int(feature), threshold, default_left, mask)
    return best, best_gain


def build_tree(X, gradients, hessians, rows, weights, params):
    """Xây cây leaf-wise: luôn tách leaf có gain lớn nhất."""
    def make_leaf(r, w, depth, histograms):
        g, h = np.sum(gradients[r] * w), np.sum(hessians[r] * w)
        node = TreeNode(leaf_value(g, h, params["reg_alpha"], params["reg_lambda"]))
        return {"node": node, "rows": r, "weights": w, "depth": depth,
                "histograms": histograms, "split": None}

    root_histograms = build_histograms(
        X, gradients, hessians, rows, weights, params["features"])
    root = make_leaf(rows, weights, 0, root_histograms)
    leaves = [root]
    while len(leaves) < params["num_leaves"]:
        candidates = []
        for leaf in leaves:
            if leaf["split"] is None and (
                    params["max_depth"] < 0 or leaf["depth"] < params["max_depth"]):
                leaf["split"] = find_best_split(
                    X, leaf["rows"], leaf["histograms"], params)
            if leaf["split"] and leaf["split"][0] is not None:
                candidates.append(leaf)
        if not candidates:
            break

        chosen = max(candidates, key=lambda x: x["split"][1])
        (feature, threshold, default_left, mask), _ = chosen["split"]
        left_rows, right_rows = chosen["rows"][mask], chosen["rows"][~mask]
        left_weights, right_weights = chosen["weights"][mask], chosen["weights"][~mask]

        # Chỉ dựng histogram cho child nhỏ; child còn lại = parent - child nhỏ.
        if len(left_rows) <= len(right_rows):
            left_hist = build_histograms(
                X, gradients, hessians, left_rows, left_weights, params["features"])
            right_hist = histogram_subtraction(chosen["histograms"], left_hist)
        else:
            right_hist = build_histograms(
                X, gradients, hessians, right_rows, right_weights, params["features"])
            left_hist = histogram_subtraction(chosen["histograms"], right_hist)

        left = make_leaf(left_rows, left_weights, chosen["depth"] + 1, left_hist)
        right = make_leaf(right_rows, right_weights, chosen["depth"] + 1, right_hist)
        node = chosen["node"]
        node.feature, node.threshold = feature, threshold
        node.default_left = default_left
        node.left, node.right = left["node"], right["node"]
        leaves.remove(chosen)
        leaves.extend([left, right])
    return root["node"]


def predict_tree(tree, X):
    """Dự đoán phần raw score do một cây tạo ra."""
    result = np.empty(len(X))
    for i, row in enumerate(X):
        node = tree
        while node.feature is not None:
            value = row[node.feature]
            go_left = node.default_left if value == 0 else value <= node.threshold
            node = node.left if go_left else node.right
        result[i] = node.value
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
        random_state=None
    ):
        for name, value in locals().copy().items():
            if name != "self":
                setattr(self, name, value)
    def fit(self, X, y):
        """Huấn luyện ensemble và trả về self."""
        X, self.category_maps_ = encode_categorical_features(
            X, self.categorical_features)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or np.isinf(X).any():
            raise ValueError("X hoặc y không hợp lệ.")
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Mô hình chỉ hỗ trợ đúng hai lớp.")
        if not (0 < self.top_rate < 1 and 0 < self.other_rate < 1
                and self.top_rate + self.other_rate <= 1):
            raise ValueError("Tỉ lệ GOSS không hợp lệ.")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction phải thuộc (0, 1].")

        self.n_features_in_ = X.shape[1]
        self.bin_thresholds_ = create_bins(X, self.max_bins)
        X_bin = bin_data(X, self.bin_thresholds_)
        # EFB giảm số feature histogram nếu các feature thưa loại trừ nhau.
        X_bin, self.feature_bundles_ = exclusive_feature_bundling(
            X_bin, self.max_conflict_rate)
        positive_rate = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        self.init_score_ = np.log(positive_rate / (1 - positive_rate))
        raw_score = np.full(len(y), self.init_score_)
        self.trees_, rng = [], np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            gradients, hessians = calculate_gradients(y, raw_score)
            rows, weights = goss_sample(
                gradients, self.top_rate, self.other_rate, rng)
            n_features = max(1, int(np.ceil(self.feature_fraction * X_bin.shape[1])))
            params = vars(self) | {"features": rng.choice(
                X_bin.shape[1], n_features, replace=False)}
            tree = build_tree(X_bin, gradients, hessians, rows, weights, params)
            raw_score += self.learning_rate * predict_tree(tree, X_bin)
            self.trees_.append(tree)
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

        # Áp dụng lại EFB đã học trong fit()
        X_bin, _ = exclusive_feature_bundling(
            X_bin,
            bundles=self.feature_bundles_
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

        # Chỉ thay đổi CÁCH HIỂN THỊ, không thay đổi giá trị xác suất
        np.set_printoptions(
            suppress=True,
            precision=6,
            floatmode="fixed"
        )

        return probabilities

    def predict(self, X):
        """Dự đoán nhãn 0/1 với ngưỡng truyền vào ."""
        class_index = (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
        return self.classes_[class_index]

__all__ = [
    "LightGBMClassification", "TreeNode", "sigmoid", "calculate_gradients",
    "encode_categorical_features", "create_bins", "bin_data",
    "exclusive_feature_bundling", "goss_sample", "histogram_subtraction",
    "build_histograms", "leaf_value", "split_gain", "find_best_split", "build_tree",
    "predict_tree"
]
