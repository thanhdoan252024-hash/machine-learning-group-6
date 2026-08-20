"""LightGBM rút gọn cho classification nhị phân, chỉ dùng NumPy."""

from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    """Chuyển raw score thành xác suất."""
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


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
    """Nén đặc trưng liên tục thành chỉ số bin nguyên."""
    result = np.empty(X.shape, dtype=np.int32)
    for j, thresholds in enumerate(cuts):
        result[:, j] = np.where(
            np.isnan(X[:, j]), 0,
            np.searchsorted(thresholds, X[:, j], side="right") + 1)
    return result


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
    left: object = None
    right: object = None


def find_best_split(X, gradients, hessians, rows, weights, params):
    """Tìm split tốt nhất bằng histogram gradient và Hessian."""
    best, best_gain = None, params["min_split_gain"]
    node_X = X[rows]
    g, h = gradients[rows] * weights, hessians[rows] * weights

    for feature in params["features"]:
        bins = node_X[:, feature]
        n_bins = int(bins.max()) + 1
        if n_bins < 2:
            continue
        # Histogram cho phép đánh giá nhanh mọi ngưỡng bin.
        gh = np.bincount(bins, weights=g, minlength=n_bins)
        hh = np.bincount(bins, weights=h, minlength=n_bins)
        nh = np.bincount(bins, minlength=n_bins)
        cg, ch, cn = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1], np.cumsum(nh)[:-1]
        total_g, total_h, total_n = gh.sum(), hh.sum(), len(rows)

        for threshold in range(n_bins - 1):
            right_h, right_n = total_h - ch[threshold], total_n - cn[threshold]
            if (cn[threshold] < params["min_child_samples"] or
                    right_n < params["min_child_samples"] or
                    ch[threshold] < params["min_child_weight"] or
                    right_h < params["min_child_weight"]):
                continue
            gain = split_gain(
                cg[threshold], ch[threshold], total_g - cg[threshold], right_h,
                params["reg_alpha"], params["reg_lambda"])
            if gain > best_gain:
                best_gain = gain
                best = (int(feature), threshold, bins <= threshold)
    return best, best_gain


def build_tree(X, gradients, hessians, rows, weights, params):
    """Xây cây leaf-wise: luôn tách leaf có gain lớn nhất."""
    def make_leaf(r, w, depth):
        g, h = np.sum(gradients[r] * w), np.sum(hessians[r] * w)
        node = TreeNode(leaf_value(g, h, params["reg_alpha"], params["reg_lambda"]))
        return {"node": node, "rows": r, "weights": w, "depth": depth, "split": None}

    root = make_leaf(rows, weights, 0)
    leaves = [root]
    while len(leaves) < params["num_leaves"]:
        candidates = []
        for leaf in leaves:
            if leaf["split"] is None and (
                    params["max_depth"] < 0 or leaf["depth"] < params["max_depth"]):
                leaf["split"] = find_best_split(
                    X, gradients, hessians, leaf["rows"], leaf["weights"], params)
            if leaf["split"] and leaf["split"][0] is not None:
                candidates.append(leaf)
        if not candidates:
            break

        chosen = max(candidates, key=lambda x: x["split"][1])
        (feature, threshold, mask), _ = chosen["split"]
        left = make_leaf(chosen["rows"][mask], chosen["weights"][mask],
                         chosen["depth"] + 1)
        right = make_leaf(chosen["rows"][~mask], chosen["weights"][~mask],
                          chosen["depth"] + 1)
        node = chosen["node"]
        node.feature, node.threshold = feature, threshold
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
            node = node.left if row[node.feature] <= node.threshold else node.right
        result[i] = node.value
    return result


class LightGBMClassification:
    """Phân loại nhị phân bằng histogram, GOSS và cây leaf-wise."""

    def __init__(
        self, n_estimators=100, learning_rate=0.1, num_leaves=31,
        max_depth=-1, max_bins=255, min_child_samples=20,
        min_child_weight=1e-3, min_split_gain=0.0, reg_alpha=0.0,
        reg_lambda=1.0, top_rate=0.2, other_rate=0.1,
        feature_fraction=1.0, random_state=None
    ):
        # Lưu hyperparameter thành thuộc tính của class.
        for name, value in locals().copy().items():
            if name != "self":
                setattr(self, name, value)

    def fit(self, X, y):
        """Huấn luyện ensemble và trả về self."""
        X, y = np.asarray(X, float), np.asarray(y)
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
        positive_rate = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        self.init_score_ = np.log(positive_rate / (1 - positive_rate))
        raw_score = np.full(len(y), self.init_score_)
        self.trees_, rng = [], np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Gradient và Hessian của binary log-loss.
            probability = sigmoid(raw_score)
            gradients = probability - y
            hessians = np.maximum(probability * (1 - probability), 1e-12)
            rows, weights = goss_sample(
                gradients, self.top_rate, self.other_rate, rng)
            n_features = max(1, int(np.ceil(self.feature_fraction * X.shape[1])))
            params = vars(self) | {"features": rng.choice(
                X.shape[1], n_features, replace=False)}
            tree = build_tree(X_bin, gradients, hessians, rows, weights, params)
            raw_score += self.learning_rate * predict_tree(tree, X_bin)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        """Trả ma trận xác suất [P(class 0), P(class 1)]."""

        if not hasattr(self, "trees_"):
            raise RuntimeError("Cần gọi fit trước khi dự đoán.")

        X = np.asarray(X, float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Số đặc trưng không phù hợp.")

        X_bin = bin_data(X, self.bin_thresholds_)

        score = np.full(len(X), self.init_score_)

        for tree in self.trees_:
            score += self.learning_rate * predict_tree(tree, X_bin)

        # Xác suất lớp 1
        positive = sigmoid(score)

        positive = np.nan_to_num(
            positive,
            nan=0.5,
            posinf=1.0,
            neginf=0.0
        )

        positive = np.clip(positive, 0.0, 1.0)

        # Xác suất lớp 0
        negative = 1.0 - positive

        probabilities = np.column_stack(
            (negative, positive)
        )

        # Làm tròn 4 chữ số
        probabilities = np.round(probabilities, 4)

        # Ép NumPy hiển thị dạng 0.xxxx thay vì e-01, e-04
        np.set_printoptions(
            suppress=True,
            precision=4
        )

        return probabilities
    def predict(self, X):
        """Dự đoán nhãn 0/1 với ngưỡng 0.5."""

        probabilities = self.predict_proba(X)

        # Lấy xác suất class 1
        positive_probability = probabilities[:, 1]

        class_index = (
            positive_probability >= 0.5
        ).astype(int)

        return self.classes_[class_index]

__all__ = ["LightGBMClassification"]
