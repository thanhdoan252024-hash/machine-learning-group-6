"""Small dependency-free metrics used by model selection and case studies."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true và y_pred phải cùng chiều và không rỗng.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def binary_logloss(y_true, positive_probability):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(positive_probability, dtype=float).reshape(-1)
    if y_true.shape != p.shape or y_true.size == 0:
        raise ValueError("y_true và probability phải cùng chiều và không rỗng.")
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("binary_logloss yêu cầu target 0/1.")
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def binary_precision_recall_curve(y_true, y_score):
    """Return thresholded precision/recall points without sklearn."""
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if y_true.shape != scores.shape or y_true.size == 0:
        raise ValueError("y_true và y_score phải cùng chiều và không rỗng.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("PR curve yêu cầu target 0/1.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score phải hữu hạn.")
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        raise ValueError("PR curve không xác định khi không có positive sample.")

    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order].astype(int)
    s_sorted = scores[order]
    thresholds = []
    precision = []
    recall = []
    tp = fp = 0
    index = 0
    while index < len(scores):
        threshold = float(s_sorted[index])
        end = index
        while end < len(scores) and s_sorted[end] == s_sorted[index]:
            if y_sorted[end] == 1:
                tp += 1
            else:
                fp += 1
            end += 1
        thresholds.append(threshold)
        precision.append(tp / (tp + fp))
        recall.append(tp / positives)
        index = end
    return {
        "thresholds": np.asarray(thresholds, dtype=float),
        "precision": np.asarray(precision, dtype=float),
        "recall": np.asarray(recall, dtype=float),
    }


def average_precision(y_true, y_score):
    """Average precision as a step-wise area under the PR curve."""
    curve = binary_precision_recall_curve(y_true, y_score)
    precision = curve["precision"]
    recall = curve["recall"]
    previous_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += float((r - previous_recall) * p)
        previous_recall = float(r)
    return float(ap)
