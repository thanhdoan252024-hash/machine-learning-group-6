"""Generate machine-readable mathematical/correctness validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightgbm_from_scratch.core.binning import create_quantile_bins, bin_matrix
from lightgbm_from_scratch.core.efb import build_efb_plan, build_efb_histograms
from lightgbm_from_scratch.objectives.binary import (
    binary_logloss_gradient_hessian,
    sigmoid,
)
from lightgbm_from_scratch.objectives.regression import squared_error_gradient_hessian


OUTPUT = ROOT / "case_study" / "artifacts" / "core_validation.json"


def binary_gradient_error() -> float:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    raw = np.array([-1.2, -0.1, 0.8, 1.7])
    analytical, hessian = binary_logloss_gradient_hessian(y, raw)
    eps = 1e-6

    def loss(score):
        p = np.clip(sigmoid(score), 1e-15, 1 - 1e-15)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    numerical = np.empty_like(raw)
    for i in range(raw.size):
        plus = raw.copy(); plus[i] += eps
        minus = raw.copy(); minus[i] -= eps
        numerical[i] = (loss(plus) - loss(minus)) / (2 * eps)
    return float(np.max(np.abs(analytical - numerical))), bool(np.all(hessian > 0))


def regression_gradient_error() -> float:
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 17.0, 29.0])
    analytical, hessian = squared_error_gradient_hessian(y, pred)
    expected = pred - y
    return float(np.max(np.abs(analytical - expected))), bool(np.all(hessian == 1))


def efb_histogram_error() -> tuple[float, bool]:
    X = np.array([
        [3.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [5.0, 0.0, np.nan],
        [0.0, 7.0, 1.0],
        [8.0, 0.0, 1.0],
        [0.0, 4.0, 1.0],
    ])
    cuts = create_quantile_bins(X, 8, quantile_style="centers")
    binned = bin_matrix(X, cuts, missing_bin="zero")
    plan = build_efb_plan(X, binned, missing_bin="zero")
    gradients = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    hessians = np.linspace(0.2, 0.7, 6)
    rows = np.arange(len(X))
    weights = np.ones(len(X))
    bundled = build_efb_histograms(
        plan, gradients, hessians, rows, weights=weights, include_count=True
    )

    max_error = 0.0
    for feature in range(X.shape[1]):
        n_bins = int(binned[:, feature].max()) + 1
        direct = np.vstack([
            np.bincount(binned[:, feature], weights=gradients, minlength=n_bins),
            np.bincount(binned[:, feature], weights=hessians, minlength=n_bins),
            np.bincount(binned[:, feature], minlength=n_bins),
        ])
        max_error = max(max_error, float(np.max(np.abs(bundled[feature] - direct))))
    return max_error, plan.applied


def main() -> None:
    binary_error, hessian_positive = binary_gradient_error()
    regression_error, regression_hessian_one = regression_gradient_error()
    efb_error, efb_applied = efb_histogram_error()

    checks = {
        "regression_gradient_max_abs_error": {
            "value": regression_error,
            "threshold": 1e-12,
            "pass": regression_error < 1e-12,
        },
        "regression_hessian_all_one": {
            "value": regression_hessian_one,
            "pass": regression_hessian_one,
        },
        "binary_gradient_finite_difference_max_abs_error": {
            "value": binary_error,
            "threshold": 1e-5,
            "pass": binary_error < 1e-5,
        },
        "binary_hessian_positive": {
            "value": hessian_positive,
            "pass": hessian_positive,
        },
        "efb_applied_on_mutually_exclusive_features": {
            "value": efb_applied,
            "pass": efb_applied,
        },
        "efb_histogram_equivalence_max_abs_error": {
            "value": efb_error,
            "threshold": 1e-12,
            "pass": efb_error < 1e-12,
        },
    }
    payload = {
        "schema_version": 1,
        "all_pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
