import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from linear_regression import LinearRegressionScratch


def test_linear_regression_scratch_fit_predict_and_loss():
    X = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=100, tolerance=1e-9)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert len(model.loss_history) > 0
    assert np.all(np.isfinite(model.get_coefficients()))
    assert np.isfinite(model.get_intercept())
    assert model.loss_history[0] >= model.loss_history[-1]
