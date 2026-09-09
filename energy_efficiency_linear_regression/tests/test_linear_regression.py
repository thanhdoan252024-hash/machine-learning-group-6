import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from linear_regression import LinearRegressionScratch
from preprocessing import preprocess_pipeline


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


def test_preprocess_pipeline_excludes_opposite_target_for_y2_prediction():
    df = pd.DataFrame({
        'X1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X3': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X4': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X5': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X6': ['A', 'B', 'A', 'B', 'A', 'B'],
        'X7': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'X8': ['C', 'D', 'C', 'D', 'C', 'D'],
        'Y1': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'Y2': [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    })

    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
        df=df,
        target_col='Y2',
        test_size=0.2,
        random_seed=42,
    )

    assert 'Y1' not in X_train.columns
    assert 'Y2' not in X_train.columns
    assert len(y_train) == 5
    assert len(y_test) == 1
