import numpy as np


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy and Gradient Descent.
    
    Supports:
        - fit(X, y)
        - predict(X)
        - get_coefficients()
        - get_intercept()
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.intercept = 0.0
        self.loss_history = []
        self.iterations_run = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0

        previous_loss = None
        for iteration in range(1, self.n_iterations + 1):
            y_pred = X @ self.weights + self.intercept
            y_pred = y_pred.reshape(-1, 1)

            error = y_pred - y
            grad_w = (2 / n_samples) * (X.T @ error).ravel()
            grad_b = (2 / n_samples) * np.sum(error)

            # Gradient descent update
            self.weights -= self.learning_rate * grad_w
            self.intercept -= self.learning_rate * grad_b

            # Current loss
            y_pred_after = X @ self.weights + self.intercept
            loss = np.mean((y_pred_after - y.ravel()) ** 2)
            self.loss_history.append(loss)
            self.iterations_run = iteration

            if previous_loss is not None and abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss

        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.weights + self.intercept

    def get_coefficients(self):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call fit(X, y) first.")
        return self.weights.copy()

    def get_intercept(self):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet. Call fit(X, y) first.")
        return float(self.intercept)
