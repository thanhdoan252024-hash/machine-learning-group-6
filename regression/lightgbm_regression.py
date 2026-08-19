from lightgbm import LGBMRegressor

class LightGBMRegression:
    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1, random_state=42):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_feature_importance(self):
        return self.model.feature_importances_
