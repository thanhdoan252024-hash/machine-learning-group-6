import lightgbm as lgb

class LightGBMRegression:
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train):
        print("Đang tiến hành huấn luyện (fit)...")
        self.model.fit(X_train, y_train)
        print("Huấn luyện hoàn tất!")

    def predict(self, X_test):
        print("Đang tiến hành dự đoán (predict)...")
        y_pred = self.model.predict(X_test)
        return y_pred
