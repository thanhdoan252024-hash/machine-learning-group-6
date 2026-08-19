from lightgbm import LGBMClassifier


class LightGBMClassification:
    """
    LightGBM model cho bài toán
    Machine Failure Classification.
    """

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=101
    ):

        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1
        )

    def fit(self, X_train, y_train):
        """
        Huấn luyện mô hình.
        """

        self.model.fit(
            X_train,
            y_train
        )

        return self

    def predict(self, X_test):
        """
        Dự đoán nhãn 0 hoặc 1.
        """

        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Dự đoán xác suất.
        """

        return self.model.predict_proba(X_test)

    def get_feature_importance(self):
        """
        Trả về Feature Importance.
        """

        return self.model.feature_importances_