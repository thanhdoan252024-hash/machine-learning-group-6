# ============================================================
# PHẦN 3: XÂY DỰNG THUẬT TOÁN LIGHTGBM REGRESSION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. XÂY DỰNG CLASS LIGHTGBM REGRESSION
# ============================================================

class LightGBMRegression:

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42
    ):

        # Các Hyperparameters của LightGBM
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=-1
        )


    # ========================================================
    # METHOD FIT
    # ========================================================

    def fit(self, X_train, y_train):

        print("\nĐang huấn luyện mô hình LightGBM...")

        self.model.fit(
            X_train,
            y_train
        )

        print("Huấn luyện mô hình thành công!")

        return self


    # ========================================================
    # METHOD PREDICT
    # ========================================================

    def predict(self, X_test):

        return self.model.predict(X_test)


    # ========================================================
    # METHOD ĐÁNH GIÁ
    # ========================================================

    def evaluate(self, X_test, y_test):

        # Dự đoán
        y_pred = self.predict(X_test)

        # MSE
        mse = mean_squared_error(
            y_test,
            y_pred
        )

        # MAE
        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        # R2
        r2 = r2_score(
            y_test,
            y_pred
        )

        print("\n" + "=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ LIGHTGBM REGRESSION")
        print("=" * 60)

        print(f"MSE : {mse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}")

        return y_pred


# ============================================================
# 2. CHƯƠNG TRÌNH CHÍNH
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LIGHTGBM REGRESSION - INSURANCE DATASET")
    print("=" * 60)


    # ========================================================
    # 3. ĐỌC TRAIN / TEST
    # ========================================================

    X_train = pd.read_csv(
        "X_train.csv"
    )

    X_test = pd.read_csv(
        "X_test.csv"
    )

    y_train = pd.read_csv(
        "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        "y_test.csv"
    ).squeeze("columns")


    print("\nKích thước dữ liệu:")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test:",
        y_test.shape
    )


    # ========================================================
    # 4. KHỞI TẠO MÔ HÌNH
    # ========================================================

    model = LightGBMRegression(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42
    )


    # ========================================================
    # 5. HUẤN LUYỆN - FIT
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # 6. DỰ ĐOÁN VÀ ĐÁNH GIÁ
    # ========================================================

    y_pred = model.evaluate(
        X_test,
        y_test
    )


    # ========================================================
    # 7. TRỰC QUAN HÓA ACTUAL VS PREDICTED
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    # Đường lý tưởng y = x
    min_value = min(
        y_test.min(),
        y_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel(
        "Chi phí thực tế (Actual Charges)"
    )

    plt.ylabel(
        "Chi phí dự đoán (Predicted Charges)"
    )

    plt.title(
        "LightGBM Regression - Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        "actual_vs_predicted.png"
    )

    plt.show()


    # ========================================================
    # 8. FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance":
            model.model.feature_importances_
    })


    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )


    print("\n" + "=" * 60)
    print("MỨC ĐỘ QUAN TRỌNG CỦA CÁC BIẾN")
    print("=" * 60)

    print(importance)


    # ========================================================
    # 9. TRỰC QUAN FEATURE IMPORTANCE
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "LightGBM Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png"
    )

    plt.show()


    print("\nHOÀN THÀNH LIGHTGBM REGRESSION!")
