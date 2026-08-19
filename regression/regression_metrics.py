from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def print_regression_metrics(y_true, y_pred):
    metrics = calculate_regression_metrics(y_true, y_pred)
    print("===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"MSE : {metrics['MSE']:.4f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"R2  : {metrics['R2']:.4f}")
    return metrics
