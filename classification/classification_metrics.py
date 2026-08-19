from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_classification_metrics(y_true, y_pred):
    """
    Tính các chỉ số đánh giá cho bài toán phân loại.
    """

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def print_classification_metrics(y_true, y_pred):
    """
    In kết quả đánh giá mô hình Classification.
    """

    metrics = calculate_classification_metrics(
        y_true,
        y_pred
    )

    print("===== ĐÁNH GIÁ MÔ HÌNH =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")


def print_confusion_matrix(y_true, y_pred):
    """
    In Confusion Matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("===== CONFUSION MATRIX =====")
    print(cm)

    return cm


def print_classification_report(y_true, y_pred):
    """
    In Classification Report.
    """

    print("===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["No failure", "Failure"],
            zero_division=0
        )
    )