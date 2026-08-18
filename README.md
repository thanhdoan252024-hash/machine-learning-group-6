## 📁 Project Structure

```text
LightGBM/
│
├── regression/
│   ├── data/
│   │   ├── raw/
│   │   │   └── regression_raw.csv
│   │   └── processed/
│   │       └── regression_cleaned.csv
│   │
│   ├── lightgbm_regression.py
│   ├── regression_metrics.py
│   │
│   └── product_sales_prediction.ipynb
│       └── ⭐ Thực hiện:
│           ├── Đọc dữ liệu
│           ├── Làm sạch dữ liệu
│           ├── Mã hóa dữ liệu nếu cần
│           ├── Lưu dataset đã xử lý
│           ├── Chia Train/Test 80/20
│           ├── Import LightGBMRegression
│           ├── Huấn luyện model
│           ├── Dự đoán
│           ├── Gọi các metrics đánh giá
│           └── Trực quan hóa kết quả
│
├── classification/
│   ├── data/
│   │   ├── raw/
│   │   │   └── classification_raw.csv
│   │   └── processed/
│   │       └── classification_cleaned.csv
│   │
│   ├── lightgbm_classification.py
│   ├── classification_metrics.py
│   │
│   └── machine_failure_prediction.ipynb
│       └── ⭐ Thực hiện:
│           ├── Đọc dữ liệu
│           ├── Làm sạch dữ liệu
│           ├── Mã hóa dữ liệu nếu cần
│           ├── Lưu dataset đã xử lý
│           ├── Chia Train/Test 80/20
│           ├── Import LightGBMClassification
│           ├── Huấn luyện model
│           ├── Dự đoán
│           ├── Gọi các metrics đánh giá
│           └── Trực quan hóa kết quả
│
├── README.md
├── requirements.txt
└── .gitignore
```
