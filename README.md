## 📁 Project Structure

```text
LightGBM/
│
├── regression/
│   ├── data/
│   │   ├── raw/
│   │   │   └── regression_raw.csv
│   │   │
│   │   └── processed/
│   │       └── regression_cleaned.csv
│   │
│   ├── preprocessing.ipynb
│   ├── lightgbm_regression.py
│   ├── regression_metrics.py
│   │
│   └── product_sales_prediction.ipynb
│       └── ⭐ Thực hiện:
│           ├── Chia Train/Test 80/20
│           ├── Huấn luyện model
│           ├── Dự đoán
│           ├── Đánh giá hiệu suất
│           └── Trực quan hóa kết quả
│
├── classification/
│   ├── data/
│   │   ├── raw/
│   │   │   └── classification_raw.csv
│   │   │
│   │   └── processed/
│   │       └── classification_cleaned.csv
│   │
│   ├── preprocessing_data.ipynb
│   ├── lightgbm_classification.py
│   ├── classification_metrics.py
│   │
│   └── machine_failure_prediction.ipynb
│       └── ⭐ Thực hiện:
│           ├── Chia Train/Test 80/20
│           ├── Huấn luyện model
│           ├── Dự đoán
│           ├── Đánh giá hiệu suất
│           └── Trực quan hóa kết quả
│
├── README.md
├── requirements.txt
└── .gitignore
```
