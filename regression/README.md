regression/
│
├── lightgbm_regression.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── student_exam_performance.csv
│   │
│   └── processed/
│       └── student_performance_processed.csv
│
├── metrics/
│   ├── __init__.py
│   ├── mean_absolute_error.py
│   ├── mean_squared_error.py
│   ├── r2_score.py
│   └── regression_evaluation.py
│
├── visualization/
│   ├── __init__.py
│   ├── actual_vs_predicted.py
│   ├── residual_plot.py
│   ├── error_distribution.py
│   ├── train_test_comparison.py
│   └── feature_importance.py
│
├── utils/
│   ├── __init__.py
│   └── validation.py
│
├── outputs/
│   ├── result/
│   │   └── train_test_result.csv
│   │
│   └── figures/
│       ├── train_actual_vs_predicted.png
│       ├── test_actual_vs_predicted.png
│       ├── train_residual_plot.png
│       ├── test_residual_plot.png
│       ├── train_error_distribution.png
│       ├── test_error_distribution.png
│       ├── train_test_metrics_comparison.png
│       └── feature_importance.png
│
└── tests/
    ├── __init__.py
    ├── test_lightgbm_regression.py
    ├── test_mean_absolute_error.py
    ├── test_mean_squared_error.py
    ├── test_r2_score.py
    ├── test_regression_evaluation.py
    └── test_validation.py

notebooks/
└── student_performance_prediction.ipynb
