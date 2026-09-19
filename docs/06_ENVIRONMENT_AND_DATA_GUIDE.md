# Environment and Data Guide

## Packages
- Python 3
- NumPy
- Pandas
- Matplotlib
- zipfile
- urllib

Không cần sklearn cho PCA.

## Dataset inputs được hỗ trợ
1. `UCI HAR Dataset/`
2. `UCI HAR Dataset.zip`
3. outer ZIP chứa nested `UCI HAR Dataset.zip`

## Colab download fallback
Nếu download lỗi:
`google.colab.files.upload()`

## Cấu trúc cần có
```text
UCI HAR Dataset/
├── train/X_train.txt
├── train/y_train.txt
├── test/X_test.txt
├── test/y_test.txt
├── features.txt
└── activity_labels.txt
```

## Portability
Notebook ưu tiên `PCA_DATA_ROOT` nếu được đặt; trên Colab dùng `/content`; trong project local tự tìm `data/raw/`. Runner sử dụng `data/raw/UCI HAR Dataset/` và ghi một thư mục mới trong `runs/`.

Môi trường chạy và hướng dẫn cài đặt: [README](../README.md), [requirements-repro.txt](../requirements-repro.txt). Nguồn và cách bố trí dữ liệu: [data/README.md](../data/README.md).

Giữ nguyên official train/test split.
