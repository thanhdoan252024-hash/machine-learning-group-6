# Dữ liệu UCI HAR

```text
data/
├── README.md
├── dataset_provenance.json
└── raw/
    ├── UCI_HAR_Dataset.zip
    └── UCI HAR Dataset/
        ├── train/
        ├── test/
        ├── features.txt
        └── activity_labels.txt
```

Dữ liệu lấy từ [UCI chính thức](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). URL tải và SHA-256 của ZIP được lưu trong `dataset_provenance.json`; hash của từng file đầu vào được ghi thêm trong `runs/<run>/dataset_manifest.json`.

Dataset hiện có 7.352 mẫu train, 2.947 mẫu test, 561 đặc trưng và 6 lớp hoạt động. Giữ nguyên split UCI.

`raw/` được bỏ qua trong Git vì có thể tải lại. Trên máy khác, tải ZIP theo URL trong provenance, giải nén ZIP ngoài và ZIP `UCI HAR Dataset.zip` bên trong vào `raw/` để có đúng cấu trúc trên, hoặc chạy notebook tương tác để dùng luồng tự tải/upload. Sau đó chạy `python scripts/run_pca.py` từ thư mục gốc project.
