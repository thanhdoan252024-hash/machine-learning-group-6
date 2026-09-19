# PCA From Scratch — UCI HAR

Project triển khai và đánh giá PCA bằng NumPy trên UCI HAR. Phần PCA gồm 11 giai đoạn; scaler và PCA chỉ fit trên train, giữ nguyên split chính thức.

## Xem project

| Thư mục / file | Nội dung |
|---|---|
| [docs/](docs/00_README_START_HERE.md) | Mục tiêu, lý thuyết, roadmap, checkpoint và tài liệu bàn giao |
| [docs/prompts/](docs/prompts/) | Prompt P00–P12 tương ứng các bước xử lý |
| [docs/templates/](docs/templates/) | 7 mẫu báo cáo/log; đã gộp các bản sao giống hệt ở run_template cũ |
| [docs/references/](docs/references/) | PDF Colab của lần chạy trước |
| [notebook/](notebook/) | Notebook nguồn của pipeline PCA |
| [scripts/](scripts/) | Chạy notebook, tạo bộ bằng chứng và kiểm tra artifacts |
| [data/](data/README.md) | Nguồn dữ liệu, mã SHA-256; dataset và ZIP nằm trong data/raw/ |
| [runs/](runs/README.md) | Notebook có output, bảng kết quả, biểu đồ và artifacts của từng lần chạy |

Thông tin trạng thái có trong [project_manifest.json](project_manifest.json). `.runtime/` là cache chạy local, được ẩn khỏi Git.

## Chạy và lưu toàn bộ bằng chứng

Môi trường đã kiểm tra: Python 3.13.5, NumPy 2.1.3, Pandas 2.2.3, Matplotlib 3.10.0.

```powershell
python -m pip install -r requirements-repro.txt
python scripts/run_pca.py
```

Trên máy hiện tại, nếu `python` trỏ tới Windows Store, dùng Python Anaconda đã có:

```powershell
& 'C:\Users\LENOVO\anaconda3\python.exe' scripts/run_pca.py
```

Runner dùng `data/raw/UCI HAR Dataset/` và tự tạo thư mục mới `runs/run_<UTC>/`. Có thể chỉ định dataset khác bằng `--dataset-dir`. Mỗi bộ kết quả gồm notebook đã thực thi, HTML, 13 hình PNG, 23 artifacts/bảng/metadata, báo cáo, phiên bản môi trường và SHA-256.

Notebook cũng có thể chạy tương tác trên Jupyter/Colab. Các biến môi trường `PCA_DATA_ROOT` và `PCA_RUN_DIR` cho phép chọn nơi đọc dữ liệu và lưu kết quả. Runner tự đặt hai biến này.

## Kiểm tra lại artifacts đã lưu

Thay `<run>` bằng tên thư mục từ [runs/latest.json](runs/latest.json):

```powershell
python scripts/verify_pca_artifacts.py --run-dir runs/<run> --dataset-dir "data/raw/UCI HAR Dataset"
```

Lệnh này đọc lại NPY/NPZ, kiểm tra phép chiếu, CSV, checkpoint, hình PNG và mã SHA-256; không fit lại PCA. Các báo cáo classification chưa thuộc phạm vi thực hiện hiện tại.
