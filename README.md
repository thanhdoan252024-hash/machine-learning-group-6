# t-SNE cho dữ liệu chữ số viết tay

Project cài đặt **exact t-SNE bằng NumPy** và dùng nó để giảm dữ liệu Optical Recognition of Handwritten Digits xuống 2 chiều. Notebook thực hiện tiền xử lý, thử nhiều giá trị perplexity, đánh giá kết quả và trực quan hóa các chữ số theo nhãn 0–9.

## Dataset

Dữ liệu gốc nằm trong `data/raw/`:

- `optdigits.tra`: 3.823 mẫu huấn luyện;
- `optdigits.tes`: 1.797 mẫu kiểm thử;
- `optdigits.names`: mô tả bộ dữ liệu.

Mỗi mẫu gốc gồm 64 giá trị nguyên từ 0 đến 16, biểu diễn ảnh 8×8, và một nhãn chữ số từ 0 đến 9. Theo file mô tả, dữ liệu không có giá trị thiếu.

Notebook ghép hai tập thành 5.620 mẫu, đặt tên các cột `Pixel_1`–`Pixel_64` và `label`, rồi lưu vào `data/processed/optdigits.csv`. Lệnh `to_csv` hiện tại không đặt `index=False`, vì vậy file đã xử lý có thêm cột index; `load_optdigits_csv()` lấy mọi cột trừ cột cuối làm `X`, nên quy trình hiện tại sử dụng cột index này cùng 64 đặc trưng pixel.

## Quy trình thực nghiệm

Quy trình trong `tsne_digits.ipynb.ipynb`:

1. Đọc hai file raw bằng pandas, kiểm tra số giá trị null và kích thước dữ liệu, sau đó ghép và lưu CSV đã xử lý.
2. Nạp CSV bằng `load_optdigits_csv()` và lấy 2.000 mẫu đầu để thử nghiệm.
3. Chạy t-SNE 2D với các perplexity `5, 10, 15, 20, 25, 30, 45, 50`; mỗi lần chạy 500 vòng với `random_state=42`.
4. Đánh giá từng cấu hình bằng KL Divergence cuối, Trustworthiness tại `k=5` và runtime; lưu bảng CSV và các biểu đồ so sánh.
5. Notebook nhận định khoảng 20–30 phù hợp nhất và chọn **perplexity = 25** vì Trustworthiness rất cao trong khi KL Divergence tương đối thấp. Trên tập thử 2.000 mẫu, cấu hình này đạt KL Divergence `0.598247`, Trustworthiness `0.998730` và runtime khoảng `64.01` giây trong lần chạy đã lưu.
6. Chạy lại trên toàn bộ 5.620 mẫu với perplexity 25, 1.000 vòng, rồi vẽ scatter plot 2D theo nhãn bằng Seaborn.

### Kết quả đánh giá theo perplexity

Bảng dưới đây được lấy từ `outputs/results/experiment_evaluation.csv`, tương ứng với thí nghiệm trên 2.000 mẫu đầu, 500 vòng lặp và `random_state=42`:

| Perplexity | KL Divergence | Trustworthiness (k=5) | Runtime (giây) |
|-----------:|--------------:|----------------------:|----------------:|
| 5  | 0.844426 | 0.998132 | 162.95 |
| 10 | 0.737080 | 0.998589 | 101.30 |
| 15 | 0.683824 | 0.998723 | 95.57 |
| 20 | 0.638327 | **0.998752** | 67.87 |
| **25** | **0.598247** | **0.998730** | **64.01** |
| 30 | 0.579535 | 0.998574 | 53.60 |
| 45 | 0.462929 | 0.998049 | 53.23 |
| 50 | 0.437503 | 0.997870 | 52.35 |

Trong lần chạy này, KL Divergence giảm dần khi perplexity tăng, còn Trustworthiness cao nhất tại perplexity 20 rồi giảm nhẹ. Notebook không chọn cấu hình chỉ dựa trên một metric: perplexity 25 được dùng cho kết quả cuối vì nằm trong khoảng 20–30 đã nhận định là phù hợp, giữ Trustworthiness gần mức cao nhất và có KL Divergence thấp hơn perplexity 20. Runtime là thời gian đo của từng lần chạy đã lưu, không phải giá trị đảm bảo cho các máy khác.

## Cách hoạt động

`core/tsne_numpy.py` triển khai t-SNE thuần NumPy: tính ma trận khoảng cách Euclid bình phương; tìm `beta` riêng cho từng điểm bằng binary search để khớp perplexity; đối xứng hóa xác suất trong không gian gốc; khởi tạo embedding gần gốc tọa độ; tính xác suất Student-t trong không gian thấp; rồi tối ưu KL Divergence bằng gradient descent với early exaggeration, momentum và adaptive gains. Đây là exact t-SNE nên cần bộ nhớ `O(N²)`.

Mô hình hỗ trợ dừng sớm sau giai đoạn early exaggeration khi KL không cải thiện đủ trong số vòng `patience`. `core/evaluation.py` chạy nhiều perplexity, đo thời gian, tính Trustworthiness, lưu kết quả và sinh các biểu đồ.

## Cách chạy

Yêu cầu Python 3.10+ và các thư viện được import trong project:

```bash
pip install numpy pandas matplotlib seaborn jupyter
jupyter notebook tsne_digits.ipynb.ipynb
```

Chạy các cell theo thứ tự từ thư mục gốc của project. Repo hiện không có file `requirements.txt`.

Có thể chạy demo trong module đánh giá bằng:

```bash
python -m core.evaluation
```

Demo này dùng 1.000 mẫu đầu, các perplexity `10, 30, 50` và 500 vòng.

## Outputs

- `outputs/results/experiment_evaluation.csv`: perplexity, KL Divergence, runtime và Trustworthiness.
- `outputs/figures/tsne_perplexity_grid.png`: lưới embedding của các perplexity đã thử.
- `outputs/figures/perplexity_vs_kl.png`: perplexity so với KL Divergence.
- `outputs/figures/perplexity_vs_runtime.png`: perplexity so với runtime.
- `outputs/figures/perplexity_vs_trustworthiness.png`: perplexity so với Trustworthiness.
- `outputs/figures/tsne_perplexity_25.png`: embedding 2D cuối trên toàn bộ dữ liệu, tô màu theo nhãn.

## Cấu trúc chính

```text
core/                       Cài đặt t-SNE và hàm đánh giá
data/raw/                   Dữ liệu Optdigits gốc và mô tả
data/processed/             CSV được tạo bởi notebook
outputs/figures/            Biểu đồ kết quả
outputs/results/            Bảng metric
tsne_digits.ipynb.ipynb     Notebook thực nghiệm chính
```
