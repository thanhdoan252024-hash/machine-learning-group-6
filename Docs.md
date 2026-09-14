Nội dung file `docs/PROJECT_PROCESS.md`:

```markdown
# Quá trình thực hiện dự án t-SNE trên dữ liệu chữ số viết tay

## 1. Giới thiệu và mục tiêu

Dự án thuộc nhánh `t-SNE` của repository `machine-learning-group-6`. Mục tiêu chính là cài đặt thuật toán **exact t-SNE thuần bằng NumPy** (không dùng thư viện scikit-learn hay các gói t-SNE có sẵn) và áp dụng để giảm chiều dữ liệu Optical Recognition of Handwritten Digits (Optdigits) xuống 2 chiều, sau đó đánh giá và trực quan hóa theo nhãn chữ số 0–9.

Các thành phần chính của dự án:

- Module `core/tsne_numpy.py`: triển khai đầy đủ thuật toán t-SNE.
- Module `core/evaluation.py`: chạy thí nghiệm trên nhiều giá trị perplexity, đo KL Divergence, Trustworthiness và runtime, đồng thời lưu kết quả và biểu đồ.
- Notebook `tsne_digits.ipynb.ipynb`: thực hiện tiền xử lý dữ liệu, gọi hàm đánh giá và trực quan hóa kết quả cuối.
- Thư mục `tests/`: kiểm thử các hàm cốt lõi và đầu ra của lớp `TSNE`.
- Dữ liệu gốc trong `data/raw/`, dữ liệu đã xử lý trong `data/processed/`, kết quả trong `outputs/`.

Theo README, đây là bản exact t-SNE với độ phức tạp bộ nhớ O(N²).

## 2. Bộ dữ liệu và tiền xử lý

### 2.1. Nguồn dữ liệu

Dữ liệu gốc nằm trong `data/raw/`:

- `optdigits.tra`: 3.823 mẫu huấn luyện.
- `optdigits.tes`: 1.797 mẫu kiểm thử.
- `optdigits.names`: mô tả bộ dữ liệu (Optical Recognition of Handwritten Digits, nguồn E. Alpaydin & C. Kaynak, 1998).

Mỗi mẫu gồm 64 giá trị nguyên trong khoảng [0, 16] (ảnh 8×8) và một nhãn chữ số từ 0 đến 9. File mô tả khẳng định không có giá trị thiếu.

### 2.2. Các bước tiền xử lý trong notebook

1. Đọc hai file raw bằng `pandas.read_csv(..., header=None)`.
2. Kiểm tra số lượng giá trị null (cả hai tập đều bằng 0) và in kích thước.
3. Ghép hai tập bằng `pd.concat([train, test], ignore_index=True, axis=0)` → tổng 5.620 mẫu.
4. Đặt tên cột: `Pixel_1` … `Pixel_64` và `label`.
5. Lưu ra `data/processed/optdigits.csv` bằng `to_csv` **không** có `index=False`.

Hệ quả: file CSV đã xử lý có thêm cột index (tên `Unnamed: 0`), tổng cộng 66 cột. Hàm `load_optdigits_csv()` trong `core/evaluation.py` lấy `df.iloc[:, :-1]` làm `X` và cột cuối làm `y`. Do đó, trong quy trình thực tế, `X` gồm 65 cột (cột index + 64 pixel). README đã ghi nhận điểm này.

Khi thí nghiệm, notebook chỉ lấy 2.000 mẫu đầu (`X[:2000]`, `y[:2000]`) để chạy lưới perplexity; sau đó chạy lại trên toàn bộ 5.620 mẫu với cấu hình được chọn.

## 3. Xây dựng t-SNE bằng NumPy

Toàn bộ thuật toán được viết trong `core/tsne_numpy.py`. Các hàm module-level và lớp `TSNE` như sau.

### 3.1. Các hàm cốt lõi

- `squared_euclidean_distances(X)`: tính ma trận khoảng cách Euclid bình phương  
  \(D_{ij} = \|x_i\|^2 + \|x_j\|^2 - 2\langle x_i, x_j\rangle\), đường chéo được gán 0.
- `entropy_and_probabilities(distances, beta)`: tính phân phối điều kiện và entropy  
  \(p_j = \exp(-\beta d_j)/\sum_k\exp(-\beta d_k)\), \(H = -\sum p_j\ln p_j\). Có dịch logits để ổn định số.
- `binary_search_beta(distances, perplexity, ...)`: tìm \(\beta = 1/(2\sigma^2)\) sao cho entropy gần \(\ln(\text{perplexity})\) bằng binary search (tối đa 50 vòng, tolerance 1e-5).
- `conditional_probabilities(D, perplexity)`: tính toàn bộ \(p(j|i)\) (đường chéo = 0) và vector \(\beta\) riêng cho từng điểm.
- `symmetric_probabilities(P_cond)`:  
  \(p_{ij} = (p(j|i) + p(i|j))/(2N)\), đường chéo = 0, \(\sum P = 1\).
- `initialize_embedding(n, n_components, rng)`: khởi tạo \(Y \sim \mathcal{N}(0, 10^{-4})\).
- `student_t_probabilities(Y)`: phân phối Student-t 1 bậc tự do  
  \(a_{ij} = (1 + \|y_i - y_j\|^2)^{-1}\), \(q_{ij} = a_{ij}/\sum_{k\neq l}a_{kl}\).
- `kl_divergence(P, Q)`: \(KL(P\|Q) = \sum_{i\neq j} p_{ij}\ln(p_{ij}/q_{ij})\).
- `tsne_gradient(P, Q, Y, numerator)`:  
  \(\frac{\partial C}{\partial y_i} = 4\sum_j (p_{ij}-q_{ij})(y_i-y_j)/(1+\|y_i-y_j\|^2)\).
- `gradient_descent_step(...)`: một bước cập nhật với momentum và adaptive gains  
  (gain tăng 0.2 khi đổi hướng, giảm 0.8 khi cùng hướng, tối thiểu `min_gain`).

### 3.2. Lớp `TSNE`

Tham số mặc định quan trọng (theo `__init__`):

| Tham số | Giá trị mặc định |
|---------|------------------|
| `n_components` | 2 |
| `perplexity` | 30.0 |
| `learning_rate` | 200.0 |
| `n_iter` | 1000 |
| `early_exaggeration` | 12.0 |
| `early_exaggeration_iter` | 250 |
| `momentum` | 0.5 |
| `final_momentum` | 0.8 |
| `min_gain` | 0.01 |
| `patience` | 50 |
| `min_kl_improvement` | 1e-7 |
| `beta_tolerance` | 1e-5 |
| `beta_search_iter` | 50 |

Hàm `fit(X)`:

1. Kiểm tra đầu vào (số chiều, finite, điều kiện \(3\times\text{perplexity} < n\)).
2. Tính \(D\), \(P_{\text{cond}}\), \(\beta\), \(\sigma\), rồi \(P\) đối xứng.
3. Khởi tạo \(Y\), velocity = 0, gains = 1.
4. Lặp tối đa `n_iter` vòng:
   - Early exaggeration: nhân \(P\) với 12 trong 250 vòng đầu, momentum = 0.5; sau đó dùng \(P\) gốc và momentum = 0.8.
   - Tính \(Q\) và gradient, cập nhật \(Y\).
   - Sau giai đoạn early exaggeration: theo dõi KL; nếu không cải thiện quá `min_kl_improvement` trong `patience` vòng liên tiếp thì dừng sớm.
5. Lưu `embedding_`, `kl_divergence_`, `kl_history_`, `n_iter_`, `stopped_early_`, v.v.

Có alias `fit_tranform` (thiếu chữ “s”) để khớp cách viết trong đề bài.

## 4. Các bước hoạt động của thuật toán (theo code thực tế)

1. Tính ma trận khoảng cách Euclid bình phương trên không gian gốc.
2. Với mỗi điểm \(i\), tìm \(\beta_i\) bằng binary search để entropy khớp \(\ln(\text{perplexity})\).
3. Xây dựng ma trận xác suất điều kiện \(P_{\text{cond}}\) rồi đối xứng hóa thành \(P\).
4. Khởi tạo embedding \(Y\) gần gốc tọa độ.
5. Trong vòng lặp tối ưu:
   - Giai đoạn early exaggeration (250 vòng): phóng đại lực hút giữa các điểm gần.
   - Tính phân phối Student-t \(Q\) trên không gian thấp.
   - Tính gradient của KL và cập nhật bằng gradient descent có momentum + adaptive gains.
   - Căn giữa \(Y\) sau mỗi bước.
6. Sau khi hội tụ (hoặc hết vòng / dừng sớm), tính KL Divergence cuối cùng giữa \(P\) và \(Q\).

## 5. Kiểm tra thuật toán

Thư mục `tests/` chứa 22 test (pytest):

- `test_tsne_core.py`: kiểm tra khoảng cách, entropy/xác suất, binary search \(\beta\), \(P\) điều kiện và đối xứng, Student-t, KL, khởi tạo embedding, gradient và một bước gradient descent.
- `test_tsne_output.py`: chạy t-SNE trên hai cụm Gaussian tự sinh (40 mẫu, 5 đặc trưng); kiểm tra shape 2D, số mẫu không đổi, giá trị hữu hạn, embedding không collapse, hành vi `fit`/`fit_transform` và alias `fit_tranform`.

Chạy bằng `python -m pytest -q` từ thư mục gốc. Lưu ý: `pytest` chưa được liệt kê trong `requirement.txt`.

## 6. Thiết kế thí nghiệm và các giá trị perplexity

Thí nghiệm chính được thực hiện trong notebook thông qua hàm `run_evaluation` của `core/evaluation.py`:

- Tập dữ liệu: 2.000 mẫu đầu.
- Perplexity được thử: `[5, 10, 15, 20, 25, 30, 45, 50, 60, 70, 80, 90, 100]`.
- Số vòng lặp: 500.
- `random_state=42`.
- Trustworthiness tính với \(k=50\).

Với mỗi perplexity, hàm tạo một mô hình `TSNE`, đo thời gian `fit_transform`, lấy `kl_divergence_`, tính Trustworthiness, lưu embedding và ghi bảng kết quả.

Sau khi có bảng và lưới embedding, cấu hình được chọn để chạy trên **toàn bộ 5.620 mẫu** với `n_iter=1000`.

## 7. Các chỉ số KL Divergence, Trustworthiness, Runtime

Kết quả thực tế được lưu tại `outputs/results/experiment_evaluation.csv` (thí nghiệm 2.000 mẫu, 500 vòng, `random_state=42`, Trustworthiness \(k=50\)):

| Perplexity | KL Divergence | Trustworthiness (k=50) | Runtime (giây) |
|-----------:|--------------:|-----------------------:|---------------:|
| 5         | 0.841149      | 0.948917               | 60.26          |
| 10        | 0.741336      | 0.984014               | 64.90          |
| 15        | 0.681136      | 0.993240               | 55.68          |
| 20        | 0.640092      | 0.995545               | 54.76          |
| **25**    | **0.596478**  | **0.996855**           | **56.28**      |
| 30        | 0.577071      | 0.997093               | 57.56          |
| 45        | 0.467548      | 0.998047               | 143.12         |
| 50        | 0.441998      | 0.998217               | 154.10         |
| 60        | 0.392190      | 0.998450               | 242.23         |
| 70        | 0.349039      | 0.998594               | 169.38         |
| 80        | 0.312239      | 0.998683               | 218.22         |
| 90        | 0.276323      | 0.998710               | 265.03         |
| 100       | 0.254121      | 0.998711               | 240.82         |

- **KL Divergence**: giảm đều khi perplexity tăng (từ ~0.84 xuống ~0.25).
- **Trustworthiness**: tăng nhanh đến perplexity 25 (~0.997), sau đó cải thiện rất nhỏ.
- **Runtime**: khoảng 55–65 giây với perplexity ≤ 30; tăng mạnh và dao động khi perplexity ≥ 45 (tối đa ~265 giây tại 90).

## 8. Trực quan hóa và phân tích kết quả

Bốn hình được sinh tự động bởi `run_evaluation` và lưu trong `outputs/figures/`:

- `tsne_perplexity_grid.png`: lưới scatter plot của tất cả perplexity đã thử (tô màu theo nhãn).
- `perplexity_vs_kl.png`: đường cong KL theo perplexity.
- `perplexity_vs_runtime.png`: đường cong runtime.
- `perplexity_vs_trustworthiness.png`: đường cong Trustworthiness.

Phân tích hình dạng embedding (theo README):

- Perplexity 5–15: embedding bị chia thành nhiều đoạn và cụm nhỏ, bố cục rời rạc.
- Vùng 20–30: cấu trúc cục bộ liền mạch hơn nhưng vẫn giữ nhiều vùng riêng biệt.
- Perplexity 45–100: các điểm tổ chức thành một vài dải cong dài; dù KL thấp và Trustworthiness cao, bố cục bị nén mạnh và khó quan sát cấu trúc cục bộ.

Hình cuối cùng `tsne_perplexity_25.png` là scatter plot 2D của toàn bộ dữ liệu (tô màu theo nhãn 0–9 bằng Seaborn `tab10`).

## 9. Lý do chọn perplexity cuối cùng

Theo README (phần phân tích và kết luận chính thức của project):

- Perplexity 25 đạt Trustworthiness 0.996855, KL Divergence 0.596478 và runtime ~56 giây trên tập 2.000 mẫu.
- So với perplexity thấp: bớt phân mảnh.
- So với perplexity rất cao: các cấu trúc cục bộ chưa bị ép thành dải dài khó quan sát.
- Cân bằng giữa bảo toàn lân cận, chi phí tính toán và khả năng diễn giải trực quan → được chọn để chạy trên toàn bộ 5.620 mẫu với 1.000 vòng.

**Lưu ý về sự không nhất quán trong notebook:**

- Cell markdown (cell 9) viết rằng “Perplexity = 10 được chọn…”.
- Cell code chạy toàn bộ dữ liệu (cell 11) dùng `perplexity=10`.
- Cell markdown tiếp theo (cell 12) và tiêu đề hình (cell 13) lại ghi “Perplexity = 25” và lưu file `tsne_perplexity_25.png`.

README và bảng kết quả nhấn mạnh lựa chọn **25**. File hình cuối cùng cũng mang tên 25. Trong tài liệu này, chúng tôi bám theo mô tả chính thức trong README và các file kết quả đã lưu.

## 10. Kết quả, hạn chế và hướng phát triển

### 10.1. Kết quả đạt được

- Đã cài đặt thành công exact t-SNE thuần NumPy với đầy đủ các thành phần (binary search \(\beta\), early exaggeration, momentum, adaptive gains, early stopping).
- Thí nghiệm hệ thống trên 13 giá trị perplexity, lưu đầy đủ metric và 4 biểu đồ.
- Có bộ test pytest kiểm tra cả hàm cốt lõi và hành vi đầu ra.
- Trực quan hóa cuối cùng cho thấy các chữ số 0–9 được phân cụm tương đối rõ trên mặt phẳng 2D (theo hình `tsne_perplexity_25.png`).

### 10.2. Hạn chế quan sát được từ repo

- File CSV đã xử lý chứa cột index; `load_optdigits_csv` đưa cột này vào `X` → đặc trưng thực tế là 65 thay vì 64.
- Exact t-SNE có độ phức tạp bộ nhớ O(N²), không phù hợp với tập dữ liệu lớn hơn nhiều.
- Notebook còn một số chỗ viết không nhất quán về giá trị perplexity được chọn (10 vs 25).
- `requirement.txt` thiếu `pytest` (dù test dùng pytest).
- Runtime thực tế phụ thuộc máy; các số liệu chỉ phản ánh lần chạy đã lưu.

### 10.3. Hướng phát triển có thể

- Sửa bước lưu CSV (`index=False`) và cập nhật hàm load để chỉ lấy 64 pixel.
- Thêm phiên bản xấp xỉ (Barnes-Hut hoặc FFT) để mở rộng quy mô.
- Bổ sung so sánh với PCA hoặc UMAP trên cùng dữ liệu.
- Chuẩn hóa đường dẫn Windows/Linux trong notebook.
- Mở rộng test coverage và đưa `pytest` vào dependency.

---

*Tài liệu này được xây dựng dựa trên nội dung thực tế của nhánh `t-SNE`: README.md, `core/tsne_numpy.py`, `core/evaluation.py`, notebook `tsne_digits.ipynb.ipynb`, file kết quả `outputs/results/experiment_evaluation.csv`, các hình trong `outputs/figures/`, dữ liệu `data/` và các file test trong `tests/`. Không có số liệu hay tham số nào được suy diễn ngoài những gì có trong repository.*
```
