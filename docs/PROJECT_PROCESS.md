# QUÁ TRÌNH DỰ ÁN t-SNE: GIẢM CHIỀU VÀ TRỰC QUAN HÓA CHỮ SỐ VIẾT TAY

---

## 1. GIỚI THIỆU VÀ MỤC TIÊU

Dự án này triển khai thuật toán **t-Distributed Stochastic Neighbor Embedding (t-SNE)** thuần túy bằng NumPy và áp dụng để giảm chiều dữ liệu **Optical Recognition of Handwritten Digits (Optdigits)** từ 64 chiều xuống 2 chiều nhằm mục đích trực quan hóa. Mục tiêu cụ thể bao gồm:

- Xây dựng t-SNE exact (không dùng Barnes-Hut) hoàn toàn từ NumPy, không dùng thư viện scikit-learn hay thư viện giảm chiều có sẵn.
- Thử nghiệm hệ thống nhiều giá trị perplexity để đánh giá tác động của tham số này đến chất lượng embedding.
- Sử dụng các chỉ số định lượng (KL Divergence, Trustworthiness) và định tính (hình dạng embedding) để lựa chọn perplexity tối ưu.
- Trực quan hóa kết quả cuối cùng trên toàn bộ tập dữ liệu 5.620 mẫu.

Phạm vi: exact t-SNE có độ phức tạp bộ nhớ O(N²), phù hợp với tập dữ liệu vừa (vài nghìn mẫu). Dự án không triển khai Barnes-Hut approximation hay các biến thể t-SNE khác.

---

## 2. BỘ DỮ LIỆU VÀ TIỀN XỬ LÝ

### 2.1 Nguồn dữ liệu gốc

Bộ dữ liệu Optdigits lấy từ UCI Machine Learning Repository, bao gồm 2 file:

| File | Số mẫu | Mô tả |
|------|--------|-------|
| `optdigits.tra` | 3.823 | Tập huấn luyện |
| `optdigits.tes` | 1.797 | Tập kiểm thử |

Mỗi mẫu gồm 64 đặc trưng nguyên (0–16) biểu diễn ảnh 8×8 (chia 32×32 thành các khối 4×4, đếm pixel bật trong mỗi khối) và 1 nhãn chữ số 0–9. Theo `optdigits.names`, dữ liệu **không có giá trị thiếu**.

### 2.2 Quy trình tiền xử lý (trong `tsne_digits.ipynb.ipynb`)

1. Đọc 2 file `.tra` và `.tes` bằng `pandas.read_csv(header=None)`.
2. Kiểm tra giá trị null: cả hai file đều trả về 0.
3. Gộp 2 tập bằng `pd.concat([train, test], ignore_index=True, axis=0)` → **5.620 mẫu**.
4. Đặt tên cột: `Pixel_1` đến `Pixel_64` cho đặc trưng, `label` cho nhãn.
5. Lưu ra `data/processed/optdigits.csv` (file CSV này có thêm cột index do `to_csv` không truyền `index=False`; hàm `load_optdigits_csv()` trong `core/evaluation.py` lấy mọi cột trừ cột cuối làm X, nên cột index bị tính vào đặc trưng – đây là một điểm cần lưu ý).

### 2.3 Dữ liệu dùng cho thí nghiệm

- Thí nghiệm đánh giá perplexity: chỉ dùng **2.000 mẫu đầu** (`X_test = X[:2000]`) để chạy nhanh.
- Chạy cuối cùng trên toàn bộ dữ liệu: **5.620 mẫu** với perplexity đã chọn.

---

## 3. XÂY DỰNG t-SNE BẰNG NUMPY (`core/tsne_numpy.py`)

Module `core/tsne_numpy.py` chứa toàn bộ logic t-SNE, được tổ chức thành các hàm module-level và lớp `TSNE`.

### 3.1 Các hàm cốt lõi

| Hàm | Chức năng |
|-----|-----------|
| `squared_euclidean_distances(X)` | Tính ma trận khoảng cách Euclid bình phương D_ij = ‖x_i - x_j‖² bằng công thức vector hóa: ‖x_i‖² + ‖x_j‖² - 2⟨x_i, x_j⟩. Điền 0 cho đường chéo. |
| `entropy_and_probabilities(distances, beta)` | Tính entropy H và phân phối xác suất p_j = exp(-β·d_j) / Σ_k exp(-β·d_k). Dịch logits trước exp để ổn định số. |
| `binary_search_beta(distances, perplexity, ...)` | Tìm β sao cho H ≈ ln(perplexity) bằng binary search (tối đa 50 vòng, tolerance 1e-5). H giảm đơn điệu khi β tăng. |
| `conditional_probabilities(D, perplexity, ...)` | Tính p(j\|i) cho mọi i (p(i\|i)=0), mỗi điểm có β riêng. Trả về ma trận P_cond và vector betas. |
| `symmetric_probabilities(P_cond)` | Đối xứng hóa: p_ij = (p(j\|i) + p(i\|j)) / (2N). Tổng P = 1, đường chéo = 0. |
| `initialize_embedding(n, n_components, rng)` | Khởi tạo y_i ~ N(0, 10⁻⁴²) gần gốc tọa độ. |
| `student_t_probabilities(Y)` | Tính phân phối Student-t 1 bậc tự do: a_ij = (1 + ‖y_i - y_j‖²)⁻¹, q_ij = a_ij / Σ_{k≠l} a_kl. Trả về Q và numerator (dùng cho gradient). |
| `kl_divergence(P, Q, epsilon=1e-12)` | KL(P‖Q) = Σ_{i≠j} p_ij ln(p_ij / q_ij). Chỉ cộng các vị trí p_ij > 0. |
| `tsne_gradient(P, Q, Y, numerator)` | Gradient: dC/dy_i = 4 Σ_j (p_ij - q_ij)(y_i - y_j) / (1 + ‖y_i - y_j‖²). Vector hóa: W = (P - Q) ⊙ numerator; grad = 4 (rowsum(W) * Y - W @ Y). |
| `gradient_descent_step(Y, velocity, gradient, gains, lr, momentum, min_gain)` | Một bước GD với momentum và adaptive gains gốc: gains tăng 0.2 nếu dấu gradient đổi chiều so với velocity, giảm 0.8 ngược lại; clamp min_gain; cập nhật velocity và Y; canh giữa Y bằng cách trừ mean. |

### 3.2 Lớp `TSNE`

```python
class TSNE:
    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 learning_rate=200.0,
                 n_iter=1000,
                 early_exaggeration=12.0,
                 early_exaggeration_iter=250,
                 momentum=0.5,
                 final_momentum=0.8,
                 min_gain=0.01,
                 patience=50,
                 min_kl_improvement=1e-7,
                 beta_tolerance=1e-5,
                 beta_search_iter=50,
                 random_state=None,
                 verbose=False):
```

**Thuộc tính sau khi `fit`:**
- `embedding_`: embedding N×n_components.
- `P_`: ma trận xác suất đối xứng N×N.
- `betas_`, `sigmas_`: β và σ của từng điểm.
- `kl_divergence_`: KL cuối cùng.
- `best_kl_divergence_`: KL tốt nhất trong quá trình chạy.
- `kl_history_`: list (iteration, KL) sau giai đoạn early exaggeration.
- `n_iter_`: số vòng cập nhật thực tế.
- `stopped_early_`: bool, dừng sớm hay không.
- `gains_`: gains cuối cùng.

**Phương thức:**
- `fit(X)`: học embedding, trả về self.
- `fit_transform(X)`: fit và trả về embedding.
- `fit_tranform(X)`: alias (thiếu chữ 's') trỏ về `fit_transform`.

### 3.3 Quy trình `fit` (tóm tắt)

1. Validate input: X phải 2D, ≥2 mẫu, finite, perplexity > 0, 3·perplexity < n_samples.
2. Tính ma trận khoảng cách D.
3. Tính P_cond, betas, sigmas bằng binary search perplexity.
4. Đối xứng hóa P.
5. Khởi tạo Y ~ N(0, 1e-4).
6. Vòng lặp tối đa `n_iter`:
   - Nếu iteration < `early_exaggeration_iter`: dùng P × early_exaggeration, momentum = `momentum`.
   - Ngược lại: dùng P gốc, momentum = `final_momentum`.
   - Tính Q, numerator từ Y.
   - **Chỉ tính KL và kiểm tra early stopping SAU giai đoạn early exaggeration** (vì P đã nhân hệ số trước đó không cùng thang đo).
   - Tính gradient, cập nhật Y, velocity, gains.
   - Early stopping: nếu KL không cải thiện > `min_kl_improvement` trong `patience` vòng liên tiếp → dừng.
7. Lưu embedding, KL cuối, các thuộc tính.

---

## 4. CÁC BƯỚC HOẠT ĐỘNG CỦA THUẬT TOÁN

1. **Tính khoảng cách**: Ma trận D = ‖x_i - x_j‖² kích thước N×N.
2. **Tìm σ_i (hoặc β_i) cho mỗi điểm**: Binary search sao cho perplexity hiệu ứng của p(j\|i) khớp perplexity mục tiêu. Mỗi điểm có bandwidth riêng.
3. **Xây dựng P đối xứng**: p_ij = (p(j\|i) + p(i\|j)) / (2N). Tổng P = 1.
4. **Khởi tạo embedding**: Y ~ N(0, 10⁻⁴) gần gốc.
5. **Tối ưu KL(P‖Q) bằng Gradient Descent**:
   - Tính Q từ Y bằng kernel Student-t 1 bậc tự do (đuôi dày giảm crowding).
   - Tính gradient ∇KL.
   - Cập nhật Y với momentum (0.5 giai đoạn đầu, 0.8 giai đoạn sau) và adaptive gains.
   - **Early exaggeration**: nhân P với 12 trong 250 vòng đầu để tạo khoảng trống, giúp các cụm tách nhau rõ ràng.
   - **Early stopping**: theo dõi KL sau giai đoạn early exaggeration; dừng nếu không cải thiện đủ trong 50 vòng.
6. **Trả về embedding 2D**.

---

## 5. KIỂM TRA THUẬT TOÁN (TESTS)

Project có 22 test trong thư mục `tests/` dùng `pytest`:

### 5.1 `tests/test_tsne_core.py` (15 test)

Kiểm tra các hàm module-level:
- `squared_euclidean_distances`: giá trị biết trước, đối xứng, không âm.
- `entropy_and_probabilities`: tổng xác suất = 1, entropy ≥ 0.
- `binary_search_beta`: entropy khớp ln(perplexity) trong tolerance.
- `conditional_probabilities`: mỗi hàng (trừ đường chéo) tổng = 1, đường chéo = 0.
- `symmetric_probabilities`: đối xứng, tổng = 1, đường chéo = 0.
- `student_t_probabilities`: Q tổng = 1, đường chéo = 0, numerator đúng công thức.
- `kl_divergence`: = 0 khi P=Q, không âm.
- `initialize_embedding`: shape đúng, finite.
- `tsne_gradient`: shape đúng, finite.
- `gradient_descent_step`: cập nhật state, embedding thay đổi.

### 5.2 `tests/test_tsne_output.py` (7 test)

Kiểm tra hành vi đầu ra của lớp `TSNE` trên dữ liệu tự sinh (2 cụm Gaussian tách biệt, 40 mẫu, 5 đặc trưng):
- Shape output (n_samples, 2).
- Số mẫu không đổi.
- Default n_components = 2.
- Không NaN/Inf.
- Không collapse (có ít nhất 2 điểm phân biệt, std > 0).
- `fit()` trả về self và set `embedding_`.
- `fit_transform()` khớp `fit().embedding_`.
- Các thuộc tính sau fit: `P_`, `betas_`, `kl_divergence_`, `n_iter_`.
- Alias `fit_tranform()` tồn tại.

Chạy test:
```bash
python -m pytest -q
```
*Lưu ý: `pytest` chưa có trong `requirement.txt`.*

---

## 6. THIẾT KẾ THÍ NGHIỆM VÀ CÁC GIÁ TRỊ PERPLEXITY

### 6.1 Tham số cố định trong thí nghiệm

| Tham số | Giá trị |
|---------|---------|
| n_components | 2 |
| n_iter | 500 (thí nghiệm), 1000 (chạy cuối) |
| early_exaggeration | 12.0 |
| early_exaggeration_iter | 250 |
| momentum | 0.5 |
| final_momentum | 0.8 |
| min_gain | 0.01 |
| patience | 50 |
| min_kl_improvement | 1e-7 |
| random_state | 42 |
| verbose | True |

### 6.2 Danh sách perplexity thử nghiệm

Trong notebook: `[5, 10, 15, 20, 25, 30, 45, 50, 60, 70, 80, 90, 100]` (13 giá trị).

*Trong `core/evaluation.py` mặc định là `(5, 10, 20, 30, 40, 50)` nhưng notebook ghi đè bằng danh sách trên.*

### 6.3 Hàm đánh giá `run_evaluation` (`core/evaluation.py`)

- Chạy TSNE cho từng perplexity, đo thời gian bằng `time.perf_counter()`.
- Tính KL divergence từ `model.kl_divergence_`.
- Tính Trustworthiness k=50 theo định nghĩa Venna & Kaski (2001):
  - T(k) = 1 - 2/(n·k·(2n-3k-1)) · Σ_i Σ_{j∈U_k(i)} (r(i,j) - k)
  - U_k(i): điểm trong k-NN của i trong không gian nhúng nhưng **KHÔNG** trong k-NN của i trong không gian gốc.
  - r(i,j): thứ hạng của j trong k-NN gốc của i.
- Lưu CSV: `outputs/results/experiment_evaluation.csv`.
- Vẽ 4 hình lưu vào `outputs/figures/`:
  1. Lưới embedding các perplexity (`tsne_perplexity_grid.png`).
  2. Perplexity vs KL Divergence (`perplexity_vs_kl.png`).
  3. Perplexity vs Runtime (`perplexity_vs_runtime.png`).
  4. Perplexity vs Trustworthiness (`perplexity_vs_trustworthiness.png`).

---

## 7. CÁC CHỈ SỐ: KL DIVERGENCE, TRUSTWORTHINESS, RUNTIME

### 7.1 Kết quả thực tế (từ `outputs/results/experiment_evaluation.csv`)

Thí nghiệm trên 2.000 mẫu, 500 vòng, random_state=42:

| Perplexity | KL Divergence | Trustworthiness (k=50) | Runtime (giây) |
|-----------:|--------------:|----------------------:|----------------:|
| 5   | 0.841149 | 0.948917 | 60.26 |
| 10  | 0.741336 | 0.984014 | 64.90 |
| 15  | 0.681136 | 0.993240 | 55.68 |
| 20  | 0.640092 | 0.995545 | 54.76 |
| **25**  | **0.596478** | **0.996855** | **56.28** |
| 30  | 0.577071 | 0.997093 | 57.56 |
| 45  | 0.467548 | 0.998047 | 143.12 |
| 50  | 0.441998 | 0.998217 | 154.10 |
| 60  | 0.392190 | 0.998450 | 242.23 |
| 70  | 0.349039 | 0.998594 | 169.38 |
| 80  | 0.312239 | 0.998683 | 218.22 |
| 90  | 0.276323 | 0.998710 | 265.03 |
| 100 | 0.254121 | 0.998711 | 240.82 |

### 7.2 Phân tích các chỉ số

#### KL Divergence
- Giảm **liên tục** từ 0.841 (perp=5) xuống 0.254 (perp=100).
- Perplexity lớn cho phép phân phối P rộng hơn, khớp tốt hơn với Q trong không gian 2D theo hàm mục tiêu t-SNE.
- Tuy nhiên, KL thấp không đảm bảo embedding dễ diễn giải nhất.

#### Trustworthiness (k=50)
- Tăng nhanh từ 0.949 (perp=5) lên 0.997 (perp=25).
- Từ perp=25 đến 100: chỉ tăng thêm ~0.002 (0.9969 → 0.9987).
- **Perplexity 25 đã đạt mức bảo toàn lân cận rất cao**; cải thiện thêm ở perplexity lớn là vi lượng.

#### Runtime
- Perplexity 5–30: 55–65 giây (không chênh lệch lớn, dao động do trạng thái máy).
- Perplexity ≥45: runtime tăng mạnh và biến động (143–265 giây).
- Runtime phụ thuộc máy, trạng thái thực thi; dùng để so sánh tương đối trong lần chạy này.

---

## 8. TRỰC QUAN HÓA VÀ PHÂN TÍCH KẾT QUẢ

### 8.1 Lưới embedding các perplexity (`tsne_perplexity_grid.png`)

Hình hiển thị 13 embedding 2D tương ứng 13 perplexity, tô màu theo nhãn 0–9.

**Quan sát:**
- **Perplexity 5–15**: Embedding bị chia thành nhiều cụm nhỏ, rời rạc, cấu trúc toàn cục chưa liên kết.
- **Perplexity 20–30**: Các cụm chữ số trở nên liền mạch, tách biệt rõ rệt, vẫn giữ được nhiều cấu trúc cục bộ.
- **Perplexity 45–100**: Các điểm tổ chức thành **vài dải cong dài**, các lớp chữ số trộn lẫn, bố cục bị nén, khó quan sát sự đa dạng cấu trúc cục bộ.

### 8.2 Biểu đồ metric

- `perplexity_vs_kl.png`: Đường giảm đơn điệu, không có điểm ngoặt rõ rệt.
- `perplexity_vs_trustworthiness.png`: Tăng nhanh ở perplexity thấp, bão hòa từ ~25 trở đi.
- `perplexity_vs_runtime.png`: Phẳng ở perplexity thấp, tăng mạnh và biến động ở perplexity lớn.

### 8.3 Embedding cuối cùng: Perplexity 25 (`tsne_perplexity_25.png`)

Chạy trên **toàn bộ 5.620 mẫu**, 1000 vòng, perplexity=25.

**Đặc điểm:**
- 10 cụm chữ số 0–9 tách biệt rõ ràng.
- Cấu trúc cục bộ trong từng cụm được bảo toàn (ví dụ: các biến thể viết của cùng một chữ số nằm gần nhau).
- Không bị nén thành dải cong như perplexity ≥45.
- Trustworthiness 0.996855, KL 0.596478, runtime ~56 giây trên tập 2000 mẫu (extrapolate: toàn bộ ~3-4 phút).

---

## 9. LÝ DO CHỌN PERPLEXITY CUỐI CÙNG = 25

Dựa trên **đối chiếu đồng thời 3 tiêu chí**:

| Tiêu chí | Perplexity 25 | Perplexity 30 | Perplexity 45+ |
|----------|---------------|---------------|----------------|
| **Trustworthiness** | 0.996855 (rất cao, gần bão hòa) | 0.997093 (+0.0002) | 0.9980+ (+0.001) |
| **KL Divergence** | 0.596 (tốt) | 0.577 (tốt hơn chút) | 0.47→0.25 (tốt hơn nhiều) |
| **Runtime** | 56.28s (thấp, ổn định) | 57.56s (tương đương) | 143–265s (cao, biến động) |
| **Hình dạng embedding** | Cụm tách biệt, liền mạch, dễ quan sát | Tương tự 25 | **Dải cong, trộn lẫn, khó quan sát** |

**Kết luận:** Perplexity 25 đạt **cân bằng tối ưu**:
- Bảo toàn lân cận gần như tối đa (Trustworthiness > 0.996).
- Chi phí tính toán thấp và ổn định.
- **Quan trọng nhất**: Embedding trực quan rõ ràng, các cụm chữ số tách biệt, không bị nén cấu trúc – phù hợp mục đích trực quan hóa dữ liệu.

*Ghi chú: Notebook có ghi "Perplexity = 10 được chọn" trong comment nhưng code và hình ảnh cuối cùng (`tsne_perplexity_25.png`, tiêu đề "Perplexity = 25") đều dùng **perplexity 25**. Báo cáo này tuân theo code và kết quả thực tế.*

---

## 10. KẾT QUẢ, HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

### 10.1 Kết quả đạt được

- ✅ Triển khai thành công **exact t-SNE thuần NumPy** (O(N²) bộ nhớ).
- ✅ Hệ thống hóa quy trình: tiền xử lý → thử perplexity → đánh giá metric → trực quan hóa → chọn tham số → chạy toàn bộ.
- ✅ Đánh giá định lượng (KL, Trustworthiness, Runtime) và định tính (hình embedding) trên 13 giá trị perplexity.
- ✅ Lựa chọn **perplexity 25** dựa trên bằng chứng thực nghiệm.
- ✅ Tạo embedding 2D chất lượng cao cho 5.620 mẫu chữ số, 10 cụm tách biệt rõ rệt.
- ✅ Bộ test 22 case kiểm tra tính đúng đắn của các phép tính cốt lõi và API.

### 10.2 Hạn chế

| Hạn chế | Chi tiết |
|---------|----------|
| **Độ phức tạp O(N²)** | Không thể áp dụng cho tập dữ liệu lớn (>10k mẫu) do bộ nhớ và thời gian. |
| **Không có Barnes-Hut** | Chỉ có exact t-SNE; thiếu approximate t-SNE cho dữ liệu lớn. |
| **Cột index trong CSV** | `data/processed/optdigits.csv` có cột index do `to_csv` không `index=False`; `load_optdigits_csv()` lấy cột này làm đặc trưng (thành feature thứ 65). |
| **Thí nghiệm trên tập con** | Đánh giá perplexity chỉ dùng 2.000/5.620 mẫu; kết quả có thể khác nếu chạy full. |
| **Chỉ 1 seed** | Chạy với `random_state=42` đơn lẻ; không báo cáo độ biến thiên qua nhiều seed. |
| **Trustworthiness k=50 cố định** | Không thử các k khác (5, 10, 100) để xem bức tranh toàn diện hơn. |
| **Không so sánh baseline** | Chưa so sánh với PCA, UMAP, hoặc scikit-learn TSNE để benchmark. |
| **Early stopping logic** | Chỉ kiểm tra KL sau early exaggeration; có thể bỏ qua cải thiện trong giai đoạn đầu. |
| **Không có learning rate schedule** | Learning rate cố định 200.0 toàn quá trình. |

### 10.3 Hướng phát triển

1. **Tối ưu hiệu năng**: Triển khai **Barnes-Hut t-SNE** (O(N log N)) để xử lý dữ liệu lớn.
2. **Sửa tiền xử lý**: Thêm `index=False` khi `to_csv` hoặc bỏ cột index khi load.
3. **Mở rộng đánh giá**:
   - Chạy nhiều `random_state` và báo cáo mean ± std.
   - Thử nhiều giá trị `k` cho Trustworthiness (5, 10, 20, 50, 100).
   - Thêm **Continuity** (chỉ số đối xứng của Trustworthiness).
   - So sánh với **PCA, UMAP, sklearn.manifold.TSNE**.
4. **Tinh chỉnh hyperparameter**: Grid search learning_rate, early_exaggeration, n_iter.
5. **Parametric t-SNE**: Huấn luyện mạng neural học mapping để transform mẫu mới không cần fit lại.
6. **Tích hợp vào pipeline ML**: Dùng embedding làm feature cho classifier downstream.
7. **Cải thiện test**: Thêm test cho early stopping, các giá trị edge-case của perplexity, numerical stability.
8. **Documentation**: Viết docstring cho tất cả hàm public, thêm type hint đầy đủ.

---

## PHỤ LỤC: CẤU TRÚC THƯ MỤC DỰ ÁN

```
t-SNE/
├── core/
│   ├── tsne_numpy.py      # Triển khai t-SNE chính
│   └── evaluation.py      # Đa perplexity evaluation + plotting
├── data/
│   ├── raw/
│   │   ├── optdigits.tra  # Train set (3823 samples)
│   │   ├── optdigits.tes  # Test set (1797 samples)
│   │   └── optdigits.names# Mô tả dataset
│   └── processed/
│       └── optdigits.csv  # CSV đã gộp (5620 samples, có cột index)
├── outputs/
│   ├── figures/
│   │   ├── tsne_perplexity_grid.png
│   │   ├── perplexity_vs_kl.png
│   │   ├── perplexity_vs_runtime.png
│   │   ├── perplexity_vs_trustworthiness.png
│   │   └── tsne_perplexity_25.png
│   └── results/
│       └── experiment_evaluation.csv
├── tests/
│   ├── test_tsne_core.py      # 15 test hàm cốt lõi
│   └── test_tsne_output.py    # 7 test đầu ra TSNE
├── requirement.txt            # Dependencies (thiếu pytest)
├── tsne_digits.ipynb.ipynb    # Notebook chính
└── docs/
    └── PROJECT_PROCESS.md     # File này
```

---

*Báo cáo được tạo dựa trên phân tích code, notebook, dữ liệu và file kết quả thực tế của dự án. Không chứa số liệu, tham số hoặc chức năng bịa đặt.*