# AI PROMPTING LOG — MASTER

# PCA From Scratch trên UCI HAR — Giai đoạn 1 đến 4

**Mục tiêu:** xây dựng PCA thủ công bằng NumPy trên bộ **UCI Human Activity Recognition Using Smartphones** mà **không sử dụng `sklearn` cho StandardScaler/PCA**.

Notebook này triển khai 4 giai đoạn đầu:

1. Chuẩn bị và kiểm tra dữ liệu.
2. Standardization tự xây dựng.
3. Xây dựng `PCAFromScratch`.
4. Kiểm chứng PCA bằng các tính chất toán học.

> **Quy tắc bài tập**
>
> - Không dùng `sklearn.preprocessing.StandardScaler`.
> - Không dùng `sklearn.decomposition.PCA`.
> - Có thể dùng NumPy cho đại số tuyến tính (`np.mean`, `np.std`, `np.linalg.eigh`, phép nhân ma trận...).
> - PCA chỉ được **fit trên tập train** để tránh data leakage.
> - `y_train/y_test` không tham gia vào quá trình fit PCA; label chỉ dùng cho kiểm tra/phân tích sau này.

---

## Quy ước AI PROMPTING LOG

Mỗi giai đoạn quan trọng có một **AI PROMPTING LOG**. Prompt được viết theo cách:

- mô tả rõ dữ liệu đầu vào;
- nêu đúng ràng buộc kỹ thuật;
- yêu cầu cụ thể đầu ra;
- nêu các kiểm tra cần có;
- đủ độc lập để copy sang một agent/máy khác và tạo ra code có chức năng tương đương.

Điều này giúp log phản ánh **logic xử lý của giai đoạn**, thay vì chỉ ghi một câu kiểu “hãy viết code PCA”.

---

## AI PROMPTING LOG — P00: Thiết kế pipeline tổng thể

**Mục tiêu:** yêu cầu agent tạo kiến trúc notebook PCA from scratch đúng với bài toán UCI HAR.

**Prompt đã sử dụng / có thể tái sử dụng:**

> Tôi đang làm bài giảm chiều dữ liệu bằng PCA trên bộ UCI Human Activity Recognition Using Smartphones. Dataset có train/test riêng và 561 features. Hãy thiết kế notebook Python chạy được trên Google Colab theo pipeline: load dữ liệu → kiểm tra dữ liệu → chuẩn hóa bằng công thức z-score tự viết → tính covariance matrix → eigendecomposition bằng NumPy → sắp xếp eigenvalues/eigenvectors → tính explained variance ratio → transform dữ liệu sang PCA space → inverse transform → kiểm chứng các tính chất toán học của PCA. Không được dùng sklearn StandardScaler hoặc sklearn PCA. PCA và mọi thống kê preprocessing chỉ được fit trên training set; test set chỉ được transform bằng tham số đã học từ train. Hãy chia code thành các hàm/class rõ ràng, có assertion/checkpoint và output chẩn đoán để phát hiện lỗi.

**Đầu ra mong đợi:** một pipeline tái lập được, tách train/test đúng và có checkpoint ở từng giai đoạn.

---

## AI PROMPTING LOG — P01: Load và audit UCI HAR

**Prompt:**

> Viết code Python/Google Colab để chuẩn bị bộ “UCI Human Activity Recognition Using Smartphones” mà không dùng sklearn. Code phải xử lý bền vững ba trường hợp. Thứ nhất, nếu dataset đã được giải nén sẵn thì tự phát hiện thư mục gốc dựa trên `train/X_train.txt`. Thứ hai, nếu cần tải từ UCI thì thử URL chính thức và URL fallback; lưu ý gói UCI có thể là ZIP ngoài chứa tiếp file `UCI HAR Dataset.zip`, vì vậy phải tự phát hiện và giải nén nested ZIP cho đến khi tìm thấy cấu trúc `train/X_train.txt`, `train/y_train.txt`, `test/X_test.txt`, `test/y_test.txt`, `features.txt`, `activity_labels.txt`. Thứ ba, nếu tải tự động thất bại hoặc sau giải nén vẫn không tìm thấy dataset, trên Google Colab hãy gọi `google.colab.files.upload()` để người dùng chọn file ZIP từ máy, rồi tiếp tục tự giải nén và phát hiện dataset. Sau khi tìm thấy dataset, đọc X/y bằng NumPy, metadata bằng Pandas; kiểm tra shape train/test, 561 features, số class, class distribution, NaN, Inf, variance gần 0 và label có khớp số mẫu. Tạo assertion cho các điều kiện quan trọng. Không dùng sklearn.

**Logic cần giữ nguyên khi tái tạo:** phát hiện dữ liệu local → xử lý ZIP/nested ZIP → thử download → nếu thất bại thì manual upload → tự tìm dataset root → parse → audit → assert.

---

## AI PROMPTING LOG — P02: Manual Standardization không leakage

**Prompt:**

> Với các biến NumPy `X_train` và `X_test` của UCI HAR, hãy tự xây dựng standardization theo z-score mà không dùng sklearn. Viết một class hoặc các hàm có `fit`, `transform`, `fit_transform`. `fit` chỉ được gọi trên training set và phải lưu `mean_` cùng `scale_`; nếu std của feature bằng hoặc gần 0 thì thay bằng 1 để tránh chia 0. Sau đó transform train và test bằng cùng tham số. Hãy kiểm tra training data sau scale có mean gần 0 và std gần 1 đối với các feature có variance khác 0; kiểm tra không có NaN/Inf. In max absolute mean, min/max std sau chuẩn hóa và tạo checkpoint PASS/FAIL. Không dùng sklearn.

**Logic cần giữ nguyên:** fit train statistics → protect zero std → transform train/test → numerical validation.

---

## AI PROMPTING LOG — P03: PCAFromScratch bằng covariance + eigendecomposition

**Prompt:**

> Viết class `PCAFromScratch` bằng NumPy, không dùng sklearn. Input là dữ liệu đã standardize. Trong `fit(X)`: tính covariance matrix thủ công theo `(X.T @ X)/(n_samples-1)`; dùng `np.linalg.eigh` vì covariance matrix đối xứng; sắp xếp eigenvalues giảm dần và reorder eigenvectors tương ứng; loại sai số eigenvalue âm rất nhỏ do floating point nếu cần; tính `explained_variance_ratio_` và `cumulative_explained_variance_`; lưu `components_` theo convention mỗi cột là một eigenvector. Hỗ trợ `n_components=None` hoặc một số nguyên hợp lệ. Trong `transform(X)`, project bằng `X @ components_`; trong `inverse_transform(Z)` reconstruct bằng `Z @ components_.T`; thêm `fit_transform`. Hãy validate kích thước và không được dùng label y. Code phải đủ tổng quát để chạy với 561 features của UCI HAR.

**Logic cần giữ nguyên:** covariance → `eigh` → descending sort → EVR → choose \(W_k\) → transform/inverse-transform.

---

## AI PROMPTING LOG — P04: Mathematical validation cho PCA tự xây dựng

**Prompt:**

> Tôi đã có class PCA tự xây dựng bằng covariance matrix và `np.linalg.eigh`, đã fit trên `X_train_scaled`. Hãy viết bộ validation không dựa vào sklearn gồm: (1) kiểm tra covariance matrix đối xứng bằng max absolute symmetry error; (2) kiểm tra eigenvalues không âm ngoài tolerance; (3) kiểm tra trực giao eigenvectors qua `V.T @ V` so với identity; (4) kiểm tra tổng explained variance ratio gần 1; (5) kiểm tra eigenvalues đã giảm dần; (6) transform toàn bộ train bằng tất cả PCs rồi tính covariance của PCA scores, đo max off-diagonal absolute value để xác nhận các PCs gần không tương quan; (7) so sánh diagonal của score covariance với eigenvalues. In từng metric và assertion với tolerance hợp lý. Không dùng sklearn.

**Logic cần giữ nguyên:** kiểm chứng từ các invariant toán học của PCA, không so sánh với thư viện PCA có sẵn.

---

## AI PROMPTING LOG — P05: Kiểm tra transform và inverse-transform

**Prompt:**

> Dùng class `PCAFromScratch` đã xây dựng để kiểm tra API transform/inverse_transform trên UCI HAR. Chọn thử một giá trị k nhỏ như 10 chỉ để kiểm tra chức năng, fit PCA trên `X_train_scaled`, transform cả train và test, kiểm tra shape đầu ra, reconstruct train bằng inverse_transform và tính reconstruction MSE. Không dùng label để fit PCA. In shape và MSE; thêm assertion để chắc chắn số cột sau transform bằng k, số sample không đổi, và MSE hữu hạn. Đây chỉ là functional test, chưa dùng k=10 để kết luận k tối ưu.

**Logic cần giữ nguyên:** test API và shape trước khi bước vào phân tích lựa chọn số components.

---

# Tổng kết Giai đoạn 1–4

Sau khi chạy thành công đến đây, ta đã có:

- dữ liệu UCI HAR được kiểm tra;
- train/test được giữ tách biệt;
- standardization tự xây dựng, fit **chỉ trên train**;
- PCA tự xây dựng bằng covariance + eigendecomposition;
- `fit`, `transform`, `fit_transform`, `inverse_transform`;
- bộ kiểm chứng toán học độc lập với sklearn;
- functional test của transform/reconstruction;
- AI PROMPTING LOG đủ chi tiết để tái tạo từng khối xử lý.

## Điều kiện để chuyển sang Giai đoạn 5

Các dòng sau phải xuất hiện:

```text
CHECKPOINT 1: PASS
CHECKPOINT 2: PASS
CHECKPOINT 3: PASS
FUNCTIONAL TEST: PASS
```

---

# Giai đoạn tiếp theo

Sau checkpoint này mới thực hiện:

- Scree Plot;
- Explained Variance Ratio;
- Cumulative Explained Variance;
- tự tìm `k80`, `k90`, `k95`, `k99`;
- 2D/3D PCA visualization;
- reconstruction error theo nhiều k;
- loading analysis;
- lựa chọn số PC cuối cùng.

**Không chọn `n_components` tối ưu trước khi chạy các phân tích trên.**

---

## AI PROMPTING LOG — P06: Explained Variance + threshold selection

**Prompt:**

> Tôi đã có `pca_full` là PCA from scratch fit trên `X_train_scaled`, trong đó `eigenvalues_all_` được sắp xếp giảm dần. Hãy viết code NumPy/Matplotlib, không dùng sklearn, để tính `explained_variance_ratio_all = eigenvalues / sum(eigenvalues)` và `cumulative_variance_all = cumsum(explained_variance_ratio_all)`. Vẽ một Scree Plot của explained variance ratio theo principal component và một biểu đồ cumulative explained variance riêng biệt. Tự tìm số component tối thiểu đạt 80%, 90%, 95%, 99% bằng NumPy, lưu thành dictionary `k_by_threshold`, và hiển thị bảng gồm threshold, số components, variance thực tế giữ lại và phần trăm số chiều được giảm từ 561 features. Thêm assertion rằng cumulative variance không giảm, giá trị cuối gần 1, và các k tăng theo threshold. Không dùng label và không fit PCA trên test.

**Logic cần giữ nguyên:** eigenvalues train → EVR → cumulative EVR → threshold search → compression summary → validation.

---

## AI PROMPTING LOG — P07: PCA 2D/3D visualization

**Prompt:**

> Dùng eigenvectors của `pca_full` đã fit trên training data để project `X_train_scaled` sang ít nhất 3 principal components bằng phép nhân ma trận NumPy, không dùng sklearn. Viết các biểu đồ Matplotlib riêng biệt cho PC1-vs-PC2, PC1-vs-PC3, PC2-vs-PC3 và một scatter 3D PC1-PC2-PC3. Dùng `y_train` và `activity_map` chỉ để nhóm các điểm khi visualize, tuyệt đối không dùng label trong fit PCA. Mỗi activity phải có legend rõ ràng. Có thể giảm alpha và kích thước điểm để xử lý overlap. Thêm kiểm tra shape và tính variance thực tế của PC1, PC2, PC3 để xác nhận thứ tự variance giảm dần.

**Logic cần giữ nguyên:** train PCA scores → first 3 PCs → label only for plotting → inspect separability/overlap → variance sanity check.

---

## AI PROMPTING LOG — P08: Reconstruction error curve

**Prompt:**

> Tôi đã có `X_train_scaled`, `X_test_scaled` và ma trận eigenvectors `V_all` từ PCA fit trên train. Không được fit lại PCA cho từng k. Hãy đánh giá reconstruction error cho một tập các k đại diện như 1,2,3,5,10,20,30,50,75,100,150,200,300,400,500,561 và tự bổ sung các k đạt 80%,90%,95%,99% variance từ `k_by_threshold`. Với mỗi k, lấy `W_k = V_all[:, :k]`, project train/test bằng `X @ W_k`, reconstruct bằng `Z @ W_k.T`, tính MSE riêng cho train và test. Lưu kết quả vào DataFrame và vẽ reconstruction MSE theo k. Kiểm tra MSE hữu hạn và không tăng khi k tăng; ở k = toàn bộ số feature, train reconstruction MSE phải gần 0. Không dùng sklearn.

**Logic cần giữ nguyên:** reuse nested PCA basis → reconstruct train/test for multiple k → MSE curve → monotonic validation → full-basis sanity check.

---

## AI PROMPTING LOG — P09: Feature loading analysis

**Prompt:**

> Tôi có `V_all` là ma trận eigenvectors PCA from scratch với shape `(n_features, n_features)`, trong đó mỗi cột là một principal component, và `feature_names` chứa tên 561 features UCI HAR. Hãy viết hàm trả về top-N feature có absolute loading lớn nhất cho một PC bất kỳ, bao gồm feature index, feature name, signed loading và absolute loading. Dùng hàm để phân tích PC1 đến PC5, mỗi PC lấy top 10 features. Hiển thị bảng rõ ràng và tạo bar chart riêng cho từng PC. Kiểm tra tổng bình phương loading của mỗi eigenvector gần 1 vì eigenvectors đã normalized. Không dùng sklearn và không dùng label.

**Logic cần giữ nguyên:** eigenvector column → absolute loading rank → preserve sign/name → top-feature interpretation → normalization sanity check.

---

## AI PROMPTING LOG — P10: Final PCA candidate selection

**Prompt:**

> Tôi đã chạy PCA from scratch trên UCI HAR và có `k_by_threshold`, `cumulative_variance_all`, `reconstruction_df`, `n_original_features`. Hãy tạo bảng so sánh PCA80, PCA90, PCA95, PCA99 với: target variance, k, actual retained variance, dimensions reduced, reduction ratio, train reconstruction MSE và test reconstruction MSE. Không dùng sklearn. Không tuyên bố một k là tối ưu tuyệt đối trước classification. Đặt PCA95 là cấu hình primary vì ưu tiên bảo toàn thông tin, PCA90 là alternative vì ưu tiên compression. Kiểm tra PCA95 có retained variance >= 0.95, PCA90 >= 0.90 và k90 <= k95. In giải thích ngắn về trade-off.

**Logic cần giữ nguyên:** threshold candidates → reconstruction metrics → trade-off minh bạch → primary/alternative, không overclaim.

---

## AI PROMPTING LOG — P11: Build PCA90/PCA95 datasets and validate

**Prompt:**

> Tôi có `X_train_scaled`, `X_test_scaled`, `V_all`, `k90`, `k95`, `y_train`, `y_test`. Hãy tạo `X_train_pca90`, `X_test_pca90`, `X_train_pca95`, `X_test_pca95` bằng phép nhân ma trận NumPy với các eigenvector đầu tiên tương ứng; không fit lại PCA và không dùng sklearn. Kiểm tra số sample không đổi, số cột đúng bằng k90/k95, tất cả giá trị hữu hạn, PCA90 phải đúng bằng prefix của PCA95 ở k90 cột đầu trong tolerance, variance từng PC giảm dần và labels không đổi số lượng. Tạo summary DataFrame cho original/PCA90/PCA95 với số chiều, retained variance và reduction ratio.

**Logic cần giữ nguyên:** reuse train-fitted eigenbasis → project train/test → nested-basis consistency → shape/finite/variance checks → summary.

---

## AI PROMPTING LOG — P12: Export reproducible PCA artifacts

**Prompt:**

> Tôi đã có pipeline PCA from scratch hoàn chỉnh với `scaler.mean_`, `scaler.scale_`, `pca_full.eigenvalues_all_`, `V_all`, `cumulative_variance_all`, `k90`, `k95`, các ma trận PCA90/PCA95 và labels. Hãy tạo thư mục output trong Colab, lưu arrays thành `.npy`, lưu toàn bộ preprocessing/PCA parameters thành một file `.npz`, và lưu các bảng `pca_candidate_df`, `final_summary_df`, `reconstruction_df` thành CSV. Không dùng sklearn hoặc joblib. Thêm `README.txt` giải thích file nào dùng cho classification, nhấn mạnh PCA chỉ fit trên train và test chỉ được transform. Sau khi lưu, kiểm tra từng file tồn tại và in đường dẫn.

**Logic cần giữ nguyên:** export transformed datasets + labels + train-fitted preprocessing/PCA parameters + evaluation tables + reproducibility note.
