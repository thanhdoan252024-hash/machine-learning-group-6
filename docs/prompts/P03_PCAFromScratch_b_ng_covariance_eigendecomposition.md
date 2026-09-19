## AI PROMPTING LOG — P03: PCAFromScratch bằng covariance + eigendecomposition

**Prompt:**

> Viết class `PCAFromScratch` bằng NumPy, không dùng sklearn. Input là dữ liệu đã standardize. Trong `fit(X)`: tính covariance matrix thủ công theo `(X.T @ X)/(n_samples-1)`; dùng `np.linalg.eigh` vì covariance matrix đối xứng; sắp xếp eigenvalues giảm dần và reorder eigenvectors tương ứng; loại sai số eigenvalue âm rất nhỏ do floating point nếu cần; tính `explained_variance_ratio_` và `cumulative_explained_variance_`; lưu `components_` theo convention mỗi cột là một eigenvector. Hỗ trợ `n_components=None` hoặc một số nguyên hợp lệ. Trong `transform(X)`, project bằng `X @ components_`; trong `inverse_transform(Z)` reconstruct bằng `Z @ components_.T`; thêm `fit_transform`. Hãy validate kích thước và không được dùng label y. Code phải đủ tổng quát để chạy với 561 features của UCI HAR.

**Logic cần giữ nguyên:** covariance → `eigh` → descending sort → EVR → choose \(W_k\) → transform/inverse-transform.
