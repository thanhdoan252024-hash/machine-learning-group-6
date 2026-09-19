## AI PROMPTING LOG — P02: Manual Standardization không leakage

**Prompt:**

> Với các biến NumPy `X_train` và `X_test` của UCI HAR, hãy tự xây dựng standardization theo z-score mà không dùng sklearn. Viết một class hoặc các hàm có `fit`, `transform`, `fit_transform`. `fit` chỉ được gọi trên training set và phải lưu `mean_` cùng `scale_`; nếu std của feature bằng hoặc gần 0 thì thay bằng 1 để tránh chia 0. Sau đó transform train và test bằng cùng tham số. Hãy kiểm tra training data sau scale có mean gần 0 và std gần 1 đối với các feature có variance khác 0; kiểm tra không có NaN/Inf. In max absolute mean, min/max std sau chuẩn hóa và tạo checkpoint PASS/FAIL. Không dùng sklearn.

**Logic cần giữ nguyên:** fit train statistics → protect zero std → transform train/test → numerical validation.
