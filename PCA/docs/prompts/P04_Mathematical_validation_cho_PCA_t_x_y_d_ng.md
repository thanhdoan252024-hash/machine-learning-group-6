## AI PROMPTING LOG — P04: Mathematical validation cho PCA tự xây dựng

**Prompt:**

> Tôi đã có class PCA tự xây dựng bằng covariance matrix và `np.linalg.eigh`, đã fit trên `X_train_scaled`. Hãy viết bộ validation không dựa vào sklearn gồm: (1) kiểm tra covariance matrix đối xứng bằng max absolute symmetry error; (2) kiểm tra eigenvalues không âm ngoài tolerance; (3) kiểm tra trực giao eigenvectors qua `V.T @ V` so với identity; (4) kiểm tra tổng explained variance ratio gần 1; (5) kiểm tra eigenvalues đã giảm dần; (6) transform toàn bộ train bằng tất cả PCs rồi tính covariance của PCA scores, đo max off-diagonal absolute value để xác nhận các PCs gần không tương quan; (7) so sánh diagonal của score covariance với eigenvalues. In từng metric và assertion với tolerance hợp lý. Không dùng sklearn.

**Logic cần giữ nguyên:** kiểm chứng từ các invariant toán học của PCA, không so sánh với thư viện PCA có sẵn.
