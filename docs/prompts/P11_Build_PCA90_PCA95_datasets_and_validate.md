## AI PROMPTING LOG — P11: Build PCA90/PCA95 datasets and validate

**Prompt:**

> Tôi có `X_train_scaled`, `X_test_scaled`, `V_all`, `k90`, `k95`, `y_train`, `y_test`. Hãy tạo `X_train_pca90`, `X_test_pca90`, `X_train_pca95`, `X_test_pca95` bằng phép nhân ma trận NumPy với các eigenvector đầu tiên tương ứng; không fit lại PCA và không dùng sklearn. Kiểm tra số sample không đổi, số cột đúng bằng k90/k95, tất cả giá trị hữu hạn, PCA90 phải đúng bằng prefix của PCA95 ở k90 cột đầu trong tolerance, variance từng PC giảm dần và labels không đổi số lượng. Tạo summary DataFrame cho original/PCA90/PCA95 với số chiều, retained variance và reduction ratio.

**Logic cần giữ nguyên:** reuse train-fitted eigenbasis → project train/test → nested-basis consistency → shape/finite/variance checks → summary.
