## AI PROMPTING LOG — P06: Explained Variance + threshold selection

**Prompt:**

> Tôi đã có `pca_full` là PCA from scratch fit trên `X_train_scaled`, trong đó `eigenvalues_all_` được sắp xếp giảm dần. Hãy viết code NumPy/Matplotlib, không dùng sklearn, để tính `explained_variance_ratio_all = eigenvalues / sum(eigenvalues)` và `cumulative_variance_all = cumsum(explained_variance_ratio_all)`. Vẽ một Scree Plot của explained variance ratio theo principal component và một biểu đồ cumulative explained variance riêng biệt. Tự tìm số component tối thiểu đạt 80%, 90%, 95%, 99% bằng NumPy, lưu thành dictionary `k_by_threshold`, và hiển thị bảng gồm threshold, số components, variance thực tế giữ lại và phần trăm số chiều được giảm từ 561 features. Thêm assertion rằng cumulative variance không giảm, giá trị cuối gần 1, và các k tăng theo threshold. Không dùng label và không fit PCA trên test.

**Logic cần giữ nguyên:** eigenvalues train → EVR → cumulative EVR → threshold search → compression summary → validation.
