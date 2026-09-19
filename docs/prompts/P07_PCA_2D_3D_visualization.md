## AI PROMPTING LOG — P07: PCA 2D/3D visualization

**Prompt:**

> Dùng eigenvectors của `pca_full` đã fit trên training data để project `X_train_scaled` sang ít nhất 3 principal components bằng phép nhân ma trận NumPy, không dùng sklearn. Viết các biểu đồ Matplotlib riêng biệt cho PC1-vs-PC2, PC1-vs-PC3, PC2-vs-PC3 và một scatter 3D PC1-PC2-PC3. Dùng `y_train` và `activity_map` chỉ để nhóm các điểm khi visualize, tuyệt đối không dùng label trong fit PCA. Mỗi activity phải có legend rõ ràng. Có thể giảm alpha và kích thước điểm để xử lý overlap. Thêm kiểm tra shape và tính variance thực tế của PC1, PC2, PC3 để xác nhận thứ tự variance giảm dần.

**Logic cần giữ nguyên:** train PCA scores → first 3 PCs → label only for plotting → inspect separability/overlap → variance sanity check.
