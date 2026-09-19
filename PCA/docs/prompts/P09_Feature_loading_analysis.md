## AI PROMPTING LOG — P09: Feature loading analysis

**Prompt:**

> Tôi có `V_all` là ma trận eigenvectors PCA from scratch với shape `(n_features, n_features)`, trong đó mỗi cột là một principal component, và `feature_names` chứa tên 561 features UCI HAR. Hãy viết hàm trả về top-N feature có absolute loading lớn nhất cho một PC bất kỳ, bao gồm feature index, feature name, signed loading và absolute loading. Dùng hàm để phân tích PC1 đến PC5, mỗi PC lấy top 10 features. Hiển thị bảng rõ ràng và tạo bar chart riêng cho từng PC. Kiểm tra tổng bình phương loading của mỗi eigenvector gần 1 vì eigenvectors đã normalized. Không dùng sklearn và không dùng label.

**Logic cần giữ nguyên:** eigenvector column → absolute loading rank → preserve sign/name → top-feature interpretation → normalization sanity check.
