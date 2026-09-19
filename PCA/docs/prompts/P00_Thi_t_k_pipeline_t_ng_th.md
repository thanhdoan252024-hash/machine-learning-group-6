## AI PROMPTING LOG — P00: Thiết kế pipeline tổng thể

**Mục tiêu:** yêu cầu agent tạo kiến trúc notebook PCA from scratch đúng với bài toán UCI HAR.

**Prompt đã sử dụng / có thể tái sử dụng:**

> Tôi đang làm bài giảm chiều dữ liệu bằng PCA trên bộ UCI Human Activity Recognition Using Smartphones. Dataset có train/test riêng và 561 features. Hãy thiết kế notebook Python chạy được trên Google Colab theo pipeline: load dữ liệu → kiểm tra dữ liệu → chuẩn hóa bằng công thức z-score tự viết → tính covariance matrix → eigendecomposition bằng NumPy → sắp xếp eigenvalues/eigenvectors → tính explained variance ratio → transform dữ liệu sang PCA space → inverse transform → kiểm chứng các tính chất toán học của PCA. Không được dùng sklearn StandardScaler hoặc sklearn PCA. PCA và mọi thống kê preprocessing chỉ được fit trên training set; test set chỉ được transform bằng tham số đã học từ train. Hãy chia code thành các hàm/class rõ ràng, có assertion/checkpoint và output chẩn đoán để phát hiện lỗi.

**Đầu ra mong đợi:** một pipeline tái lập được, tách train/test đúng và có checkpoint ở từng giai đoạn.
