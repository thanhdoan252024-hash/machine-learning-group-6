## AI PROMPTING LOG — P01: Load và audit UCI HAR

**Prompt:**

> Viết code Python/Google Colab để chuẩn bị bộ “UCI Human Activity Recognition Using Smartphones” mà không dùng sklearn. Code phải xử lý bền vững ba trường hợp. Thứ nhất, nếu dataset đã được giải nén sẵn thì tự phát hiện thư mục gốc dựa trên `train/X_train.txt`. Thứ hai, nếu cần tải từ UCI thì thử URL chính thức và URL fallback; lưu ý gói UCI có thể là ZIP ngoài chứa tiếp file `UCI HAR Dataset.zip`, vì vậy phải tự phát hiện và giải nén nested ZIP cho đến khi tìm thấy cấu trúc `train/X_train.txt`, `train/y_train.txt`, `test/X_test.txt`, `test/y_test.txt`, `features.txt`, `activity_labels.txt`. Thứ ba, nếu tải tự động thất bại hoặc sau giải nén vẫn không tìm thấy dataset, trên Google Colab hãy gọi `google.colab.files.upload()` để người dùng chọn file ZIP từ máy, rồi tiếp tục tự giải nén và phát hiện dataset. Sau khi tìm thấy dataset, đọc X/y bằng NumPy, metadata bằng Pandas; kiểm tra shape train/test, 561 features, số class, class distribution, NaN, Inf, variance gần 0 và label có khớp số mẫu. Tạo assertion cho các điều kiện quan trọng. Không dùng sklearn.

**Logic cần giữ nguyên khi tái tạo:** phát hiện dữ liệu local → xử lý ZIP/nested ZIP → thử download → nếu thất bại thì manual upload → tự tìm dataset root → parse → audit → assert.
