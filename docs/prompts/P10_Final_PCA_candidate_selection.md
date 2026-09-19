## AI PROMPTING LOG — P10: Final PCA candidate selection

**Prompt:**

> Tôi đã chạy PCA from scratch trên UCI HAR và có `k_by_threshold`, `cumulative_variance_all`, `reconstruction_df`, `n_original_features`. Hãy tạo bảng so sánh PCA80, PCA90, PCA95, PCA99 với: target variance, k, actual retained variance, dimensions reduced, reduction ratio, train reconstruction MSE và test reconstruction MSE. Không dùng sklearn. Không tuyên bố một k là tối ưu tuyệt đối trước classification. Đặt PCA95 là cấu hình primary vì ưu tiên bảo toàn thông tin, PCA90 là alternative vì ưu tiên compression. Kiểm tra PCA95 có retained variance >= 0.95, PCA90 >= 0.90 và k90 <= k95. In giải thích ngắn về trade-off.

**Logic cần giữ nguyên:** threshold candidates → reconstruction metrics → trade-off minh bạch → primary/alternative, không overclaim.
