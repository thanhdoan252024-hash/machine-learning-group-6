## AI PROMPTING LOG — P05: Kiểm tra transform và inverse-transform

**Prompt:**

> Dùng class `PCAFromScratch` đã xây dựng để kiểm tra API transform/inverse_transform trên UCI HAR. Chọn thử một giá trị k nhỏ như 10 chỉ để kiểm tra chức năng, fit PCA trên `X_train_scaled`, transform cả train và test, kiểm tra shape đầu ra, reconstruct train bằng inverse_transform và tính reconstruction MSE. Không dùng label để fit PCA. In shape và MSE; thêm assertion để chắc chắn số cột sau transform bằng k, số sample không đổi, và MSE hữu hạn. Đây chỉ là functional test, chưa dùng k=10 để kết luận k tối ưu.

**Logic cần giữ nguyên:** test API và shape trước khi bước vào phân tích lựa chọn số components.
