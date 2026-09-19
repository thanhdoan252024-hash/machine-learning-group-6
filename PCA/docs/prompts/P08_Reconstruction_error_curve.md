## AI PROMPTING LOG — P08: Reconstruction error curve

**Prompt:**

> Tôi đã có `X_train_scaled`, `X_test_scaled` và ma trận eigenvectors `V_all` từ PCA fit trên train. Không được fit lại PCA cho từng k. Hãy đánh giá reconstruction error cho một tập các k đại diện như 1,2,3,5,10,20,30,50,75,100,150,200,300,400,500,561 và tự bổ sung các k đạt 80%,90%,95%,99% variance từ `k_by_threshold`. Với mỗi k, lấy `W_k = V_all[:, :k]`, project train/test bằng `X @ W_k`, reconstruct bằng `Z @ W_k.T`, tính MSE riêng cho train và test. Lưu kết quả vào DataFrame và vẽ reconstruction MSE theo k. Kiểm tra MSE hữu hạn và không tăng khi k tăng; ở k = toàn bộ số feature, train reconstruction MSE phải gần 0. Không dùng sklearn.

**Logic cần giữ nguyên:** reuse nested PCA basis → reconstruct train/test for multiple k → MSE curve → monotonic validation → full-basis sanity check.
