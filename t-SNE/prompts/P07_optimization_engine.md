# AI PROMPTING LOG — P07: Optimization engine

## Prompt

Dùng gradient đã verify để xây optimize_tsne_embedding và class TSNEFromScratch với learning rate, early exaggeration, momentum, adaptive gains, recentering, KL history, best_Y restoration, early stopping. True KL luôn dùng P gốc; P_work exaggeration không renormalize. Best state phải lưu Y_current tương ứng exact KL, không suy ngược Y-update. Không cung cấp transform(X_new). Functional test trên subset, không gọi perplexity=30 là tối ưu.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
