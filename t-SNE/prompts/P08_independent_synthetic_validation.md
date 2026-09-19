# AI PROMPTING LOG — P08: Independent synthetic validation

## Prompt

Tạo 3 Gaussian clusters high-dimensional bằng NumPy, fit TSNEFromScratch chỉ với X, labels chỉ post-hoc. Tự xây kNN overlap và local label purity. So sánh observed overlap với random baseline; không dùng scatter đẹp làm bằng chứng duy nhất, không diễn giải global inter-cluster distance.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
