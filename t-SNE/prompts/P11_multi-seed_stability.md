# AI PROMPTING LOG — P11: Multi-seed stability

## Prompt

Với 2–3 candidate perplexities từ P10, build P một lần mỗi perplexity rồi optimize seeds [0,42,123]. Tính T/C mỗi seed và pairwise kNN overlap giữa seed embeddings ở nhiều k. Không dùng coordinate MSE. Aggregate mean/std/worst; chọn PRIMARY + ALTERNATIVE bằng robust neighborhood + stability, runtime chỉ tie-break. Không gọi global optimum.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
