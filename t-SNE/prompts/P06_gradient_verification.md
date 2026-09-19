# AI PROMPTING LOG — P06: Gradient verification

## Prompt

Tự viết analytic t-SNE gradient 4 sum_j (Pij-Qij)(yi-yj)/(1+||yi-yj||²), vectorized. Viết brute-force reference và central finite-difference KL numerical gradient trên tiny dataset được xây lại đúng P. So sánh analytic vs brute-force và numerical; kiểm tra total gradient ≈0. Chưa optimizer.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
