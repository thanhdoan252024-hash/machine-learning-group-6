# AI PROMPTING LOG — P12: Final full-data run

## Prompt

Khóa PRIMARY_PERPLEXITY từ P11 và chạy exact t-SNE trên full 5620×64. Trước chạy in memory preflight; build D→P rồi giải phóng D/P_cond trước optimization. Labels chỉ post-hoc. Xuất final embedding CSV, figures, config, run summary, optimization history, reproducibility manifest. Không tune lại.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
