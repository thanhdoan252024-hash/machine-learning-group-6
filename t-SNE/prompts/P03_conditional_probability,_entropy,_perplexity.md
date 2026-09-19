# AI PROMPTING LOG — P03: Conditional probability, entropy, perplexity

## Prompt

Từ một distance row và beta>0, tự viết Gaussian conditional probability p(j|i), đặt self probability=0, dùng numerically-stable normalization. Tính Shannon entropy bằng natural log và perplexity=exp(H). Kiểm tra sum=1, range, beta tăng → entropy/perplexity không tăng. Chưa binary-search beta.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
