# AI PROMPTING LOG — P09: Perplexity sensitivity

## Prompt

Tạo deterministic stratified Optdigits screening subset; giữ dataset/optimizer/seed cố định, chỉ thay perplexity [5,10,20,30,40,50]. Lưu embedding, runtime, iteration, KL diagnostics và achieved perplexity. Không rank bằng KL và chưa chọn best. Tạo visual comparison chỉ post-hoc.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
