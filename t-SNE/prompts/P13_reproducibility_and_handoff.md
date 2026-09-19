# AI PROMPTING LOG — P13: Reproducibility and handoff

## Prompt

Không thay đổi thuật toán. Audit required state/artifacts, tạo phase status, AI prompting log summary, README, requirements, structure, final results summary, expected-artifact contract và ZIP. Phân biệt artifact đã tạo với artifact chỉ sinh sau heavy run. Restart/Run-all phải không phụ thuộc hidden state. Không mang lại legacy index leakage hay hard-coded perplexity mismatch.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
