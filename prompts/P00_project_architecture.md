# AI PROMPTING LOG — P00: Project architecture

## Prompt

Tôi đang xây dựng case study t-SNE from scratch trên UCI Optdigits. Hãy khóa pipeline end-to-end: raw Optdigits → data integrity → squared Euclidean distance → conditional Gaussian similarities → perplexity matching bằng binary search → symmetric P → random 2D Y → Student-t Q → KL(P||Q) → analytic gradient → optimization → mathematical validation → synthetic validation → perplexity sensitivity → multi-scale Trustworthiness/Continuity → multi-seed stability → final full-data run → export. Không dùng sklearn.manifold.TSNE. Labels không tham gia fitting. Mỗi phase phải có assertions/checkpoint. Không chọn perplexity bằng KL thấp nhất và không hard-code winner trước experiment. Logic cần giữ nguyên: input → operation → validation → output.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
