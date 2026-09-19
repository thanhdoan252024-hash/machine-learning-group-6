# AI PROMPTING LOG — P04: Perplexity matching and P

## Prompt

Tự viết binary_search_beta để tìm beta_i sao cho entropy gần log(target perplexity), adaptive bounds không dùng scipy. Sau đó xây P_cond row-wise và P=(P_cond+P_cond.T)/(2N). Kiểm tra achieved perplexity, P_cond row sums, P symmetry, diagonal 0, sum(P)=1. Không dùng heuristic 3*perplexity<n như ràng buộc toán học.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
