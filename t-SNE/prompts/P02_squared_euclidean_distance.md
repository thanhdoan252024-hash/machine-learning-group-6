# AI PROMPTING LOG — P02: Squared Euclidean distance

## Prompt

Từ X sạch 5620×64, tự viết squared_euclidean_distances(X) bằng ||xi||²+||xj||²−2xi·xj, NumPy vectorized, float64. Kiểm tra symmetry, diagonal 0, nonnegative, finite, và so với brute-force trên tiny subset. Không sqrt. Chưa xây P/Q/t-SNE. In memory estimate cho N×N.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
