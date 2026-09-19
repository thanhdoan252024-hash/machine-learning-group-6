# 05 — AI PROMPTING LOG MASTER

# AI PROMPTING LOG — P00: Project architecture

## Prompt

Tôi đang xây dựng case study t-SNE from scratch trên UCI Optdigits. Hãy khóa pipeline end-to-end: raw Optdigits → data integrity → squared Euclidean distance → conditional Gaussian similarities → perplexity matching bằng binary search → symmetric P → random 2D Y → Student-t Q → KL(P||Q) → analytic gradient → optimization → mathematical validation → synthetic validation → perplexity sensitivity → multi-scale Trustworthiness/Continuity → multi-seed stability → final full-data run → export. Không dùng sklearn.manifold.TSNE. Labels không tham gia fitting. Mỗi phase phải có assertions/checkpoint. Không chọn perplexity bằng KL thấp nhất và không hard-code winner trước experiment. Logic cần giữ nguyên: input → operation → validation → output.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P01: Load and audit Optdigits

## Prompt

Tôi có raw optdigits.tra/optdigits.tes. Hãy load bằng pandas/NumPy, kiểm tra shape 3823×65 và 1797×65, gán 64 tên Pixel_1..Pixel_64 + label, concat thành 5620×65, tách X bằng explicit feature selection và y bằng label. Kiểm tra finite, range 0–16, class distribution, variance. Không scale, không PCA preprocessing. Khi lưu CSV bắt buộc index=False; reload và assert không có Unnamed column và X vẫn 5620×64. Chỉ khi PASS mới sang phase sau.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P02: Squared Euclidean distance

## Prompt

Từ X sạch 5620×64, tự viết squared_euclidean_distances(X) bằng ||xi||²+||xj||²−2xi·xj, NumPy vectorized, float64. Kiểm tra symmetry, diagonal 0, nonnegative, finite, và so với brute-force trên tiny subset. Không sqrt. Chưa xây P/Q/t-SNE. In memory estimate cho N×N.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P03: Conditional probability, entropy, perplexity

## Prompt

Từ một distance row và beta>0, tự viết Gaussian conditional probability p(j|i), đặt self probability=0, dùng numerically-stable normalization. Tính Shannon entropy bằng natural log và perplexity=exp(H). Kiểm tra sum=1, range, beta tăng → entropy/perplexity không tăng. Chưa binary-search beta.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P04: Perplexity matching and P

## Prompt

Tự viết binary_search_beta để tìm beta_i sao cho entropy gần log(target perplexity), adaptive bounds không dùng scipy. Sau đó xây P_cond row-wise và P=(P_cond+P_cond.T)/(2N). Kiểm tra achieved perplexity, P_cond row sums, P symmetry, diagonal 0, sum(P)=1. Không dùng heuristic 3*perplexity<n như ràng buộc toán học.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P05: Student-t Q and KL

## Prompt

Khởi tạo Y 2D reproducible bằng NumPy; xây Student-t numerator=(1+||yi-yj||²)^−1, diagonal 0, normalize thành Q. Tự viết KL(P||Q), chỉ tính nơi P>0 và lỗi nếu Q<=0 ở đó. Kiểm tra Q symmetry/sum, KL(P||P)=0, translation invariance. Chưa gradient/update.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P06: Gradient verification

## Prompt

Tự viết analytic t-SNE gradient 4 sum_j (Pij-Qij)(yi-yj)/(1+||yi-yj||²), vectorized. Viết brute-force reference và central finite-difference KL numerical gradient trên tiny dataset được xây lại đúng P. So sánh analytic vs brute-force và numerical; kiểm tra total gradient ≈0. Chưa optimizer.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P07: Optimization engine

## Prompt

Dùng gradient đã verify để xây optimize_tsne_embedding và class TSNEFromScratch với learning rate, early exaggeration, momentum, adaptive gains, recentering, KL history, best_Y restoration, early stopping. True KL luôn dùng P gốc; P_work exaggeration không renormalize. Best state phải lưu Y_current tương ứng exact KL, không suy ngược Y-update. Không cung cấp transform(X_new). Functional test trên subset, không gọi perplexity=30 là tối ưu.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P08: Independent synthetic validation

## Prompt

Tạo 3 Gaussian clusters high-dimensional bằng NumPy, fit TSNEFromScratch chỉ với X, labels chỉ post-hoc. Tự xây kNN overlap và local label purity. So sánh observed overlap với random baseline; không dùng scatter đẹp làm bằng chứng duy nhất, không diễn giải global inter-cluster distance.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P09: Perplexity sensitivity

## Prompt

Tạo deterministic stratified Optdigits screening subset; giữ dataset/optimizer/seed cố định, chỉ thay perplexity [5,10,20,30,40,50]. Lưu embedding, runtime, iteration, KL diagnostics và achieved perplexity. Không rank bằng KL và chưa chọn best. Tạo visual comparison chỉ post-hoc.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P10: Trustworthiness and Continuity

## Prompt

Tự viết multi-scale Trustworthiness và Continuity ở k=[5,10,20,50] từ neighbor ranks, không dùng sklearn metric. Identity geometry phải cho T=C=1. Tạo quality table và Pareto shortlist; không weighted score tùy ý và chưa kết luận global optimum.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P11: Multi-seed stability

## Prompt

Với 2–3 candidate perplexities từ P10, build P một lần mỗi perplexity rồi optimize seeds [0,42,123]. Tính T/C mỗi seed và pairwise kNN overlap giữa seed embeddings ở nhiều k. Không dùng coordinate MSE. Aggregate mean/std/worst; chọn PRIMARY + ALTERNATIVE bằng robust neighborhood + stability, runtime chỉ tie-break. Không gọi global optimum.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P12: Final full-data run

## Prompt

Khóa PRIMARY_PERPLEXITY từ P11 và chạy exact t-SNE trên full 5620×64. Trước chạy in memory preflight; build D→P rồi giải phóng D/P_cond trước optimization. Labels chỉ post-hoc. Xuất final embedding CSV, figures, config, run summary, optimization history, reproducibility manifest. Không tune lại.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.


# AI PROMPTING LOG — P13: Reproducibility and handoff

## Prompt

Không thay đổi thuật toán. Audit required state/artifacts, tạo phase status, AI prompting log summary, README, requirements, structure, final results summary, expected-artifact contract và ZIP. Phân biệt artifact đã tạo với artifact chỉ sinh sau heavy run. Restart/Run-all phải không phụ thuộc hidden state. Không mang lại legacy index leakage hay hard-coded perplexity mismatch.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
