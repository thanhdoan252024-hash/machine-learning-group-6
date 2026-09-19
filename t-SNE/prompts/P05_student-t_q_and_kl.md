# AI PROMPTING LOG — P05: Student-t Q and KL

## Prompt

Khởi tạo Y 2D reproducible bằng NumPy; xây Student-t numerator=(1+||yi-yj||²)^−1, diagonal 0, normalize thành Q. Tự viết KL(P||Q), chỉ tính nơi P>0 và lỗi nếu Q<=0 ở đó. Kiểm tra Q symmetry/sum, KL(P||P)=0, translation invariance. Chưa gradient/update.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
