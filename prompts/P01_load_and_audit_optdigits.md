# AI PROMPTING LOG — P01: Load and audit Optdigits

## Prompt

Tôi có raw optdigits.tra/optdigits.tes. Hãy load bằng pandas/NumPy, kiểm tra shape 3823×65 và 1797×65, gán 64 tên Pixel_1..Pixel_64 + label, concat thành 5620×65, tách X bằng explicit feature selection và y bằng label. Kiểm tra finite, range 0–16, class distribution, variance. Không scale, không PCA preprocessing. Khi lưu CSV bắt buộc index=False; reload và assert không có Unnamed column và X vẫn 5620×64. Chỉ khi PASS mới sang phase sau.

## Logic cần giữ nguyên

Input/state từ phase trước → nhiệm vụ toán học/experiment của phase → assertions/sanity checks → output có tên rõ ràng → chỉ chuyển phase khi checkpoint hợp lệ.
