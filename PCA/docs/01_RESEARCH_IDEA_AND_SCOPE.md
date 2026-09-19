# Research Idea and Scope

## Câu hỏi nghiên cứu
PCA có thể giảm đáng kể số chiều của UCI HAR trong khi vẫn bảo toàn đủ cấu trúc thông tin để phục vụ phân loại hoạt động con người đến mức nào?

## Dataset
UCI Human Activity Recognition Using Smartphones:
- 30 volunteers
- 6 activities
- 10,299 samples
- 561 features
- accelerometer + gyroscope
- train/test split chính thức

## Phạm vi PCA
Phải chứng minh:
1. preprocessing đúng;
2. PCA implementation đúng;
3. explained variance hợp lý;
4. reconstruction loss giảm theo k;
5. visualization có thể diễn giải;
6. loadings hợp lệ;
7. dữ liệu PCA có thể bàn giao reproducibly.

## Candidate cuối
- PCA90: ưu tiên compression.
- PCA95: ưu tiên giữ thông tin.
- Không gọi cấu hình nào là tối ưu tuyệt đối trước classification.
