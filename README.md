# Machine Learning Group 6

## 📌 Giới thiệu

Đây là repository của **Group 6** cho project Machine Learning.

Mục tiêu của project là tìm hiểu và triển khai các thuật toán Machine Learning thông qua việc nghiên cứu nguyên lý hoạt động, xây dựng chương trình, thực nghiệm trên dataset và đánh giá kết quả.

Nhóm triển khai **4 thuật toán/phương pháp**:

1. **Linear Regression** – thuật toán hồi quy tuyến tính.
2. **LightGBM** – thuật toán Gradient Boosting cho bài toán phân loại.
3. **PCA (Principal Component Analysis)** – phương pháp giảm số chiều dữ liệu.
4. **t-SNE (t-distributed Stochastic Neighbor Embedding)** – phương pháp giảm chiều và trực quan hóa dữ liệu.

---

# 📚 Nội dung project

Mỗi thuật toán được nhóm triển khai theo quy trình:

**Tìm hiểu lý thuyết → Phân tích cơ chế hoạt động → Triển khai → Thực nghiệm → Đánh giá kết quả**

Nhóm tập trung vào việc hiểu cách các thuật toán hoạt động và triển khai các thành phần cần thiết cho quá trình thực nghiệm.

---

## 1. Linear Regression

**Linear Regression** là thuật toán hồi quy được sử dụng để mô hình hóa mối quan hệ giữa các biến đầu vào và biến mục tiêu.

Nội dung triển khai:

* Tìm hiểu nguyên lý Linear Regression.
* Xây dựng mô hình hồi quy.
* Huấn luyện mô hình trên dataset.
* Dự đoán giá trị đầu ra.
* Đánh giá kết quả dự đoán.

---

## 2. LightGBM

**LightGBM** là thuật toán Gradient Boosting dựa trên Decision Tree, được thiết kế nhằm tăng tốc quá trình huấn luyện và giảm mức sử dụng bộ nhớ.

Nội dung triển khai:

* Gradient Boosting Decision Trees
* Histogram-based Learning
* Leaf-wise Tree Growth
* Gradient và Hessian
* Tìm kiếm điểm chia
* Huấn luyện và dự đoán
* Đánh giá mô hình

Dataset được sử dụng để thực nghiệm là **Diabetes Binary Health Indicators**.

---

## 3. PCA

**PCA (Principal Component Analysis)** là phương pháp giảm số chiều dữ liệu bằng cách tìm các thành phần chính có khả năng giữ lại phần lớn thông tin của dữ liệu.

Nội dung triển khai:

* Tiền xử lý dữ liệu.
* Tính toán các thành phần chính.
* Chuyển đổi dữ liệu sang không gian có số chiều thấp hơn.
* Thực nghiệm với số lượng components khác nhau.
* Trực quan hóa và phân tích kết quả.

---

## 4. t-SNE

**t-SNE (t-distributed Stochastic Neighbor Embedding)** là phương pháp giảm chiều dữ liệu, thường được sử dụng để trực quan hóa dữ liệu nhiều chiều trong không gian 2D hoặc 3D.

Nội dung triển khai:

* Pairwise Distance
* Similarity Probability
* Perplexity
* Student-t Distribution
* KL Divergence
* Gradient Descent
* Trustworthiness
* Runtime

Dataset được sử dụng là **Optical Digits** với 64 đặc trưng pixel.

Nhóm thực nghiệm với nhiều giá trị **Perplexity** để phân tích ảnh hưởng của tham số đến chất lượng embedding và thời gian chạy.

---

# 📁 Cấu trúc repository

```text
machine-learning-group-6/
│
├── core/
│   └── Các module triển khai thuật toán
│
├── data/
│   └── Dataset và dữ liệu xử lý
│
├── outputs/
│   └── Kết quả thực nghiệm và hình ảnh
│
├── tests/
│   └── Các test kiểm tra chương trình
│
└── README.md
```

### `core/`

Chứa các module code chính dùng để triển khai các thuật toán.

### `data/`

Chứa dataset và dữ liệu được sử dụng trong quá trình thực nghiệm.

### `outputs/`

Chứa kết quả thực nghiệm, biểu đồ và các file output được tạo ra trong quá trình chạy chương trình.

### `tests/`

Chứa các chương trình kiểm tra nhằm đảm bảo những thành phần chính của project hoạt động đúng.

---

# 🔬 Quy trình triển khai

Các thuật toán trong project được thực hiện theo quy trình chung:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Algorithm Implementation
   ↓
Training / Transformation
   ↓
Experiment
   ↓
Evaluation
   ↓
Visualization & Analysis
```

Tùy theo từng thuật toán, các bước thực nghiệm và phương pháp đánh giá sẽ được điều chỉnh phù hợp.

---

# 🎯 Mục tiêu của project

Thông qua project, nhóm hướng tới việc:

* Hiểu nguyên lý hoạt động của các thuật toán Machine Learning.
* Hiểu các thành phần quan trọng bên trong thuật toán.
* Thực hành triển khai thuật toán bằng Python.
* Làm quen với quy trình thực nghiệm Machine Learning.
* Đánh giá và phân tích kết quả thực nghiệm.

---

# 👥 Thành viên

| Thành viên             | Vai trò |
| ---------------------- | ------- |
| Thành Đoàn             | Leader  |
| Hà Thị Trâm Anh        | Member  |
| Nguyễn Thị Vân Anh     | Member  |
| Trần Trung Châu        | Member  |
| Dương Ngọc Khánh Giang | Member  |
| Trần Nghĩa Nhân        | Member  |
