## AI PROMPTING LOG — P12: Export reproducible PCA artifacts

**Prompt:**

> Tôi đã có pipeline PCA from scratch hoàn chỉnh với `scaler.mean_`, `scaler.scale_`, `pca_full.eigenvalues_all_`, `V_all`, `cumulative_variance_all`, `k90`, `k95`, các ma trận PCA90/PCA95 và labels. Hãy tạo thư mục output trong Colab, lưu arrays thành `.npy`, lưu toàn bộ preprocessing/PCA parameters thành một file `.npz`, và lưu các bảng `pca_candidate_df`, `final_summary_df`, `reconstruction_df` thành CSV. Không dùng sklearn hoặc joblib. Thêm `README.txt` giải thích file nào dùng cho classification, nhấn mạnh PCA chỉ fit trên train và test chỉ được transform. Sau khi lưu, kiểm tra từng file tồn tại và in đường dẫn.

**Logic cần giữ nguyên:** export transformed datasets + labels + train-fitted preprocessing/PCA parameters + evaluation tables + reproducibility note.
