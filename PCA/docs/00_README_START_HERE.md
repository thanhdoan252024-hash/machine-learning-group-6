# UCI HAR — PCA From Scratch Project Handoff Package

## Mục tiêu
Xây dựng và đánh giá PCA từ đầu bằng NumPy trên UCI HAR, không dùng `sklearn.StandardScaler` hoặc `sklearn.PCA`.

## Trạng thái
**Phần PCA đã hoàn thành sau Phase 11.**  
Từ bước tiếp theo chỉ còn downstream classification.

## Quy tắc cốt lõi
- Fit scaler chỉ trên train.
- Fit PCA chỉ trên train.
- Test chỉ transform.
- Label không tham gia PCA fit.
- Giữ nguyên split train/test gốc.
- PCA90 và PCA95 dùng chung eigenbasis đã học từ train.

## Thứ tự đọc
1. `01_RESEARCH_IDEA_AND_SCOPE.md`
2. `02_PCA_THEORY_KNOWLEDGE.md`
3. `03_IMPLEMENTATION_ROADMAP.md`
4. `04_STAGE_GATES_AND_QA.md`
5. `05_AI_PROMPTING_LOG_MASTER.md`
6. `06_ENVIRONMENT_AND_DATA_GUIDE.md`
7. `07_MULTI_AGENT_WORK_BREAKDOWN.md`
8. `08_REPRODUCIBILITY_AND_HANDOFF.md`
9. `09_CLASSIFICATION_HANDOFF.md`
10. `10_REFERENCES.md`
11. `11_RUN_ORDER_QUICK_GUIDE.md`

Notebook:
[Notebook nguồn](../notebook/PCA_From_Scratch_UCI_HAR_Phase1_11_With_AI_Prompting_Log.ipynb)

Prompt tách riêng:
`prompts/P00...` đến `prompts/P12...`


## Tài liệu bổ sung để handoff chặt chẽ hơn

12. `12_RESULTS_AND_EVIDENCE_GUIDE.md`
13. `18_EXECUTION_CHECKLIST.md`
14. `19_EXPECTED_ARTIFACTS_CONTRACT.md`

Thư mục `templates/` chứa:

- `PHASE_REPORT_TEMPLATE.md`
- `RESULTS_TEMPLATE.md`
- `AGENT_HANDOFF_TEMPLATE.md`
- `ISSUE_LOG_TEMPLATE.md`
- `DECISION_LOG_TEMPLATE.md`
- `ENVIRONMENT_RECORD_TEMPLATE.md`
- `FINAL_PCA_REPORT_TEMPLATE.md`

Chạy `python scripts/run_pca.py` từ thư mục gốc project để tự tạo `runs/run_<UTC>/` với notebook có output, báo cáo và các artifacts. Xem [hướng dẫn ở README](../README.md) và [các lần chạy](../runs/README.md).
