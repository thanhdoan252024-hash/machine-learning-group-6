# t-SNE From Scratch on UCI Optdigits — FINAL REPRODUCIBLE PACKAGE

This package contains the completed case study, verified reference outputs, and a hardened cross-machine reproduction workflow.

## Final verified status

- P00–P13: **PASS**
- Dataset: **5,620 × 64** clean features
- Exact NumPy t-SNE from scratch; no `sklearn.manifold.TSNE`
- P09: six perplexity runs completed
- P10: Trustworthiness/Continuity completed
- P11: multi-seed robustness completed
- PRIMARY: **p=40**
- ALTERNATIVE: **p=30**
- P12 full exact run: **5,620 samples completed**
- Final embedding: **5620 × 2**
- Final/restored KL: **1.081797712299418**
- Direct final-embedding neighborhood audit: **PASS** (T@5=0.996845, C@5=0.992437, T@50=0.986279, C@50=0.977478)
- Reference full-run time: **1808.58 s (~30.14 min)**
- Static, empirical, final-KL, and ZIP audits: **PASS**

## Read in this order

1. `report/Chuong_Case_Study_Thuc_nghiem_tSNE_UCI_Optdigits_FINAL.docx` — final Word report ready for submission.
2. `README_RUN_ON_NEW_MACHINE.md` — run/reproduce safely on a new computer.
3. `17_FINAL_CASE_STUDY_AUDIT.md` — complete audit of docs, prompts, code, experiments and portability.
4. `FINAL_RESULTS_SUMMARY.md` — verified empirical results.
5. `16_FEATURE_COMPLETENESS_MATRIX.md` — requirement-to-evidence matrix.
6. `05_AI_PROMPTING_LOG_MASTER.md` — P00–P13 prompt record.
7. `notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb` — full source notebook.

## First commands on another machine

```bash
python scripts/verify_environment.py
python scripts/run_all_checks.py
python scripts/verify_checksums.py
python scripts/run_case_study.py --profile quick --output-root rerun_outputs
```

Use `screening` or `full` only when you intentionally want to reproduce the empirical study. New outputs go to `rerun_outputs/`; bundled verified outputs remain under `outputs/`.
