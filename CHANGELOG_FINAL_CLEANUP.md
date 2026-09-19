# Final Submission Cleanup

This cleanup addresses the final review feedback without changing Phase-3 predictive metrics.

1. Added an explicit allowed/reference-only library table to root README and Word report.
2. Documented Official LightGBM reference budget: `n_estimators=2000`, early stopping 50; actual best iterations are 151 (Regression) and 230 (Classification), both below scratch caps 240/450. Clarified that runtime is not an iso-compute benchmark.
3. Distinguished native missing-bin capability from the final Regression protocol, which uses train-only numeric-mean / categorical-mode imputation.
4. Documented Regression hyperparameter screening on fixed 5,000 train + 2,000 validation samples, followed by full 70,000 train / 15,000 validation retraining and 15,000 test reporting.
5. Synchronized the Word report with README/PHASE3 results and removed the placeholder `CHƯƠNG X` / `X.*` numbering.
6. Moved the historical 80/20 classification executable pipeline, notebook and outputs to `legacy/phase1_classification/`; final code now imports a protocol-neutral loader from `classification/data.py`.
7. Full automated suite after restructuring: **173 passed + 2 subtests, 0 failed**.
8. Final Word render QA: **20 pages**, visually inspected, no clipping/overlap.
