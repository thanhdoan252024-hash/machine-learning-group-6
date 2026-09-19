# t-SNE From Scratch on UCI Optdigits — Final Reproducible Case Study

This package contains the completed exact t-SNE case study, verified empirical outputs, AI Prompting Log, source code, tests, notebooks, audits, and cross-machine reproduction tooling.

## Verified final status

- P00–P13: PASS
- Dataset: `5620 × 64`
- Core: exact NumPy t-SNE from scratch
- `sklearn.manifold.TSNE`: not used
- P09: six perplexity experiments complete
- P10: multi-scale Trustworthiness/Continuity complete
- P11: multi-seed robustness complete
- PRIMARY: `40`
- ALTERNATIVE: `30`
- P12 full exact run: complete
- Final embedding: `5620 × 2`
- Final/restored KL: `1.081797712299418`
- Reference full-run time: `1808.58 s` (~30.14 min)

## Start here

1. `README_RUN_ON_NEW_MACHINE.md` — exact cross-machine instructions.
2. `17_FINAL_CASE_STUDY_AUDIT.md` — completeness/audit conclusion.
3. `FINAL_RESULTS_SUMMARY.md` — verified results.
4. `FINAL_TSNE_REPORT.md` — compact final report.
5. `notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb` — complete methodology notebook.
6. `notebook/TSNE_From_Scratch_Optdigits_FINAL_RESULTS.ipynb` — compact verified-results notebook.

## Fast validation

From the package root:

```bash
python scripts/verify_environment.py
python scripts/run_all_checks.py
python scripts/verify_checksums.py
```

## Reproduce on another machine

```bash
python scripts/run_case_study.py --profile quick --output-root rerun_outputs
python scripts/run_case_study.py --profile screening --output-root rerun_outputs
# complete expensive reproduction
python scripts/run_case_study.py --profile full --output-root rerun_outputs
python scripts/audit_rerun_outputs.py --output-root rerun_outputs
```

Verified reference outputs stay under `outputs/`; new reproduction outputs go to `rerun_outputs/`.

## Scientific interpretation

t-SNE is used here for local-neighborhood visualization. Apparent global 2D distances, cluster areas and cluster sizes are not treated as faithful metric properties of the original 64-dimensional space. PRIMARY p=40 is the selected configuration among the evaluated grid under the declared protocol, not a universal/global optimum.
