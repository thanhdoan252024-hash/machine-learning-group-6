# 08 — Reproducibility and Handoff

## Reproducibility principles

1. Raw data and audited clean data are included.
2. Random seeds are explicit.
3. Empirical outputs are never pre-filled with invented values.
4. The exact selection protocol is documented before final configuration selection.
5. Canonical verified outputs are preserved under `outputs/`.
6. Cross-machine reruns write to `rerun_outputs/` by default.
7. Tests, static package audits and empirical-output audits are separate gates.

## Verified evidence included

The package already contains the final reference outputs for P09–P12, including:
- six screening embeddings;
- multi-scale T/C tables;
- multi-seed embeddings and stability tables;
- selected primary/alternative settings;
- final 5,620-sample embedding;
- convergence history and figures;
- independent empirical audit artifacts.

## Cross-machine reproduction

Recommended order:

```bash
python scripts/verify_environment.py
python scripts/run_all_checks.py
python scripts/run_case_study.py --profile quick --output-root rerun_outputs
python scripts/run_case_study.py --profile screening --output-root rerun_outputs
# Optional complete expensive reproduction:
python scripts/run_case_study.py --profile full --output-root rerun_outputs
python scripts/audit_rerun_outputs.py --output-root rerun_outputs
```

## Notebook handoff

Use `scripts/run_notebook.py` if notebook execution is required. The main notebook reads `TSNE_EXECUTION_PROFILE` and `TSNE_OUTPUT_ROOT` environment variables, so automated runs do not require source-cell editing.

## Integrity

`CHECKSUMS.sha256` records package-file hashes. Use `scripts/verify_checksums.py` after transfer to detect corruption.
