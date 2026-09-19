# Package Validation Report — FINAL REPRODUCIBLE BUILD

Validation date: 2026-09-15

## Submission status

**READY FOR SUBMISSION AND CROSS-MACHINE REPRODUCTION**

## Static/package checks

- `pytest -q`: **4 passed**.
- `scripts/audit_package.py`: **PASS**.
- `scripts/audit_empirical_outputs.py`: **PASS**.
- Clean processed dataset: **5620 × 65** (64 explicit `Pixel_*` features + `label`), no `Unnamed` index column.
- Python source/scripts compile successfully.
- Main source notebook executes from a clean kernel in `quick` profile without error cells.
- CLI quick reproduction profile: **PASS**.

## P00–P08 core validation

All checkpoints **PASS**. Independent finite-difference gradient relative error ≈ `1.45e-09`; synthetic mean neighborhood overlap ≈ `0.476` vs random baseline ≈ `0.0559`, with post-hoc local purity `1.0`.

## P09 — Perplexity sensitivity

Executed on the same deterministic balanced 1,000-sample screening subset for perplexities `[5, 10, 20, 30, 40, 50]`. All six reference runs completed and the artifacts are bundled.

## P10 — Multi-scale neighborhood quality

Trustworthiness and Continuity evaluated at k = 5, 10, 20, 50. Stored metrics were independently recomputed from saved embeddings; maximum absolute discrepancy ≈ `1.11e-16`.

## P11 — Multi-seed stability

Candidates 20, 30, 40 were evaluated at seeds 0, 42, 123. Pairwise seed-neighborhood overlap was independently recomputed from saved embeddings.

- PRIMARY perplexity: **40**
- ALTERNATIVE perplexity: **30**

Selection wording: *selected primary configuration among evaluated candidates*; no claim of global optimality.

## P12 — Full exact run

- Samples: **5620**
- Features: **64**
- Perplexity: **40**
- Initial KL: **4.806500153**
- Final/restored KL: **1.081797712299418**
- Iterations: **1000**
- Runtime: **1808.58 s** (~30.14 min)
- Final embedding: **5620 × 2**
- Full-data T@5/C@5: **0.996845 / 0.992437**
- Full-data T@50/C@50: **0.986279 / 0.977478**
- Full-data worst neighborhood score: **0.977478**

The final Trustworthiness/Continuity metrics were independently recomputed from the saved full embedding. The final KL was independently recomputed from the saved full embedding and a freshly rebuilt full high-dimensional P. Absolute discrepancy vs saved final KL ≈ `6.66e-16`.

## Cross-machine hardening

Added and validated:
- pinned tested dependencies (`requirements-lock.txt`);
- notebook execution dependencies (`requirements-notebook.txt`);
- new-machine guide;
- environment verifier;
- full check runner;
- CLI `quick/screening/full` reproduction runner;
- clean-kernel notebook runner;
- rerun-output auditor;
- isolated `rerun_outputs/` destination;
- Windows `.bat` and Unix `.sh` convenience launchers;
- SHA-256 checksum verification;
- code/API reference and feature completeness matrix.

## Heavy reproduction note

The final full exact run is intentionally expensive. The canonical full reference run has already been completed and independently audited. During package-hardening, quick notebook and quick CLI execution were rerun from clean state; the full 30-minute empirical study was not redundantly rerun solely to create the ZIP.

## Final artifact audit

The final Word report is bundled under `report/` and matches the verified P09–P12 reference artifacts. The P12 CLI/notebook export schema has been synchronized with the bundled reference configuration/run-summary fields, including optimizer time and achieved-perplexity min/max values.


P09/P10 screening tables and embeddings, P11 per-seed outputs, P12 final embedding/configuration/history/full-data neighborhood-quality tables, final figures, reproducibility manifest, empirical audit, independent KL audit, docs, prompts, source, tests, runners and portability material are present and non-empty.
