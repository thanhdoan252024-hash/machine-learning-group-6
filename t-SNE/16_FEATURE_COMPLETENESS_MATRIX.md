# 16 — Feature Completeness Matrix

| Requirement | Implementation / Evidence | Status |
|---|---|---|
| Clean Optdigits data | `data/processed/optdigits_clean.csv` + audit files | PASS |
| No accidental index feature | dataset tests + package audit | PASS |
| Squared Euclidean from scratch | `src/tsne_from_scratch.py` | PASS |
| Conditional Gaussian similarities | `entropy_and_probabilities` | PASS |
| Per-point perplexity matching | `binary_search_beta` | PASS |
| Symmetric joint P | `symmetrize_probabilities` | PASS |
| Student-t Q | `low_dimensional_affinities` | PASS |
| KL(P||Q) | `kl_divergence` | PASS |
| Analytic gradient | `tsne_gradient` | PASS |
| Independent finite-difference gradient verification | main notebook + validation report | PASS |
| Early exaggeration | optimizer | PASS |
| Momentum | optimizer | PASS |
| Adaptive gains | optimizer | PASS |
| Re-centering | optimizer | PASS |
| Best-state restoration | optimizer | PASS |
| Early stopping | optimizer | PASS |
| Synthetic structural validation | P08 | PASS |
| Perplexity sensitivity | P09 outputs | PASS |
| Trustworthiness | P10 + independent recomputation | PASS |
| Continuity | P10 + independent recomputation | PASS |
| Pareto shortlist | P10 | PASS |
| Multi-seed testing | P11 | PASS |
| Structural seed stability | kNN overlap | PASS |
| Primary/alternative selection | p40 / p30 | PASS |
| Full exact 5,620-sample run | P12 | PASS |
| Final embedding export | `outputs/embeddings/final_optdigits_embedding_p40.csv` | PASS |
| Convergence evidence | final history + figures | PASS |
| Independent final-KL audit | `outputs/p12_independent_kl_audit.json` | PASS |
| AI Prompting Log P00–P13 | master log + 14 prompt files | PASS |
| Prompt ↔ code mapping | `02_PROMPT_CODE_MAPPING.md` | PASS |
| Unit/data tests | `tests/` | PASS |
| Static package audit | `scripts/audit_package.py` | PASS |
| Empirical artifact audit | `scripts/audit_empirical_outputs.py` | PASS |
| Cross-machine CLI runner | `scripts/run_case_study.py` | PASS |
| Cross-machine notebook runner | `scripts/run_notebook.py` | PASS |
| Pinned tested dependencies | `requirements-lock.txt` | PASS |
| Checksum manifest | `CHECKSUMS.sha256` | PASS |

The package is complete for the stated educational/research scope. It does not implement Barnes–Hut/FFT approximations or a learned out-of-sample mapping; these are explicitly out of scope rather than missing required features.
