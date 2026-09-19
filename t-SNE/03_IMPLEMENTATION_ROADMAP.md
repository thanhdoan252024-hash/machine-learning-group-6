# 03 — Implementation Roadmap

The project is organized as gated phases. Every phase follows:

**objective/theory → AI Prompting Log → implementation → diagnostics → assertions → checkpoint**.

| Phase | Purpose | Main evidence |
|---|---|---|
| P00 | Lock scientific architecture and constraints | pipeline + no-leakage/no-sklearn rules |
| P01 | Load/audit Optdigits | clean `5620×64` X, labels, variance, no accidental index |
| P02 | Pairwise squared distances | vectorized vs brute-force equivalence |
| P03 | Fixed-beta conditional probabilities | normalization, entropy/perplexity behavior |
| P04 | Perplexity matching and symmetric P | beta convergence, achieved perplexity, P invariants |
| P05 | Student-t Q and KL | symmetry, normalization, KL identity, translation invariance |
| P06 | Gradient verification | brute-force + finite-difference checks |
| P07 | Optimization engine/class | finite embedding, KL improvement, best-state restoration |
| P08 | Independent synthetic validation | neighborhood overlap vs random baseline |
| P09 | Perplexity sensitivity | six real screening embeddings |
| P10 | Quantitative neighborhood evaluation | T/C at multiple k, Pareto shortlist |
| P11 | Multi-seed robustness | 3 seeds × candidate settings + kNN overlap |
| P12 | Final full run | exact `5620×64 → 5620×2` output and convergence artifacts |
| P13 | Reproducibility/handoff | audits, manifests, docs, packaging |

## Reproduction profiles

- `quick`: correctness/core validation only.
- `screening`: P09–P10.
- `full`: P09–P12 and final outputs.

See `15_REPRODUCTION_PROFILES.md` and `README_RUN_ON_NEW_MACHINE.md` for exact commands.
