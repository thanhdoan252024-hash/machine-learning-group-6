# 17 — Final Case Study Audit

## Audit conclusion

**READY FOR SUBMISSION AND CROSS-MACHINE REPRODUCTION** for the defined exact t-SNE case-study scope.

## Scope verified

The case study implements exact t-SNE from scratch using NumPy on UCI Optdigits, validates the mathematics independently, evaluates neighborhood preservation, checks stochastic stability, selects a primary configuration using a pre-declared protocol, and runs the selected configuration on all 5,620 samples.

## Documentation audit

Present and populated:
- start/readme documents;
- research question and scope;
- t-SNE theory;
- implementation roadmap;
- stage gates and QA;
- AI Prompting Log P00–P13;
- prompt-code mapping;
- environment/data guide;
- reproducibility/handoff guide;
- evaluation/selection protocol;
- references;
- results/evidence guide;
- legacy-project audit;
- code/API reference;
- reproduction profiles;
- feature completeness matrix;
- execution checklist;
- expected-artifact contract;
- final results summary;
- final case-study report;
- package validation report.

## Prompt audit

There are 14 phase prompt files (`P00`–`P13`) plus `05_AI_PROMPTING_LOG_MASTER.md`. Every phase appears in the master log and in the prompt-code mapping. The prompts preserve the project constraints: no sklearn t-SNE, no label leakage into fitting, no cross-perplexity KL leaderboard, structural rather than coordinate seed stability, and no global-optimum overclaim.

## Code audit

The package contains:
- reusable `src/` implementation;
- evaluation utilities;
- unit/data tests;
- quick smoke test;
- static package audit;
- strict empirical reference audit;
- rerun-output audit;
- cross-platform command-line reproduction runner;
- clean-kernel notebook runner.

## Empirical audit

Bundled reference outputs pass independent recomputation:
- P09 artifacts present and finite;
- P10 Trustworthiness/Continuity recomputed from stored embeddings;
- P11 per-seed quality and kNN-overlap stability recomputed;
- selected Primary = 40 and Alternative = 30;
- P12 final embedding is `5620 × 2` and finite;
- P12 full-data Trustworthiness/Continuity is independently recomputed from the saved final embedding;
- final/restored KL is consistent with the independently reconstructed full high-dimensional P distribution.

## Verified final reference values

- Samples: `5620`
- Features: `64`
- Primary perplexity: `40`
- Alternative perplexity: `30`
- Final seed: `42`
- Initial KL: `4.806500153`
- Final/restored KL: `1.081797712299418`
- Iterations: `1000`
- Reference runtime: `1808.58 s` (~30.14 min)
- Full T@5/C@5: `0.996845 / 0.992437`
- Full T@50/C@50: `0.986279 / 0.977478`
- Full worst neighborhood score: `0.977478`

## Portability hardening added

To minimize cross-machine errors the final reproducible package adds:
- `requirements-lock.txt` with tested versions;
- `requirements-notebook.txt` for automated notebook execution;
- `README_RUN_ON_NEW_MACHINE.md`;
- `scripts/verify_environment.py`;
- `scripts/run_all_checks.py`;
- `scripts/run_case_study.py` with `quick/screening/full` profiles;
- `scripts/run_notebook.py` using a clean kernel;
- `scripts/audit_rerun_outputs.py`;
- separate `rerun_outputs/` destination so verified reference outputs are not overwritten;
- Windows `.bat` and Unix `.sh` convenience launchers;
- SHA-256 checksum manifest.

## Limitations explicitly retained

- Exact t-SNE has `O(N^2)` memory/time cost.
- The implementation is intentionally educational/auditable rather than production-scale.
- Standard non-parametric t-SNE has no simple learned transform for unseen samples here.
- Embeddings remain stochastic; multi-seed robustness reduces but does not eliminate that fact.
- Global 2D cluster distances/areas are not interpreted literally.
- Primary p=40 is the selected configuration among the evaluated grid, not a claimed global optimum.

## Final gate

The package is considered valid when all of the following pass on the target machine:

```bash
python scripts/verify_environment.py
python scripts/run_all_checks.py
python scripts/run_case_study.py --profile quick --output-root rerun_outputs
```

For full scientific reproduction additionally run:

```bash
python scripts/run_case_study.py --profile full --output-root rerun_outputs
python scripts/audit_rerun_outputs.py --output-root rerun_outputs
```
