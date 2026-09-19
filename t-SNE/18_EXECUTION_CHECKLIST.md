# 18 — Execution Checklist

## Package/reference validation

- [ ] Use a fresh 64-bit Python environment.
- [ ] Install `requirements-lock.txt` (or fallback `requirements.txt`).
- [ ] Run `python scripts/verify_environment.py`.
- [ ] Run `python scripts/run_all_checks.py`.
- [ ] Run `python scripts/verify_checksums.py` after file transfer.

## Core correctness

- [ ] P01 clean-data assertions pass.
- [ ] P02–P06 mathematical checks pass.
- [ ] P07 optimizer restored-KL check passes.
- [ ] P08 synthetic validation passes.

## Empirical reproduction

- [ ] P09 all six perplexity runs complete.
- [ ] P10 T/C table + shortlist saved.
- [ ] P11 all candidate×seed runs complete.
- [ ] P12 full 5,620-sample embedding saved.
- [ ] Artifact filenames derive from the selected configuration.
- [ ] Rerun outputs are written to `rerun_outputs/`, not over bundled reference outputs.
- [ ] Run `python scripts/audit_rerun_outputs.py --output-root rerun_outputs`.

## Final handoff

- [ ] No `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`, or temporary files in the final ZIP.
- [ ] ZIP integrity test passes.
- [ ] README, Prompt Log, Prompt-Code Mapping, source, tests, data, final reports and all P09–P12 evidence are present.
