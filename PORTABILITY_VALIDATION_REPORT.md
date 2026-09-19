# Portability Validation Report

Date: 2026-09-15

## Verified during final package hardening

- Python source/scripts compile successfully.
- `pytest -q`: 4 passed.
- Static package audit: PASS.
- Independent empirical reference-output audit: PASS.
- Quick smoke test: PASS.
- Main P00–P13 notebook executed from a clean kernel with `quick` profile: PASS, no error cells.
- Cross-platform CLI runner `scripts/run_case_study.py --profile quick`: PASS.
- Dataset and final empirical artifacts are self-contained in the package.
- Main notebook accepts environment-variable profile/output overrides for automated execution.
- Rerun outputs are isolated from canonical verified outputs.

## Heavy reproduction note

The full exact reproduction is intentionally expensive and was not repeated solely for package-hardening, because the bundled P09–P12 reference run has already been completed and independently audited. The bundled final full run used 1000 exact iterations on 5,620 samples and took approximately 30.14 minutes on the recorded reference environment.

The heavy `screening/full` CLI paths use the same validated `src/` implementation and the same pre-declared experiment protocol. On a different machine, use `scripts/audit_rerun_outputs.py` after completion.

## Portability conclusion

The package is hardened for a clean-machine rerun. Hardware/BLAS differences may change runtime and can produce tiny floating-point differences, so reproducibility is validated by mathematical/artifact contracts rather than byte-identical output files.
