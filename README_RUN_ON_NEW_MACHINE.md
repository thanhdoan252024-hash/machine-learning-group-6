# Run on a New Machine — Exact t-SNE From Scratch on UCI Optdigits

This is the recommended entry point when reproducing the case study on another computer.
The package contains both **verified reference outputs** and the code needed to reproduce them.

## 1. Requirements

Recommended:
- Python 3.10–3.13, 64-bit
- At least 4 GiB free RAM for `full`; 8 GiB+ is preferable
- Several GiB free disk space
- CPU execution is sufficient; the implementation is NumPy-only and does not require a GPU

The exact full-data run is intentionally expensive because it uses exact `O(N^2)` t-SNE on all 5,620 samples.
On the verified reference environment, the P12 optimization took about 30 minutes.
Runtime will vary by CPU/BLAS implementation.

## 2. Create a clean virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

If the exact pinned versions are not available for your Python version, use:

```bash
pip install -r requirements.txt
```

For notebook execution support also install:

```bash
pip install -r requirements-notebook.txt
```

## 3. Verify the package before running experiments

Run from the **package root**:

```bash
python scripts/verify_environment.py
python scripts/run_all_checks.py
```

Expected high-level result:

```text
pytest: PASS
package audit: PASS
empirical reference-output audit: PASS
```

The empirical audit verifies the bundled reference outputs; it does not rerun the 30-minute full optimization.

## 4. Recommended reproduction workflow

### A. Quick/core verification

This verifies the mathematical/core implementation without launching the heavy study:

```bash
python scripts/quick_smoke_test.py
python scripts/run_notebook.py --profile quick
```

### B. Screening study (P09–P10)

This reruns the six perplexity screening experiments and neighborhood evaluation into a separate folder:

```bash
python scripts/run_case_study.py --profile screening --output-root rerun_outputs
```

The canonical verified outputs in `outputs/` are not overwritten.

### C. Full reproduction (P09–P12)

This reruns the complete empirical study from scratch:

```bash
python scripts/run_case_study.py --profile full --output-root rerun_outputs
```

The `full` profile includes:
- P09: perplexity grid `{5,10,20,30,40,50}`
- P10: Trustworthiness/Continuity at `k={5,10,20,50}`
- P11: multi-seed robustness with seeds `{0,42,123}`
- P12: exact full-data `5620 × 64 → 5620 × 2` run

Then audit the rerun outputs:

```bash
python scripts/audit_rerun_outputs.py --output-root rerun_outputs
```

## 5. Notebook reproduction

The source notebook is:

```text
notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb
```

You can execute it without manually editing the first cell:

```bash
python scripts/run_notebook.py --profile quick
python scripts/run_notebook.py --profile screening
python scripts/run_notebook.py --profile full
```

By default, executed notebooks are written under `rerun_outputs/notebooks/` and do not overwrite the source notebook.

## 6. Verified reference outputs versus rerun outputs

- `outputs/` contains the **verified reference empirical outputs** used by the final report.
- `rerun_outputs/` is reserved for outputs generated on another machine.

Do not overwrite the verified reference outputs while checking portability.

## 7. Expected final scientific decisions

The verified reference study selected:

- PRIMARY perplexity: **40**
- ALTERNATIVE perplexity: **30**

A rerun on a different BLAS/CPU should be numerically very close, but tiny floating-point differences are possible.
The rerun audit therefore validates the scientific protocol and artifact contract rather than requiring byte-for-byte identity.

## 8. Important scientific constraints

Do not:
- replace the core implementation with `sklearn.manifold.TSNE`;
- use labels during t-SNE fitting;
- rank perplexities solely by cross-perplexity KL;
- use raw coordinate MSE as the multi-seed stability metric;
- interpret global 2D cluster distances/areas as faithful high-dimensional geometry;
- call the selected perplexity a global optimum.

## 9. If a run fails

1. Run `python scripts/verify_environment.py`.
2. Confirm you are executing from the package root.
3. Confirm `data/processed/optdigits_clean.csv` exists and is `5620 × 65`.
4. Confirm enough memory is available.
5. Run `pytest -q` and `python scripts/audit_package.py`.
6. Do not weaken assertions or fabricate missing empirical outputs.

See `FINAL_CASE_STUDY_AUDIT.md` for the complete audit and `FINAL_RESULTS_SUMMARY.md` for verified reference results.
