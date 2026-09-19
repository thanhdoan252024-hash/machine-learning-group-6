# PCA reproduction run

UTC start: 2026-09-10T08:12:00.658952+00:00

- [Notebook with outputs](notebook.executed.ipynb)
- [Browser-readable notebook](notebook.html)
- [Results](reports/RESULTS.md)
- [Final report](reports/FINAL_PCA_REPORT.md)
- [Phase evidence](reports/PHASE_REPORTS.md)
- [Exported arrays and tables](outputs/README.txt)
- Figures: `figures/` (13 PNGs).
- Integrity: `artifact_manifest.json` (SHA-256 and byte sizes).

## Reproduce from the project root

```powershell
python -m pip install -r requirements-repro.txt
python scripts/run_pca.py --dataset-dir "data/raw/UCI HAR Dataset"
```

Each execution creates a new run directory. The source snapshot in this bundle identifies the executed code.
To rerun from this bundle itself, install requirements-lock.txt and supply the absolute dataset directory to scripts/run_pca.py.

## Verify without fitting PCA again

```powershell
python scripts/verify_pca_artifacts.py --run-dir "runs/run_20260910T081200Z" --dataset-dir "data/raw/UCI HAR Dataset"
```

The dataset hashes identify the six raw inputs. Downloaded raw data is kept outside this run bundle.
verification.json is excluded from the checksum manifest because it records verification of that manifest.
