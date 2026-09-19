# PCA reproduction run

UTC start: 2026-09-09T18:23:18.903717+00:00

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
python scripts/run_pca.py --dataset-dir "UCI HAR Dataset"
```

Each execution creates a new run directory. The source snapshot in this bundle identifies the executed code.

## Verify without fitting PCA again

```powershell
python scripts/verify_pca_artifacts.py --run-dir "runs/run_20260909T182318Z" --dataset-dir "UCI HAR Dataset"
```

The dataset hashes identify the six raw inputs. Downloaded raw data is kept outside this run bundle.
