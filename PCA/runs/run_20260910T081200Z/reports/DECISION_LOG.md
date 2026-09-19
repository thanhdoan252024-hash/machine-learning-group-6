# Decisions

- Preserve original PCA mathematics and official split.
- Derive k from training cumulative variance; keep PCA95 primary and PCA90 alternative.
- Export all plots and evaluation tables, with numeric checkpoint evidence.
- Validate saved arrays by reloading, including projection with persisted parameters.
- Preserve each execution in a separate run directory.
