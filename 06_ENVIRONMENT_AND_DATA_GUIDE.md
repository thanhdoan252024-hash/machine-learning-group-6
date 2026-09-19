# 06 — Environment and Data Guide

## Supported environment

Recommended:
- 64-bit Python 3.10–3.13;
- NumPy, Pandas, Matplotlib, pytest, nbformat;
- optional notebook runner dependencies in `requirements-notebook.txt`.

The package records the verified reference environment in `environment.json` and pins the tested library versions in `requirements-lock.txt`.

## Installation

Preferred exact reproduction:

```bash
python -m venv .venv
# activate the environment
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

If pinned wheels are unavailable for your platform/Python version:

```bash
pip install -r requirements.txt
```

For automated notebook execution:

```bash
pip install -r requirements-notebook.txt
```

## Data layout

Raw data:
- `data/raw/optdigits.tra`
- `data/raw/optdigits.tes`
- `data/raw/optdigits.names`

Audited processed data:
- `data/processed/optdigits_clean.csv`

Audit evidence:
- `data/audit/class_distribution.csv`
- `data/audit/feature_variance.csv`
- `data/audit/data_integrity_summary.json`

## Data contract

`optdigits_clean.csv` must have:
- 5,620 rows;
- exactly 64 columns named `Pixel_1` … `Pixel_64`;
- one final `label` column;
- no accidental `Unnamed: 0` index column;
- finite features in the canonical 0–16 range;
- labels 0–9.

The package intentionally rejects the legacy processed CSV that exported a row index and accidentally made it a 65th feature.

## Resource requirements

For `N=5620`:
- `N² = 31,584,400` elements;
- one float64 `N×N` matrix ≈ 241 MiB;
- the optimizer can require several such matrices transiently.

Recommended for `full` reproduction:
- at least 4 GiB free RAM;
- 8 GiB+ preferred;
- sufficient uninterrupted CPU time.

A GPU is not required by this NumPy implementation.
