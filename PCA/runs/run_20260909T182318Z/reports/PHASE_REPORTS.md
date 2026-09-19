# Phase reports

Run UTC: 2026-09-09T18:23:18.903717+00:00

Inputs: official UCI HAR split; source hashes in dataset_manifest.json.

Scaler and PCA fit train only. Labels excluded from fit. No sklearn PCA/scaler.

## Phase 1: Dataset preparation

Prompt: P01. Cells: [5, 6, 7].

Output and actual metrics: [execution log](../evidence/PHASE_01.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP1: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 2: Standardization

Prompt: P02. Cells: [10, 11].

Output and actual metrics: [execution log](../evidence/PHASE_02.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP2: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 3: PCA from scratch

Prompt: P03. Cells: [14].

Output and actual metrics: [execution log](../evidence/PHASE_03.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

PCA fit completed; mathematical acceptance is evaluated in Phase 4 / CP3.

## Phase 4: Mathematical and API validation

Prompt: P04/P05. Cells: [17, 19].

Output and actual metrics: [execution log](../evidence/PHASE_04.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP3: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 5: Explained variance

Prompt: P06. Cells: [23, 24, 25, 26].

Output and actual metrics: [execution log](../evidence/PHASE_05.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP4: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 6: Visualization

Prompt: P07. Cells: [29, 30, 31].

Output and actual metrics: [execution log](../evidence/PHASE_06.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP5: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 7: Reconstruction

Prompt: P08. Cells: [34, 35, 36].

Output and actual metrics: [execution log](../evidence/PHASE_07.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP6: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 8: Loadings

Prompt: P09. Cells: [39, 40].

Output and actual metrics: [execution log](../evidence/PHASE_08.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP7: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 9: Candidate selection

Prompt: P10. Cells: [44].

Output and actual metrics: [execution log](../evidence/PHASE_09.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP8: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 10: Final datasets

Prompt: P11. Cells: [47, 48, 49].

Output and actual metrics: [execution log](../evidence/PHASE_10.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP9: **PASS**; numeric evidence in `evidence/checkpoints.json`.

## Phase 11: Export and reload validation

Prompt: P12. Cells: [52].

Output and actual metrics: [execution log](../evidence/PHASE_11.txt).

Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.

CP10: **PASS**; numeric evidence in `evidence/checkpoints.json`.
