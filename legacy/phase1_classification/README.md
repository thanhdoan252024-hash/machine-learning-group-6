# Legacy Phase-1 Classification Baseline

This directory is **historical reference only**. It preserves the original
classification protocol used before the final case study was redesigned.

- Split: stratified 80/20 train/test
- Estimators: 100
- Decision threshold: 0.5
- No validation split for model/threshold selection
- Historical test accuracy: 0.9875

**Do not use these metrics for grading the final submission.** The canonical
final protocol is `classification/case_study_pipeline.py` with a stratified
70/15/15 split, validation-only model/threshold selection, early stopping, and
Phase-3 artifacts under `case_study/artifacts/phase3/`.
