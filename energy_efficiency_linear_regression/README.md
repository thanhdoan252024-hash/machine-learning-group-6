# Energy Efficiency Building Performance Prediction using Linear Regression from Scratch

This project implements a complete linear regression workflow for predicting both Heating Load (Y1) and Cooling Load (Y2) from architectural design parameters in the energy_efficiency_building_heating_cooling_load_dataset.csv file.

## Dataset
The dataset used here is the file energy_efficiency_building_heating_cooling_load_dataset.csv found in the data directory. It is a real Energy Efficiency dataset with architectural variables X1..X8 and target-like columns Y1 (Heating Load) and Y2 (Cooling Load). The model is trained in safe target-pair runs: Y2 is excluded when predicting Y1, and Y1 is excluded when predicting Y2.

## Target
The primary targets are Y1 (Heating Load) and Y2 (Cooling Load). For each run, the opposite target must be excluded from the predictor matrix to prevent target leakage.

## Features
The selected candidate predictors are the architectural features X1..X8: Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, and Glazing Area Distribution.

## Cleaning
The project validates Y1 as numeric, removes invalid target rows, and checks dataset schema and feature type consistency.

## Leakage prevention
Y2 is not allowed in the predictor X matrix. This is a strict leakage control rule.

## Linear Regression implementation
The project uses a custom NumPy-based linear regression class with Gradient Descent and a Mean Baseline comparator.

## Evaluation
The workflow reports MAE, RMSE, and R2 for the baseline and Linear Regression model.

## Results
The results are generated from the model training process and stored under the outputs directory.

## How to run
Install dependencies and run the stages in order from the project root:

```bash
python -m pip install -r requirements.txt
python src/stage2_cleaning_eda.py
python src/stage3_preprocessing.py
python src/stage5_training_eval.py
python src/stage6_report.py
python src/stage7_final_artifacts.py
python src/stage8_validation.py
python -m pytest -q
```

Stage 8 checks the current Y1/Y2 metrics, all target-specific figures and coefficient tables, leakage rules, and the pytest exit code. It fails when any required artifact is missing or stale.

## Limitations
The linear model may exhibit residual patterns and heteroscedasticity because building thermal behavior is nonlinear and influenced by complex heat transfer phenomena.
