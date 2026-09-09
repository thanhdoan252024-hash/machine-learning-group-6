# Energy Efficiency Building Performance Prediction using Linear Regression from Scratch

This project implements a complete linear regression workflow for predicting Heating Load (Y1) from architectural design parameters in the energy_efficiency_building_heating_cooling_load_dataset.csv file.

## Dataset
The dataset used here is the file energy_efficiency_building_heating_cooling_load_dataset.csv found in the data directory. It is a real Energy Efficiency dataset with architectural variables X1..X8 and target Y1 (Heating Load). Y2 (Cooling Load) is a second target-like column and must be excluded from the predictor matrix to prevent target leakage.

## Target
The main target is Y1. Y2 is a second target-like variable and must be excluded from the predictor matrix to prevent target leakage.

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
Use the notebook or run the project scripts in the source directory. Ensure the Python dependencies in requirements.txt are installed.

## Limitations
The linear model may exhibit residual patterns and heteroscedasticity because building thermal behavior is nonlinear and influenced by complex heat transfer phenomena.
