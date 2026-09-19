# Phase 3 Results — Controlled Tuning, Ablation and Reference Comparison

## Experimental integrity

Phase 3 preserves a strict train/validation/test protocol. Hyperparameters and classification thresholds are selected from validation data only. The test split is held out for final reporting and for pre-specified reference-model comparison. The from-scratch implementation does not call the `lightgbm` package during training; the official package is used only as a reference benchmark.

## Submission protocol clarifications

### Library boundary

The from-scratch estimators do not call scikit-learn estimators or the official `lightgbm` package. NumPy/pandas are used for numerical/data handling; `sklearn.model_selection.train_test_split` is used for splitting. Random Forest, Logistic Regression, DummyRegressor/StandardScaler and official LightGBM appear only in the pre-specified reference-comparison stage. Matplotlib/seaborn are visualization-only.

### Regression screening subset -> full retrain

Controlled Regression hyperparameter screening uses a fixed **5,000-row train subset + 2,000-row validation subset**, sampled only from the already-separated train/validation partitions. The test split is never read by screening. After configuration freeze, the model is retrained on the full **70,000 train** rows, early-stopped on the full **15,000 validation** rows, then reported once on **15,000 test** rows.

### Missing values: implementation capability vs final metric protocol

The Regression core contains a native missing bin/branch and unit tests verify that behavior. The primary 100k Regression case study nevertheless applies **train-only mean imputation for numeric columns and train-only mode imputation for categorical columns** before model fitting. Therefore the final RMSE/R² quantify the imputed-data protocol; missing-bin capability is supported by unit tests rather than by the 100k final metric. The six Classification features used in the final experiment contain no missing values.

### Official LightGBM reference budget

The official reference is **not a strict equal-iteration-budget benchmark**. Official LightGBM is allowed `n_estimators=2000` with early stopping 50 so that validation selects its stopping point. In the reported run, Official stops at **151** iterations for Regression and **230** for Classification; both are below the corresponding scratch caps (**240** and **450**). Thus the large official cap was not exhausted, but training-time ratios should still be interpreted as Python-vs-native implementation comparisons rather than iso-compute measurements.

## Regression final configuration

Primary scenario: **pre-exam student score prediction** with post-exam leakage features removed.

Final validation-selected configuration:

- learning_rate = 0.08
- num_leaves = 15
- max_bins = 32
- min_data_in_leaf = 30
- reg_lambda = 1.0
- feature_fraction = 0.8
- GOSS = enabled
- EFB = enabled (not activated on this dense dataset)
- boosting budget = 240, early stopping selected 125 trees

Final test metrics:

| Metric | Phase 2 | Phase 3 |
|---|---:|---:|
| MAE | 7.5452 | **7.3425** |
| RMSE | 9.4816 | **9.2096** |
| R² | 0.6360 | **0.6566** |

Phase 3 reduces test RMSE by about **2.87%** and improves R² by about **0.0206** absolute. Train, validation and test metrics remain close, indicating limited overfitting under the final configuration.

### Regression ablation

On the fixed screening split, disabling GOSS worsened validation RMSE from about 9.49 to 9.61 and increased runtime. Disabling regularization also worsened validation RMSE. EFB ON/OFF produced identical metrics because all 37 features remain separate bundles on this dense dataset. `feature_fraction=0.8` produced the strongest screening result and was retained in the final model.

## Classification final configuration

Primary scenario: **machine-failure predictive maintenance** with 3.39% positives and stratified train/validation/test splits.

Final validation-selected configuration:

- learning_rate = 0.05
- num_leaves = 15
- max_depth = 5
- max_bins = 127
- min_child_samples = 20
- reg_lambda = 1.0
- feature_fraction = 1.0
- scale_pos_weight = 1.0
- GOSS = enabled
- EFB = enabled (not activated on this dense dataset)
- boosting budget = 450, early stopping selected 195 trees
- decision threshold = **0.22**, selected on validation only

Final test metrics:

| Metric | Phase 2 | Phase 3 |
|---|---:|---:|
| Precision | 0.6905 | 0.6129 |
| Recall | 0.5686 | **0.7451** |
| F1 | 0.6237 | **0.6726** |
| Balanced Accuracy | 0.7798 | **0.8643** |
| ROC-AUC | 0.9629 | **0.9637** |
| PR-AUC / Average Precision | 0.7349 | **0.7825** |
| False negatives | 22 | **13** |

The lower threshold intentionally trades precision for recall. For predictive maintenance this is meaningful because missed failures are reduced from 22 to 13 on the held-out test split.

### Classification ablation

GOSS improved validation PR-AUC from about 0.7532 without GOSS to 0.7807 with GOSS. Increasing positive-class weighting to 5 did not improve the primary validation objective. EFB ON/OFF again produced identical predictions because the six input features are dense and cannot be bundled.

## EFB sparse microbenchmark

To evaluate EFB under conditions for which it is designed, a synthetic mutually-exclusive sparse dataset was generated with 30,000 rows and 40 features.

- 40 original sparse features were compressed into **1 bundle**.
- Direct and EFB-reconstructed histograms differed by at most **3.64e-11**, confirming numerical equivalence.
- Median histogram construction time improved from about **0.0157 s** to **0.00369 s**, or approximately **4.27x speedup**.

This reconciles the two main-dataset results: EFB is implemented correctly, but it naturally provides little value for dense data without mutually-exclusive sparse features.

## Reference comparison

### Regression

| Model | Test RMSE | Test R² | Training time (s) |
|---|---:|---:|---:|
| LightGBM From Scratch | **9.20960** | **0.65659** | 31.99 |
| Official LightGBM 4.6.0 | **9.20946** | **0.65660** | 0.78 |
| Random Forest | 9.51932 | 0.63310 | 18.20 |
| Dummy Mean | 15.71611 | ~0.000 | <0.01 |

The from-scratch model reproduces the official LightGBM regression quality extremely closely: the absolute RMSE difference is only about **0.00014** on the test set. However, the optimized official implementation is about **41x faster** in this environment.

### Classification

| Model | Recall | F1 | ROC-AUC | PR-AUC | Training time (s) |
|---|---:|---:|---:|---:|---:|
| LightGBM From Scratch | 0.7451 | 0.6726 | 0.9637 | **0.7825** | 14.96 |
| Official LightGBM 4.6.0 | **0.7843** | **0.7477** | 0.9676 | **0.8001** | 0.17 |
| Random Forest | 0.6667 | 0.7010 | **0.9779** | 0.7763 | 0.56 |
| Logistic Regression | 0.4902 | 0.4762 | 0.8993 | 0.4345 | <0.01 |

The scratch classifier is close to the official LightGBM in PR-AUC (gap about **0.0176**) and slightly exceeds Random Forest in PR-AUC, although Random Forest has a higher F1 on this particular held-out split. Official LightGBM remains substantially faster (about **90x** here), as expected from its optimized native implementation.

## Phase 3 conclusion

Phase 3 provides the evidence needed to freeze the final configurations for reporting. The from-scratch models show credible predictive behavior, controlled model selection, measurable GOSS contributions, correct EFB behavior under sparse conditions, and results that are directionally consistent with official LightGBM. The remaining gap is primarily computational efficiency rather than basic predictive correctness.
