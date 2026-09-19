"""Phase 3: controlled hyperparameter studies, ablations and references.

Design principles
-----------------
* Hyperparameters are selected on validation data only.
* Regression screening uses a fixed subset of the training/validation splits for
  tractable repeated from-scratch experiments. The chosen configuration is then
  retrained on the full 70k training split and evaluated once on the untouched
  15k test split.
* Classification is small enough to use the complete 70/15 train/validation
  splits for screening. Test is evaluated only after validation selection.
* Reference models are pre-specified and are not selected using test results.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from regression.case_study_pipeline import (
    TrainOnlyPreprocessor,
    load_primary_regression_dataset,
    regression_metrics,
    split_regression_dataset,
)
from regression.lightgbm_regression import LightGBMRegression
from classification.case_study_pipeline import (
    binary_metrics,
    split_classification_dataset,
    tune_threshold,
)
from classification.data import (
    DEFAULT_DATA_PATH as CLASSIFICATION_DATA_PATH,
    FEATURE_COLUMNS,
    load_machine_failure_dataset,
)
from classification.lightgbm_classification import LightGBMClassification
from lightgbm_from_scratch.core.binning import create_quantile_bins, bin_matrix
from lightgbm_from_scratch.core.efb import build_efb_plan
from lightgbm_from_scratch.core.histogram import build_feature_histograms
from lightgbm_from_scratch.metrics import average_precision


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "case_study" / "artifacts" / "phase3"
RANDOM_STATE = 42


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fit_regression(model, X, y, X_val, y_val, early_stopping_rounds=15):
    # Historical regressor prints fit/predict messages; keep experiment logs compact.
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
    elapsed = time.perf_counter() - start
    return elapsed


def _predict_regression(model, X):
    with contextlib.redirect_stdout(io.StringIO()):
        return model.predict(X)


def _regression_screen_data():
    X, y = load_primary_regression_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_regression_dataset(
        X, y, random_state=RANDOM_STATE
    )
    pre = TrainOnlyPreprocessor().fit(X_train)
    # Fixed screening samples from train/validation only. Test is never sampled/read
    # by hyperparameter selection code.
    train_idx = X_train.sample(n=5_000, random_state=RANDOM_STATE).index
    val_idx = X_val.sample(n=2_000, random_state=RANDOM_STATE).index
    screen = {
        "X_train": pre.transform(X_train.loc[train_idx]),
        "y_train": y_train.loc[train_idx].to_numpy(),
        "X_val": pre.transform(X_val.loc[val_idx]),
        "y_val": y_val.loc[val_idx].to_numpy(),
    }
    full = {
        "X_train": pre.transform(X_train),
        "y_train": y_train.to_numpy(),
        "X_val": pre.transform(X_val),
        "y_val": y_val.to_numpy(),
        "X_test": pre.transform(X_test),
        "y_test": y_test.to_numpy(),
        "feature_names": list(pre.feature_names_),
    }
    return screen, full


def _regression_base():
    return {
        "n_estimators": 60,
        "learning_rate": 0.08,
        "num_leaves": 7,
        "max_depth": -1,
        "max_bins": 16,
        "min_data_in_leaf": 30,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "feature_fraction": 1.0,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "use_goss": True,
        "use_efb": True,
        "random_state": RANDOM_STATE,
    }


def _evaluate_regression_screen(name, family, value, params, data):
    model = LightGBMRegression(**params)
    seconds = _fit_regression(
        model, data["X_train"], data["y_train"], data["X_val"], data["y_val"], 8
    )
    pred = _predict_regression(model, data["X_val"])
    metrics = regression_metrics(data["y_val"], pred)
    return {
        "name": name,
        "family": family,
        "value": value,
        "validation_mae": metrics["mae"],
        "validation_rmse": metrics["rmse"],
        "validation_r2": metrics["r2"],
        "best_iteration": int(model.best_iteration_),
        "train_seconds": seconds,
        "efb_applied": bool(model.efb_applied),
        "n_efb_bundles": int(len(model.efb_bundles)),
        "params": params,
    }


def regression_hyperparameter_study(screen):
    base = _regression_base()
    families = {
        # Compact one-factor controlled screening. The final winner is always
        # retrained on the full training split before test reporting.
        "num_leaves": [7, 15],
        "learning_rate": [0.05, 0.08],
        "max_bins": [16, 32],
        "min_data_in_leaf": [30, 50],
        "reg_lambda": [1.0, 5.0],
        "feature_fraction": [0.8, 1.0],
    }
    rows = []
    seen = set()
    for family, values in families.items():
        for value in values:
            params = dict(base)
            params[family] = value
            key = tuple(sorted(params.items()))
            if key in seen:
                # Baseline duplicated across controlled families: keep a row per
                # family by copying the already-computed baseline result later.
                baseline = next(r for r in rows if r["name"] == "baseline")
                clone = dict(baseline)
                clone.update(name=f"{family}={value}", family=family, value=value)
                rows.append(clone)
                continue
            name = "baseline" if params == base else f"{family}={value}"
            row = _evaluate_regression_screen(name, family, value, params, screen)
            rows.append(row)
            seen.add(key)

    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "params"} for row in rows])
    family_best = {}
    for family in families:
        subset = frame[frame.family == family].sort_values(
            ["validation_rmse", "train_seconds"], ascending=[True, True]
        )
        family_best[family] = subset.iloc[0]["value"]
    combined = dict(base)
    for family, value in family_best.items():
        if family in ("num_leaves", "max_bins", "min_data_in_leaf"):
            value = int(value)
        combined[family] = value
    combined_row = _evaluate_regression_screen(
        "combined_family_best", "combined", "combined", combined, screen
    )
    rows.append(combined_row)
    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "params"} for row in rows])
    selection = min(rows, key=lambda r: (r["validation_rmse"], r["train_seconds"]))
    return frame, selection, family_best, rows


def regression_ablation(screen, selected_params):
    variants = {
        "selected": {},
        "no_goss": {"use_goss": False},
        "no_efb": {"use_efb": False},
        "no_regularization": {"reg_alpha": 0.0, "reg_lambda": 0.0},
        "feature_fraction_1.0": {"feature_fraction": 1.0},
        "feature_fraction_0.8": {"feature_fraction": 0.8},
    }
    rows = []
    for name, changes in variants.items():
        params = dict(selected_params)
        params.update(changes)
        row = _evaluate_regression_screen(name, "ablation", name, params, screen)
        rows.append(row)
    return pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in rows])


def regression_final_run(full, selected_params):
    params = dict(selected_params)
    params["n_estimators"] = max(180, int(params.get("n_estimators", 60)))
    model = LightGBMRegression(**params)
    seconds = _fit_regression(
        model, full["X_train"], full["y_train"], full["X_val"], full["y_val"], 20
    )
    preds = {
        "train": _predict_regression(model, full["X_train"]),
        "validation": _predict_regression(model, full["X_val"]),
        "test": _predict_regression(model, full["X_test"]),
    }
    targets = {"train": full["y_train"], "validation": full["y_val"], "test": full["y_test"]}
    metrics = {split: regression_metrics(targets[split], preds[split]) for split in preds}
    return model, params, seconds, metrics, preds


def _classification_data():
    X, y = load_machine_failure_dataset(CLASSIFICATION_DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_classification_dataset(
        X, y, random_state=RANDOM_STATE
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def _classification_base():
    return {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "max_bins": 63,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "feature_fraction": 1.0,
        "scale_pos_weight": 1.0,
        "use_goss": True,
        "use_efb": True,
        "random_state": RANDOM_STATE,
    }


def _evaluate_classification_validation(name, family, value, params, data):
    model = LightGBMClassification(**params)
    start = time.perf_counter()
    model.fit(
        data["X_train"], data["y_train"],
        eval_set=(data["X_val"], data["y_val"]), early_stopping_rounds=25,
    )
    seconds = time.perf_counter() - start
    p = model.predict_proba(data["X_val"])[:, 1]
    threshold, _ = tune_threshold(data["y_val"].to_numpy(), p)
    metrics = binary_metrics(data["y_val"].to_numpy(), p, threshold)
    return {
        "name": name,
        "family": family,
        "value": value,
        "validation_pr_auc": float(average_precision(data["y_val"].to_numpy(), p)),
        "validation_roc_auc": metrics["roc_auc"],
        "validation_f1": metrics["f1"],
        "validation_precision": metrics["precision"],
        "validation_recall": metrics["recall"],
        "selected_threshold": threshold,
        "best_iteration": int(model.best_iteration_),
        "train_seconds": seconds,
        "efb_applied": bool(model.efb_applied_),
        "n_efb_bundles": int(len(model.feature_bundles_)),
        "params": params,
    }


def classification_hyperparameter_study(data):
    base = _classification_base()
    families = {
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.03, 0.05, 0.10],
        "max_bins": [31, 63, 127],
        "min_child_samples": [10, 20, 50],
        "reg_lambda": [0.0, 1.0, 5.0],
        "feature_fraction": [0.7, 0.85, 1.0],
        "scale_pos_weight": [1.0, 2.0, 3.0, 5.0],
    }
    rows = []
    seen = {}
    for family, values in families.items():
        for value in values:
            params = dict(base)
            params[family] = value
            key = tuple(sorted(params.items()))
            if key in seen:
                clone = dict(seen[key])
                clone.update(name=f"{family}={value}", family=family, value=value)
                rows.append(clone)
                continue
            name = "baseline" if params == base else f"{family}={value}"
            row = _evaluate_classification_validation(name, family, value, params, data)
            rows.append(row)
            seen[key] = row

    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in rows])
    family_best = {}
    for family in families:
        subset = frame[frame.family == family].sort_values(
            ["validation_pr_auc", "validation_f1", "validation_recall"],
            ascending=[False, False, False],
        )
        family_best[family] = subset.iloc[0]["value"]
    combined = dict(base)
    for family, value in family_best.items():
        if family in ("num_leaves", "max_bins", "min_child_samples"):
            value = int(value)
        combined[family] = value
    combined_row = _evaluate_classification_validation(
        "combined_family_best", "combined", "combined", combined, data
    )
    rows.append(combined_row)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in rows])
    selection = max(
        rows,
        key=lambda r: (r["validation_pr_auc"], r["validation_f1"], r["validation_recall"]),
    )
    return frame, selection, family_best, rows


def classification_ablation(data, selected_params):
    variants = {
        "selected": {},
        "no_goss": {"use_goss": False},
        "no_efb": {"use_efb": False},
        "no_regularization": {"reg_alpha": 0.0, "reg_lambda": 0.0},
        "no_positive_weighting": {"scale_pos_weight": 1.0},
        "feature_fraction_1.0": {"feature_fraction": 1.0},
    }
    rows = []
    for name, changes in variants.items():
        params = dict(selected_params)
        params.update(changes)
        rows.append(_evaluate_classification_validation(name, "ablation", name, params, data))
    return pd.DataFrame([{k: v for k, v in r.items() if k != "params"} for r in rows])


def classification_final_run(data, selected_params):
    params = dict(selected_params)
    params["n_estimators"] = max(450, int(params.get("n_estimators", 350)))
    model = LightGBMClassification(**params)
    start = time.perf_counter()
    model.fit(
        data["X_train"], data["y_train"],
        eval_set=(data["X_val"], data["y_val"]), early_stopping_rounds=30,
    )
    seconds = time.perf_counter() - start
    val_p = model.predict_proba(data["X_val"])[:, 1]
    threshold, threshold_frame = tune_threshold(data["y_val"].to_numpy(), val_p)
    split_metrics = {}
    probabilities = {}
    for split in ("train", "validation", "test"):
        p = model.predict_proba(data[f"X_{split if split != 'validation' else 'val'}"])[:, 1]
        probabilities[split] = p
        y = data[f"y_{split if split != 'validation' else 'val'}"].to_numpy()
        split_metrics[split] = binary_metrics(y, p, threshold)
    return model, params, seconds, threshold, threshold_frame, split_metrics, probabilities


def efb_sparse_microbenchmark():
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows, n_features, max_bins = 30_000, 40, 16
    X = np.zeros((n_rows, n_features), dtype=float)
    active_feature = rng.integers(0, n_features, size=n_rows)
    X[np.arange(n_rows), active_feature] = rng.normal(loc=2.0, scale=0.5, size=n_rows)
    cuts = create_quantile_bins(X, max_bins, quantile_style="boundaries")
    X_bin = bin_matrix(X, cuts, missing_bin="last", max_bins=max_bins)
    plan = build_efb_plan(X, X_bin, missing_bin="last", max_bins=max_bins)
    gradients = rng.normal(size=n_rows)
    hessians = rng.uniform(0.2, 1.2, size=n_rows)
    rows = np.arange(n_rows)
    weights = np.ones(n_rows)
    features = np.arange(n_features)

    def run(efb_plan):
        start = time.perf_counter()
        out = build_feature_histograms(
            X_bin, gradients, hessians, rows, weights, features,
            efb_plan=efb_plan, include_count=True, missing_bin=max_bins,
        )
        return time.perf_counter() - start, out

    direct_times, efb_times = [], []
    direct = bundled = None
    for _ in range(5):
        t, direct = run(None); direct_times.append(t)
        t, bundled = run(plan); efb_times.append(t)
    max_error = max(float(np.max(np.abs(direct[f] - bundled[f]))) for f in features)
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_bundles": len(plan.groups),
        "compression_ratio_features_per_bundle": n_features / len(plan.groups),
        "direct_histogram_seconds_median": float(np.median(direct_times)),
        "efb_histogram_seconds_median": float(np.median(efb_times)),
        "speedup_direct_over_efb": float(np.median(direct_times) / np.median(efb_times)),
        "max_histogram_absolute_error": max_error,
        "exact_equivalence": bool(max_error < 1e-10),
    }


def reference_comparison(reg_full, reg_final, cls_data, cls_final):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    rows_reg = []
    scratch_model, scratch_params, scratch_seconds, scratch_metrics, _ = reg_final
    rows_reg.append({
        "model": "LightGBM From Scratch",
        **scratch_metrics["test"],
        "train_seconds": scratch_seconds,
        "selection": "validation",
    })

    refs_reg = [
        ("Dummy Mean", DummyRegressor(strategy="mean")),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]
    for name, model in refs_reg:
        start = time.perf_counter(); model.fit(reg_full["X_train"], reg_full["y_train"]); seconds = time.perf_counter()-start
        pred = model.predict(reg_full["X_test"])
        rows_reg.append({"model": name, **regression_metrics(reg_full["y_test"], pred), "train_seconds": seconds, "selection": "pre-specified"})

    official_reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=float(scratch_params["learning_rate"]),
        num_leaves=int(scratch_params["num_leaves"]),
        max_depth=int(scratch_params["max_depth"]),
        max_bin=int(scratch_params["max_bins"]),
        min_child_samples=int(scratch_params["min_data_in_leaf"]),
        reg_alpha=float(scratch_params["reg_alpha"]),
        reg_lambda=float(scratch_params["reg_lambda"]),
        colsample_bytree=float(scratch_params["feature_fraction"]),
        data_sample_strategy="goss" if scratch_params["use_goss"] else "bagging",
        top_rate=float(scratch_params["top_rate"]),
        other_rate=float(scratch_params["other_rate"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    start=time.perf_counter()
    official_reg.fit(
        reg_full["X_train"], reg_full["y_train"],
        eval_set=[(reg_full["X_val"], reg_full["y_val"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    seconds=time.perf_counter()-start
    pred=official_reg.predict(reg_full["X_test"], num_iteration=official_reg.best_iteration_)
    rows_reg.append({"model":f"Official LightGBM {lgb.__version__}", **regression_metrics(reg_full["y_test"], pred), "train_seconds":seconds, "selection":"validation"})

    # Classification references: every probabilistic model tunes threshold on
    # validation, then reports untouched test with the frozen threshold.
    rows_cls = []
    scratch_model, scratch_params, scratch_seconds, scratch_threshold, _, scratch_metrics, _ = cls_final
    rows_cls.append({
        "model":"LightGBM From Scratch", **scratch_metrics["test"],
        "train_seconds": scratch_seconds, "selected_threshold": scratch_threshold,
        "selection":"validation",
    })

    refs_cls = [
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))),
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]
    for name, model in refs_cls:
        start=time.perf_counter(); model.fit(cls_data["X_train"], cls_data["y_train"]); seconds=time.perf_counter()-start
        val_p=model.predict_proba(cls_data["X_val"])[:,1]
        threshold,_=tune_threshold(cls_data["y_val"].to_numpy(),val_p)
        test_p=model.predict_proba(cls_data["X_test"])[:,1]
        m=binary_metrics(cls_data["y_test"].to_numpy(),test_p,threshold)
        rows_cls.append({"model":name, **m, "train_seconds":seconds, "selected_threshold":threshold, "selection":"validation"})

    official_cls=lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=float(scratch_params["learning_rate"]),
        num_leaves=int(scratch_params["num_leaves"]),
        max_depth=int(scratch_params["max_depth"]),
        max_bin=int(scratch_params["max_bins"]),
        min_child_samples=int(scratch_params["min_child_samples"]),
        reg_alpha=float(scratch_params["reg_alpha"]),
        reg_lambda=float(scratch_params["reg_lambda"]),
        colsample_bytree=float(scratch_params["feature_fraction"]),
        scale_pos_weight=float(scratch_params["scale_pos_weight"]),
        data_sample_strategy="goss" if scratch_params["use_goss"] else "bagging",
        top_rate=float(scratch_params["top_rate"]),
        other_rate=float(scratch_params["other_rate"]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    start=time.perf_counter()
    # Use NumPy arrays here because the source CSV contains feature names with
    # JSON-special characters (for example square brackets), which official
    # LightGBM rejects as feature names even though the numeric data are valid.
    X_train_np = cls_data["X_train"].to_numpy()
    X_val_np = cls_data["X_val"].to_numpy()
    X_test_np = cls_data["X_test"].to_numpy()
    y_train_np = cls_data["y_train"].to_numpy()
    y_val_np = cls_data["y_val"].to_numpy()
    y_test_np = cls_data["y_test"].to_numpy()
    official_cls.fit(
        X_train_np, y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        callbacks=[lgb.early_stopping(50,verbose=False)],
    )
    seconds=time.perf_counter()-start
    val_p=official_cls.predict_proba(X_val_np,num_iteration=official_cls.best_iteration_)[:,1]
    threshold,_=tune_threshold(y_val_np,val_p)
    test_p=official_cls.predict_proba(X_test_np,num_iteration=official_cls.best_iteration_)[:,1]
    m=binary_metrics(y_test_np,test_p,threshold)
    rows_cls.append({"model":f"Official LightGBM {lgb.__version__}", **m, "train_seconds":seconds, "selected_threshold":threshold, "selection":"validation"})

    return pd.DataFrame(rows_reg), pd.DataFrame(rows_cls)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[1/7] Regression controlled study")
    reg_screen, reg_full = _regression_screen_data()
    reg_hp, reg_selected, reg_family_best, _ = regression_hyperparameter_study(reg_screen)
    reg_hp.to_csv(OUTPUT_ROOT / "regression_hyperparameter_study.csv", index=False)
    _json_dump(OUTPUT_ROOT / "regression_selection.json", {
        "screening_train_rows": len(reg_screen["y_train"]),
        "screening_validation_rows": len(reg_screen["y_val"]),
        "selection_metric": "validation_rmse",
        "family_best": reg_family_best,
        "selected_name": reg_selected["name"],
        "selected_params": reg_selected["params"],
        "selected_validation_rmse": reg_selected["validation_rmse"],
    })

    print("[2/7] Regression ablation")
    reg_ablation = regression_ablation(reg_screen, reg_selected["params"])
    reg_ablation.to_csv(OUTPUT_ROOT / "regression_ablation.csv", index=False)

    print("[3/7] Regression final full-data run")
    reg_final = regression_final_run(reg_full, reg_selected["params"])
    reg_model, reg_params, reg_seconds, reg_metrics, reg_preds = reg_final
    _json_dump(OUTPUT_ROOT / "regression_final.json", {
        "selected_from": "validation-only screening",
        "full_train_rows": len(reg_full["y_train"]),
        "full_validation_rows": len(reg_full["y_val"]),
        "test_rows": len(reg_full["y_test"]),
        "params": reg_params,
        "best_iteration": int(reg_model.best_iteration_),
        "best_validation_rmse": float(reg_model.best_score_),
        "train_seconds": reg_seconds,
        "metrics": reg_metrics,
        "efb_applied": bool(reg_model.efb_applied),
        "n_efb_bundles": len(reg_model.efb_bundles),
    })
    pd.DataFrame({
        "feature": reg_full["feature_names"],
        "gain_importance": reg_model.get_feature_importance(),
    }).sort_values("gain_importance",ascending=False).to_csv(OUTPUT_ROOT/"regression_final_feature_importance.csv",index=False)

    print("[4/7] Classification controlled study")
    cls_data = _classification_data()
    cls_hp, cls_selected, cls_family_best, _ = classification_hyperparameter_study(cls_data)
    cls_hp.to_csv(OUTPUT_ROOT / "classification_hyperparameter_study.csv", index=False)
    _json_dump(OUTPUT_ROOT / "classification_selection.json", {
        "selection_metric": "validation_pr_auc (tie: F1, Recall)",
        "family_best": cls_family_best,
        "selected_name": cls_selected["name"],
        "selected_params": cls_selected["params"],
        "selected_validation_pr_auc": cls_selected["validation_pr_auc"],
        "selected_validation_f1": cls_selected["validation_f1"],
    })

    print("[5/7] Classification ablation + final run")
    cls_ablation = classification_ablation(cls_data, cls_selected["params"])
    cls_ablation.to_csv(OUTPUT_ROOT / "classification_ablation.csv", index=False)
    cls_final = classification_final_run(cls_data, cls_selected["params"])
    cls_model, cls_params, cls_seconds, cls_threshold, cls_threshold_frame, cls_metrics, cls_prob = cls_final
    cls_threshold_frame.to_csv(OUTPUT_ROOT / "classification_final_threshold_analysis.csv", index=False)
    _json_dump(OUTPUT_ROOT / "classification_final.json", {
        "params": cls_params,
        "best_iteration": int(cls_model.best_iteration_),
        "best_validation_binary_logloss": float(cls_model.best_score_),
        "train_seconds": cls_seconds,
        "selected_threshold": cls_threshold,
        "threshold_selection_split": "validation",
        "metrics": cls_metrics,
        "efb_applied": bool(cls_model.efb_applied_),
        "n_efb_bundles": len(cls_model.feature_bundles_),
    })
    pd.DataFrame({"feature": list(FEATURE_COLUMNS), "gain_importance": cls_model.get_feature_importance()}).sort_values("gain_importance",ascending=False).to_csv(OUTPUT_ROOT/"classification_final_feature_importance.csv",index=False)

    print("[6/7] EFB sparse microbenchmark")
    efb = efb_sparse_microbenchmark()
    _json_dump(OUTPUT_ROOT / "efb_sparse_microbenchmark.json", efb)

    print("[7/7] Pre-specified reference comparison")
    ref_reg, ref_cls = reference_comparison(reg_full, reg_final, cls_data, cls_final)
    ref_reg.to_csv(OUTPUT_ROOT / "regression_reference_comparison.csv", index=False)
    ref_cls.to_csv(OUTPUT_ROOT / "classification_reference_comparison.csv", index=False)

    summary = {
        "regression": {
            "selected_params": reg_params,
            "best_iteration": int(reg_model.best_iteration_),
            "metrics": reg_metrics,
            "train_seconds": reg_seconds,
        },
        "classification": {
            "selected_params": cls_params,
            "best_iteration": int(cls_model.best_iteration_),
            "selected_threshold": cls_threshold,
            "metrics": cls_metrics,
            "train_seconds": cls_seconds,
        },
        "efb_sparse_microbenchmark": efb,
        "selection_integrity": {
            "hyperparameter_selection": "validation only",
            "threshold_selection": "validation only",
            "test_role": "final reporting / pre-specified reference comparison",
        },
    }
    _json_dump(OUTPUT_ROOT / "phase3_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
