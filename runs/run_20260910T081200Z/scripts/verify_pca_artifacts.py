"""Verify a completed UCI HAR PCA run without fitting a scaler or PCA again."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


ARRAY_NAMES = [f"X_{split}_{rep}" for split in ("train", "test")
               for rep in ("scaled", "pca90", "pca95")] + ["y_train", "y_test"]
PARAM_NAMES = {"train_mean", "train_scale", "zero_variance_mask", "eigenvalues",
               "eigenvectors", "explained_variance_ratio", "cumulative_variance", "k90", "k95"}
TABLE_NAMES = ["pca_candidate_comparison", "pca_final_summary", "pca_reconstruction_analysis"]


class Verification:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checks: list[dict] = []
        self.required: set[str] = set()

    def check(self, name, condition, detail=None):
        row = {"check": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        self.checks.append(row)
        return row["passed"]

    def close(self, name, actual, expected, atol=1e-8):
        a, b = np.asarray(actual), np.asarray(expected)
        return self.check(name, a.shape == b.shape and np.allclose(a, b, rtol=1e-8, atol=atol))

    def required_file(self, relative):
        self.required.add(relative)
        path = self.root / relative
        return path if self.check(f"file: {relative}", path.is_file() and path.stat().st_size > 0) else None

    def load_json(self, relative):
        path = self.required_file(relative)
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.check(f"JSON content: {relative}", isinstance(value, (dict, list)) and bool(value))
            return value
        except (ValueError, OSError):
            self.check(f"JSON readable: {relative}", False)
            return None

    def section(self, name, action):
        try:
            action()
        except Exception as exc:
            # Avoid including array values or input data in diagnostics.
            self.check(name, False, f"Validation could not complete ({type(exc).__name__}).")


def verify_arrays(v, arrays, params, dataset_dir):
    if not v.check("required arrays and parameters loaded", set(arrays) == set(ARRAY_NAMES)
                   and PARAM_NAMES.issubset(params)):
        return
    p = 561
    for key in PARAM_NAMES:
        v.check(f"finite parameter: {key}", np.isfinite(params[key]).all())
    for key in PARAM_NAMES - {"eigenvectors", "k90", "k95"}:
        v.check(f"parameter shape: {key}", params[key].shape == (p,))
    v.check("eigenvector shape", params["eigenvectors"].shape == (p, p))
    for key in ("k90", "k95"):
        v.check(f"integer scalar: {key}", params[key].shape == () and params[key].dtype.kind in "iu")
    k90, k95 = int(params["k90"]), int(params["k95"])
    if not v.check("component counts valid", 1 <= k90 <= k95 <= p):
        return
    eig, basis, cumulative = (params[k] for k in ("eigenvalues", "eigenvectors", "cumulative_variance"))
    v.check("positive total variance and nonnegative spectrum", eig.sum() > 0 and np.min(eig) >= -1e-10)
    v.check("descending eigenvalues", np.all(np.diff(eig) <= 1e-8))
    v.close("explained variance ratio", params["explained_variance_ratio"], eig / eig.sum())
    v.close("cumulative variance", cumulative, np.cumsum(eig / eig.sum()))
    v.close("orthonormal eigenbasis", basis.T @ basis, np.eye(p))
    for threshold, k in ((0.90, k90), (0.95, k95)):
        v.check(f"minimal k for threshold {threshold}", k == int(np.searchsorted(cumulative, threshold) + 1))
    v.check("positive train scales", np.all(params["train_scale"] > 0))
    v.check("boolean zero-variance mask", params["zero_variance_mask"].dtype.kind == "b")
    for split, n in (("train", 7352), ("test", 2947)):
        for rep, width in (("scaled", p), ("pca90", k90), ("pca95", k95)):
            value = arrays[f"X_{split}_{rep}"]
            v.check(f"shape: {split} {rep}", value.shape == (n, width))
            v.check(f"finite: {split} {rep}", np.isfinite(value).all())
        y = arrays[f"y_{split}"]
        v.check(f"labels: {split}", y.shape == (n,) and y.dtype.kind in "iu"
                and np.array_equal(np.unique(y), np.arange(1, 7)))
        v.close(f"nested PCA basis: {split}", arrays[f"X_{split}_pca90"], arrays[f"X_{split}_pca95"][:, :k90])
        projected = arrays[f"X_{split}_scaled"] @ basis[:, :k95]
        v.close(f"persisted projection PCA95: {split}", arrays[f"X_{split}_pca95"], projected)
        v.close(f"persisted projection PCA90: {split}", arrays[f"X_{split}_pca90"], projected[:, :k90])
    train = arrays["X_train_scaled"]
    v.close("train is centered", train.mean(axis=0), np.zeros(p))
    mask = params["zero_variance_mask"].astype(bool)
    v.close("train unit standard deviation", train.std(axis=0)[~mask], np.ones((~mask).sum()))
    v.close("constant features have zero variance", train.var(axis=0)[mask], np.zeros(mask.sum()))
    covariance = train.T @ train / (len(train) - 1)
    v.close("saved eigensystem matches independent train covariance", covariance @ basis, basis * eig)
    v.close("total eigenvalue sum matches train covariance trace", eig.sum(), np.trace(covariance))
    if dataset_dir is not None:
        for split in ("train", "test"):
            raw = np.loadtxt(dataset_dir / split / f"X_{split}.txt", dtype=np.float64)
            labels = np.loadtxt(dataset_dir / split / f"y_{split}.txt", dtype=np.int64)
            v.close(f"raw dataset scaling: {split}", arrays[f"X_{split}_scaled"],
                    (raw - params["train_mean"]) / params["train_scale"])
            v.check(f"raw labels unchanged: {split}", np.array_equal(labels, arrays[f"y_{split}"]))
            if split == "train":
                raw_std = raw.std(axis=0)
                raw_mask = raw_std < 1e-12
                v.close("train-only scaler mean", params["train_mean"], raw.mean(axis=0))
                v.close("train-only scaler scale", params["train_scale"], np.where(raw_mask, 1., raw_std))
                v.check("train-only zero-variance mask", np.array_equal(params["zero_variance_mask"], raw_mask))


def verify_tables(v, arrays, params, tables):
    if not v.check("all CSV tables loaded", set(tables) == set(TABLE_NAMES)):
        return
    eig, cumulative, p = params["eigenvalues"], params["cumulative_variance"], 561
    train, test = arrays["X_train_scaled"], arrays["X_test_scaled"]
    test_scores = test @ params["eigenvectors"]
    test_energy = float(np.mean(test ** 2))
    test_projected_energy = np.cumsum(np.sum(test_scores ** 2, axis=0)) / test.size

    def expected(k):
        return [float(cumulative[k - 1]), 1. - k / p,
                float((len(train) - 1) * eig[k:].sum() / train.size),
                float(test_energy - test_projected_energy[k - 1])]

    rec = tables["pca_reconstruction_analysis"]
    v.check("reconstruction rows ordered and unique", len(rec) > 1 and np.all(np.diff(rec["k"]) > 0))
    for _, row in rec.iterrows():
        k = int(row["k"])
        if not v.check(f"reconstruction k valid: {k}", row["k"] == k and 1 <= k <= p):
            continue
        v.close(f"reconstruction metrics: k={k}", row[["Train retained variance", "Reduction ratio", "Train MSE", "Test MSE"]].to_numpy(dtype=float), expected(k))
    v.check("full-basis reconstruction row exists", p in set(rec["k"]))
    candidates = tables["pca_candidate_comparison"]
    v.check("four candidate configurations", len(candidates) == 4 and set(candidates["Configuration"]) == {"PCA80", "PCA90", "PCA95", "PCA99"})
    for _, row in candidates.iterrows():
        threshold = int(row["Configuration"][3:]) / 100.
        k = int(np.searchsorted(cumulative, threshold) + 1)
        v.close(f"candidate selection: {row['Configuration']}", [row["Target variance"], row["k"], row["Dimensions reduced"]], [threshold, k, p - k])
        v.close(f"candidate metrics: {row['Configuration']}", row[["Actual retained variance", "Reduction ratio", "Train reconstruction MSE", "Test reconstruction MSE"]].to_numpy(dtype=float), expected(k))
    summary = tables["pca_final_summary"]
    expected_labels = {"Original standardized", "PCA90", "PCA95"}
    v.check("three final representations", len(summary) == 3 and set(summary["Dataset representation"]) == expected_labels)
    for _, row in summary.iterrows():
        label = row["Dataset representation"]
        k = p if label == "Original standardized" else int(params["k" + label[3:]])
        v.close(f"final summary: {label}", row[["Dimensions", "Retained variance", "Reduction ratio"]].to_numpy(dtype=float), [k, cumulative[k - 1], 1. - k / p])


def verify_evidence(v):
    notebook = v.load_json("notebook.executed.ipynb")
    if notebook is not None:
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        v.check("all 26 code cells executed", len(cells) == 26 and all(cell.get("execution_count") is not None for cell in cells))
        outputs = [output for cell in cells for output in cell.get("outputs", [])]
        v.check("notebook has no error outputs", not any(output.get("output_type") == "error" for output in outputs))
        text = "\n".join("".join(output.get("text", [])) for output in outputs)
        passes = set(int(cp) for cp in re.findall(r"CHECKPOINT\s+(\d+)[^\n]*PASS", text))
        v.check("notebook checkpoints 1 through 10 passed", set(range(1, 11)).issubset(passes))
    figures = sorted((v.root / "figures").glob("*.png"))
    v.check("13 exported PNG figures", len(figures) == 13)
    for figure in figures:
        relative = figure.relative_to(v.root).as_posix()
        path = v.required_file(relative)
        if path is not None:
            with path.open("rb") as stream:
                signature = stream.read(8)
            v.check(f"PNG signature: {figure.name}", signature == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 8)
    for relative in ("evidence/run_metrics.json", "evidence/checkpoints.json"):
        v.load_json(relative)


def verify_manifest(v):
    manifest = v.load_json("artifact_manifest.json")
    if manifest is None:
        return
    entries = manifest.get("files")
    if not v.check("manifest files schema", isinstance(entries, list) and bool(entries)):
        return
    covered = set()
    for entry in entries:
        relative = entry.get("path", "")
        posix = PurePosixPath(relative)
        safe = bool(relative) and not posix.is_absolute() and ".." not in posix.parts and "\\" not in relative
        path = (v.root / relative).resolve()
        safe = safe and path.is_relative_to(v.root)
        if not v.check(f"manifest safe unique path: {relative}", safe and relative not in covered):
            continue
        covered.add(relative)
        if not v.check(f"manifest target exists: {relative}", path.is_file()):
            continue
        v.check(f"manifest size: {relative}", type(entry.get("size_bytes")) is int and path.stat().st_size == entry["size_bytes"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        v.check(f"manifest SHA256: {relative}", bool(re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))) and digest.hexdigest() == entry["sha256"])
    missing = sorted(v.required - {"artifact_manifest.json"} - covered)
    v.check("manifest covers all required artifacts", not missing, missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    v = Verification(args.run_dir)
    arrays, params, tables = {}, {}, {}
    for name in ARRAY_NAMES:
        path = v.required_file(f"outputs/{name}.npy")
        if path is not None:
            v.section(f"load array: {name}", lambda name=name, path=path: arrays.update({name: np.load(path, allow_pickle=False)}))
    path = v.required_file("outputs/pca_from_scratch_parameters.npz")
    if path is not None:
        def load_parameters():
            with np.load(path, allow_pickle=False) as archive:
                params.update({key: archive[key] for key in archive.files})
        v.section("load parameter archive", load_parameters)
    for name in TABLE_NAMES:
        path = v.required_file(f"outputs/{name}.csv")
        if path is not None:
            v.section(f"load CSV: {name}", lambda name=name, path=path: tables.update({name: pd.read_csv(path)}))
    v.required_file("outputs/README.txt")
    v.section("array and parameter verification", lambda: verify_arrays(v, arrays, params, args.dataset_dir))
    v.section("CSV consistency verification", lambda: verify_tables(v, arrays, params, tables))
    v.section("execution and figure evidence verification", lambda: verify_evidence(v))
    if not args.skip_manifest:
        v.section("manifest verification", lambda: verify_manifest(v))
    failures = [row for row in v.checks if not row["passed"]]
    result = {"passed": not failures, "run_dir": str(v.root), "manifest_checked": not args.skip_manifest,
              "raw_dataset_checked": args.dataset_dir is not None, "checks_total": len(v.checks),
              "checks_failed": len(failures), "failures": failures, "checks": v.checks}
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
