"""Execute the project notebook and retain a self-contained evidence bundle."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebook/PCA_From_Scratch_UCI_HAR_Phase1_11_With_AI_Prompting_Log.ipynb'
if not NOTEBOOK.exists():
    NOTEBOOK = ROOT / 'notebook.source.ipynb'
PACKAGES = ['numpy', 'pandas', 'matplotlib', 'nbformat', 'nbclient', 'nbconvert',
            'ipykernel', 'jupyter-client', 'ipython', 'psutil']
PHASES = [
    (1, 'Dataset preparation', 'P01', [5, 6, 7], 'CP1'),
    (2, 'Standardization', 'P02', [10, 11], 'CP2'),
    (3, 'PCA from scratch', 'P03', [14], None),
    (4, 'Mathematical and API validation', 'P04/P05', [17, 19], 'CP3'),
    (5, 'Explained variance', 'P06', [23, 24, 25, 26], 'CP4'),
    (6, 'Visualization', 'P07', [29, 30, 31], 'CP5'),
    (7, 'Reconstruction', 'P08', [34, 35, 36], 'CP6'),
    (8, 'Loadings', 'P09', [39, 40], 'CP7'),
    (9, 'Candidate selection', 'P10', [44], 'CP8'),
    (10, 'Final datasets', 'P11', [47, 48, 49], 'CP9'),
    (11, 'Export and reload validation', 'P12', [52], 'CP10'),
]
ACCEPTANCE = {
    1: {'features': 561, 'nonfinite_count': 0, 'labels': 'row counts match; labels in activity metadata'},
    2: {'max_abs_mean_lt': 1e-10, 'max_std_error_lt': 1e-10, 'nonfinite_count': 0},
    3: {'symmetry_error_lt': 1e-8, 'min_eigenvalue_gte': -1e-8, 'orthogonality_error_lt': 1e-8,
        'abs_evr_sum_minus_one_lt': 1e-8, 'score_off_diagonal_lt': 1e-7, 'score_diagonal_error_lt': 1e-7},
    4: {'cumulative_diff_gte': -1e-12, 'abs_final_cumulative_minus_one_lt': 1e-10,
        'threshold_k': 'nondecreasing for 0.8, 0.9, 0.95, 0.99'},
    5: {'shape': [7352, 3], 'pc_variance_order_tolerance': 1e-10, 'figures': 'three 2D views and one 3D view'},
    6: {'mse_diff_lte': 1e-10, 'full_train_mse_lt': 1e-20, 'mse': 'finite on train and test'},
    7: {'abs_loading_norm_squared_minus_one_lt': 1e-10},
    8: {'retained_variance_pca90_gte': 0.90, 'retained_variance_pca95_gte': 0.95, 'k90_lte_k95': True},
    9: {'nested_train_error_lt': 1e-10, 'nested_test_error_lt': 1e-10, 'pc_variance_diff_lte': 1e-8,
        'shape': 'same rows as labels; columns k90/k95; all finite'},
    10: {'files': 'all required files exist and nonempty', 'disk_arrays': 'exact match, including dtype',
         'projection_atol': 1e-10, 'projection_rtol': 0.0, 'figures': '13 PNG signatures checked'},
}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def file_record(path, base):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': path.relative_to(base).as_posix(), 'size_bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def cell_text(cell):
    parts = []
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            parts.append(output.text)
        elif 'text/plain' in output.get('data', {}):
            parts.append(output.data['text/plain'] + '\n')
        elif output.output_type == 'error':
            parts.append('\n'.join(output.get('traceback', [])))
    return ''.join(parts)


def run_verification(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Artifact verification failed: {result.stdout}\n{result.stderr}')
    report = json.loads(result.stdout)
    print(f"Artifact verification: {report['checks_total']} checks PASS; manifest={report['manifest_checked']}", flush=True)


def update_run_index(run_dir, status):
    runs_dir = ROOT / 'runs'
    runs_dir.mkdir(exist_ok=True)
    relative = os.path.relpath(run_dir, ROOT).replace(os.sep, '/')
    write_json(runs_dir / 'latest.json', {'run_dir': relative, 'status': 'completed',
                                        'completed_at_utc': status['completed_at_utc']})
    run_link = os.path.relpath(run_dir, runs_dir).replace(os.sep, '/')
    lines = ['# Kết quả chạy PCA', '', f'Bộ kết quả mới nhất: [{run_dir.name}]({run_link}/README.md).', '',
             f'- [Notebook có đầy đủ output]({run_link}/notebook.executed.ipynb)',
             f'- [Xem notebook bằng trình duyệt]({run_link}/notebook.html)',
             f'- [Bảng kết quả và kiểm chứng]({run_link}/reports/RESULTS.md)',
             f'- [Báo cáo PCA]({run_link}/reports/FINAL_PCA_REPORT.md)',
             f'- [13 biểu đồ PNG]({run_link}/figures/)',
             f'- [Dữ liệu NPY/NPZ và bảng CSV]({run_link}/outputs/)',
             f'- [Kết quả kiểm tra artifacts và SHA-256]({run_link}/verification.json)', '',
             'Mỗi lần chạy được giữ riêng. Tên thư mục sử dụng thời gian UTC; các đường dẫn trong environment/log là vị trí tại thời điểm chạy.', '',
             '| Lần chạy | Trạng thái |', '|---|---|']
    for path in sorted(runs_dir.glob('run_*'), reverse=True):
        state_path = path / 'execution_status.json'
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            label = f'[{path.name}]({path.name}/README.md)' if (path / 'README.md').exists() else path.name
            lines.append(f"| {label} | {state['status']} |")
    (runs_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest_path = ROOT / 'project_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest.update(latest_verified_run=relative, status='PCA complete',
                        source_notebook=os.path.relpath(NOTEBOOK, ROOT).replace(os.sep, '/'),
                        phases=[{'phase': phase, 'name': name, 'prompt': prompt, 'checkpoint': checkpoint,
                                 'status': 'complete'} for phase, name, prompt, indices, checkpoint in PHASES])
        write_json(manifest_path, manifest)


def markdown_table(frame):
    """Keep reports independent of optional tabulate packages."""
    def render(value):
        if isinstance(value, float):
            return f'{value:.8g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines.extend('| ' + ' | '.join(map(render, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def write_reports(run_dir, nb, environment):
    import pandas as pd
    metrics = json.loads((run_dir / 'evidence/run_metrics.json').read_text(encoding='utf-8'))
    checkpoints = []
    phase_lines = ['# Phase reports', '', f"Run UTC: {environment['started_at_utc']}", '',
                   'Inputs: official UCI HAR split; source hashes in dataset_manifest.json.', '',
                   'Scaler and PCA fit train only. Labels excluded from fit. No sklearn PCA/scaler.', '']
    metric_keys = {1: 'dataset', 2: 'standardization', 3: 'pca_math', 4: 'thresholds',
                   5: 'final_validation', 6: 'reconstruction', 7: 'loading_norms',
                   8: 'selection', 9: 'final_validation', 10: 'export_validation'}
    for phase, name, prompt, indices, checkpoint in PHASES:
        content = '\n'.join(f'CELL {i}\n{cell_text(nb.cells[i])}' for i in indices)
        log_name = f'PHASE_{phase:02d}.txt'
        (run_dir / 'evidence' / log_name).write_text(content, encoding='utf-8')
        phase_lines += [f'## Phase {phase}: {name}', '', f'Prompt: {prompt}. Cells: {indices}.', '',
                        f'Output and actual metrics: [execution log](../evidence/{log_name}).', '',
                        'Acceptance conditions: assertions in the corresponding cells of notebook.source.ipynb.', '']
        if checkpoint:
            import re
            number = int(checkpoint[2:])
            passed = bool(re.search(rf'CHECKPOINT\s+{number}\b[^\n]*PASS', content))
            if not passed:
                raise RuntimeError(f'{checkpoint} PASS output missing')
            record = {'checkpoint': checkpoint, 'phase': phase, 'prompt': prompt, 'status': 'PASS',
                      'source_cells_zero_based': indices, 'log': f'evidence/{log_name}',
                      'metrics': metrics[metric_keys[number]],
                      'acceptance_conditions': ACCEPTANCE[number]}
            if number == 5:
                spectrum = pd.read_csv(run_dir / 'outputs/CP04_eigenvalue_spectrum.csv')
                record['metrics'] = {'score_shape': [metrics['dataset']['shapes']['X_train'][0], 3],
                                     'train_pc1_pc3_variance': spectrum['Eigenvalue'].head(3).tolist(),
                                     'actual_score_variance_log': f'evidence/{log_name}'}
            checkpoints.append(record)
            phase_lines += [f'{checkpoint}: **PASS**; numeric evidence in `evidence/checkpoints.json`.', '']
        else:
            phase_lines += ['PCA fit completed; mathematical acceptance is evaluated in Phase 4 / CP3.', '']
    write_json(run_dir / 'evidence/checkpoints.json', checkpoints)
    reports = run_dir / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'PHASE_REPORTS.md').write_text('\n'.join(phase_lines), encoding='utf-8')
    candidates = pd.read_csv(run_dir / 'outputs/pca_candidate_comparison.csv')
    summary = pd.read_csv(run_dir / 'outputs/pca_final_summary.csv')
    reconstruction = pd.read_csv(run_dir / 'outputs/pca_reconstruction_analysis.csv')
    results = '\n'.join([
        '# PCA results', '', f"Executed: {environment['started_at_utc']}", '',
        'These values were generated by execution of notebook.source.ipynb, not copied from the earlier PDF.', '',
        '## Final representations', '', markdown_table(summary), '',
        '## Candidate comparison', '', markdown_table(candidates), '',
        'Retained variance refers to standardized training data. Reconstruction MSE is in standardized feature space.', '',
        '## Reconstruction curve data', '', markdown_table(reconstruction), '',
        '## Numerical validation', '', '```json',
        json.dumps({key: metrics[key] for key in ['dataset', 'standardization', 'pca_math', 'loading_norms', 'final_validation']},
                   indent=2, ensure_ascii=False), '```', '',
        'All CP1–CP10 passed. See PHASE_REPORTS.md and ../evidence/checkpoints.json for inputs, prompts and logs.', '',
    ])
    (reports / 'RESULTS.md').write_text(results, encoding='utf-8')
    final = '\n'.join([
        '# Final PCA report', '',
        'Objective: implement PCA with NumPy on UCI HAR and quantify dimensionality reduction and reconstruction loss.', '',
        '## Data and method', '',
        'The official train/test split is retained. Standardization uses train mean and population standard deviation.',
        'The centered train covariance uses n−1. NumPy eigh yields the orthonormal eigenbasis, sorted by decreasing eigenvalue.',
        'Train-fitted eigenvectors transform both train and test. Labels are used for audit and visualization only.', '',
        '## Results', '', markdown_table(summary), '', markdown_table(candidates), '',
        '## Validation and interpretation', '',
        'CP1–CP10 and the transform/inverse-transform functional test passed. All 13 figures are saved under ../figures/.',
        'PCA90 is the compression-oriented candidate; PCA95 is the primary variance-preserving candidate.',
        'Explained variance and training reconstruction error are related views of the same approximation objective.',
        'Low-dimensional scatter overlap does not, by itself, establish classification performance.', '',
        '## Scope and limitations', '',
        'Variance retained is not classification accuracy. Classification and runtime comparisons have not been performed in this run.',
        'The PCA class assumes input centered by the preceding scaler. Inverse transform reconstructs standardized features.',
        'Eigenvector signs and tiny numerical residuals can differ across BLAS/LAPACK implementations.', '',
        '## Reproducibility', '',
        'The executed notebook, HTML, input hashes, environment, locked package versions and output hashes accompany this report.',
        'See ../README.md for rerun and independent artifact verification commands.', '',
    ])
    (reports / 'FINAL_PCA_REPORT.md').write_text(final, encoding='utf-8')
    (reports / 'ENVIRONMENT_RECORD.md').write_text(
        '# Environment record\n\n```json\n' + json.dumps(environment, indent=2, ensure_ascii=False) + '\n```\n\n'
        'NumPy BLAS/LAPACK configuration: ../evidence/numpy_config.txt\n\n'
        'Dataset file hashes and provenance: ../dataset_manifest.json\n', encoding='utf-8')
    (reports / 'DECISION_LOG.md').write_text(
        '# Decisions\n\n- Preserve original PCA mathematics and official split.\n'
        '- Derive k from training cumulative variance; keep PCA95 primary and PCA90 alternative.\n'
        '- Export all plots and evaluation tables, with numeric checkpoint evidence.\n'
        '- Validate saved arrays by reloading, including projection with persisted parameters.\n'
        '- Preserve each execution in a separate run directory.\n', encoding='utf-8')
    (reports / 'ISSUE_LOG.md').write_text(
        '# Execution issue log\n\nNo notebook execution error was present in this completed run.\n\n'
        'Classification remains outside this run. See execution_status.json for completion status and verification.json for artifact checks.\n',
        encoding='utf-8')
    (reports / 'AGENT_HANDOFF.md').write_text(
        '# PCA handoff\n\nPhases 1–11 completed; CP1–CP10 PASS.\n\n'
        'Use ../outputs/X_train_scaled.npy and X_test_scaled.npy for the baseline, '
        'X_*_pca90.npy / X_*_pca95.npy for reduced representations, and y_train.npy / y_test.npy for labels.\n\n'
        'Persisted mean/scale and eigenbasis are in pca_from_scratch_parameters.npz. Load with allow_pickle=False.\n\n'
        'Keep the official held-out test split. These final artifacts fit preprocessing on all training data. '
        'If performing cross-validation within train, refit preprocessing on each training fold to avoid validation leakage.\n\n'
        'Independent consumer checks: ../verification.json. File integrity: ../artifact_manifest.json.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/raw/UCI HAR Dataset')
    parser.add_argument('--run-dir', type=Path, help='New directory; existing directories are never overwritten.')
    parser.add_argument('--timeout', type=int, default=1200, help='Timeout per code cell in seconds.')
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    input_names = ['train/X_train.txt', 'train/y_train.txt', 'test/X_test.txt', 'test/y_test.txt',
                   'features.txt', 'activity_labels.txt']
    for name in input_names:
        if not (dataset / name).is_file():
            parser.error(f'Missing dataset input: {dataset / name}')
    stamp = datetime.now(timezone.utc)
    run_dir = (args.run_dir or ROOT / 'runs' / stamp.strftime('run_%Y%m%dT%H%M%SZ')).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'evidence').mkdir()
    runtime = ROOT / '.runtime'
    env_paths = {'MPLCONFIGDIR': runtime / 'matplotlib', 'IPYTHONDIR': runtime / 'ipython',
                 'JUPYTER_RUNTIME_DIR': runtime / 'jupyter_runtime', 'JUPYTER_CONFIG_DIR': runtime / 'jupyter_config'}
    for key, path in env_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(PCA_RUN_DIR=str(run_dir), PCA_DATA_ROOT=str(dataset.parent),
                      PYTHONIOENCODING='utf-8', MPLBACKEND='module://matplotlib_inline.backend_inline',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    import nbformat
    import numpy as np
    import psutil
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    print(f'RUN_DIR={run_dir}', flush=True)
    started = time.perf_counter()
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    environment = {'started_at_utc': stamp.isoformat(), 'python': sys.version, 'python_executable': sys.executable,
                   'platform': platform.platform(), 'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
                   'ram_bytes': psutil.virtual_memory().total, 'packages': versions,
                   'dataset_dir': str(dataset), 'run_dir': str(run_dir), 'runtime': 'local Jupyter kernel',
                   'thread_limits': {key: os.environ[key] for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']}}
    write_json(run_dir / 'environment.json', environment)
    (run_dir / 'requirements-lock.txt').write_text(''.join(f'{key}=={value}\n' for key, value in versions.items()), encoding='utf-8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (run_dir / 'evidence/pip-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    (run_dir / 'evidence/numpy_config.txt').write_text(config.getvalue(), encoding='utf-8')
    data_manifest = {'dataset': 'UCI Human Activity Recognition Using Smartphones',
                     'source_url': 'https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones',
                     'dataset_root': str(dataset), 'files': [file_record(dataset / name, dataset) for name in input_names]}
    provenance = ROOT / 'data/dataset_provenance.json'
    if provenance.exists():
        data_manifest['download_provenance'] = json.loads(provenance.read_text(encoding='utf-8-sig'))
    write_json(run_dir / 'dataset_manifest.json', data_manifest)
    shutil.copy2(NOTEBOOK, run_dir / 'notebook.source.ipynb')
    (run_dir / 'scripts').mkdir()
    for script in ['run_pca.py', 'verify_pca_artifacts.py']:
        shutil.copy2(ROOT / 'scripts' / script, run_dir / 'scripts' / script)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    kernel_root = runtime / 'kernels'
    kernel_dir = kernel_root / 'pca-run'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    write_json(kernel_dir / 'kernel.json', {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                          'display_name': 'PCA reproduction', 'language': 'python'})
    manager = KernelManager(kernel_name='pca-run', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    def on_complete(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'Completed code cell {cell_index}', flush=True)
    client = NotebookClient(nb, km=manager, timeout=args.timeout, allow_errors=False,
                            resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=on_complete)
    status = {'status': 'running', 'started_at_utc': stamp.isoformat()}
    write_json(run_dir / 'execution_status.json', status)
    try:
        try:
            client.execute(cwd=str(ROOT))
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(nb, run_dir / 'notebook.executed.ipynb')
        nbformat.validate(nb)
        html, _ = HTMLExporter(template_name='lab').from_notebook_node(nb)
        (run_dir / 'notebook.html').write_text(html, encoding='utf-8')
        write_reports(run_dir, nb, environment)
        (run_dir / 'execution.log').write_text('\n'.join(f'CELL {i}\n{cell_text(cell)}' for i, cell in enumerate(nb.cells)
                                                        if cell.cell_type == 'code'), encoding='utf-8')
        readme = '\n'.join([
            '# PCA reproduction run', '', f'UTC start: {stamp.isoformat()}', '',
            '- [Notebook with outputs](notebook.executed.ipynb)', '- [Browser-readable notebook](notebook.html)',
            '- [Results](reports/RESULTS.md)', '- [Final report](reports/FINAL_PCA_REPORT.md)',
            '- [Phase evidence](reports/PHASE_REPORTS.md)', '- [Exported arrays and tables](outputs/README.txt)',
            '- Figures: `figures/` (13 PNGs).', '- Integrity: `artifact_manifest.json` (SHA-256 and byte sizes).', '',
            '## Reproduce from the project root', '', '```powershell',
            'python -m pip install -r requirements-repro.txt',
            'python scripts/run_pca.py --dataset-dir "data/raw/UCI HAR Dataset"', '```', '',
            'Each execution creates a new run directory. The source snapshot in this bundle identifies the executed code.',
            'To rerun from this bundle itself, install requirements-lock.txt and supply the absolute dataset directory to scripts/run_pca.py.', '',
            '## Verify without fitting PCA again', '', '```powershell',
            f'python scripts/verify_pca_artifacts.py --run-dir "{run_dir.relative_to(ROOT).as_posix() if run_dir.is_relative_to(ROOT) else run_dir}" --dataset-dir "data/raw/UCI HAR Dataset"',
            '```', '', 'The dataset hashes identify the six raw inputs. Downloaded raw data is kept outside this run bundle.',
            'verification.json is excluded from the checksum manifest because it records verification of that manifest.', '',
        ])
        (run_dir / 'README.md').write_text(readme, encoding='utf-8')
        verify_command = [sys.executable, str(ROOT / 'scripts/verify_pca_artifacts.py'), '--run-dir', str(run_dir),
                          '--dataset-dir', str(dataset), '--skip-manifest', '--report', str(run_dir / 'verification.json')]
        run_verification(verify_command)
        status.update(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter() - started)
        write_json(run_dir / 'execution_status.json', status)
        files = [file_record(path, run_dir) for path in sorted(run_dir.rglob('*'))
                 if path.is_file() and path.name not in {'artifact_manifest.json', 'verification.json'}]
        write_json(run_dir / 'artifact_manifest.json', {'algorithm': 'SHA-256', 'files': files})
        run_verification([sys.executable, str(ROOT / 'scripts/verify_pca_artifacts.py'), '--run-dir', str(run_dir),
                          '--dataset-dir', str(dataset), '--report', str(run_dir / 'verification.json')])
        update_run_index(run_dir, status)
        print(f'COMPLETED: {run_dir}', flush=True)
    except BaseException as exc:
        nbformat.write(nb, run_dir / 'notebook.failed.ipynb')
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter() - started)
        write_json(run_dir / 'execution_status.json', status)
        raise


if __name__ == '__main__':
    main()
