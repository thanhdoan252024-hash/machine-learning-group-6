from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'case_study' / 'artifacts' / 'phase3'
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180, bbox_inches='tight')
    plt.close(fig)

reg_hp = pd.read_csv(ROOT/'regression_hyperparameter_study.csv').sort_values('validation_rmse')
fig, ax = plt.subplots(figsize=(9,5))
show = reg_hp.head(8).iloc[::-1]
ax.barh(show['name'], show['validation_rmse'])
ax.set_xlabel('Validation RMSE (lower is better)')
ax.set_title('Regression controlled hyperparameter screening')
save(fig, 'regression_hyperparameter_screening.png')

cls_hp = pd.read_csv(ROOT/'classification_hyperparameter_study.csv').sort_values('validation_pr_auc', ascending=False)
fig, ax = plt.subplots(figsize=(9,6))
show = cls_hp.head(10).iloc[::-1]
ax.barh(show['name'], show['validation_pr_auc'])
ax.set_xlabel('Validation PR-AUC / Average Precision (higher is better)')
ax.set_title('Classification controlled hyperparameter screening')
save(fig, 'classification_hyperparameter_screening.png')

reg_ab = pd.read_csv(ROOT/'regression_ablation.csv').sort_values('validation_rmse')
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(reg_ab['name'], reg_ab['validation_rmse'])
ax.set_xlabel('Validation RMSE')
ax.set_title('Regression ablation study')
save(fig, 'regression_ablation.png')

cls_ab = pd.read_csv(ROOT/'classification_ablation.csv').sort_values('validation_pr_auc')
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(cls_ab['name'], cls_ab['validation_pr_auc'])
ax.set_xlabel('Validation PR-AUC / Average Precision')
ax.set_title('Classification ablation study')
save(fig, 'classification_ablation.png')

reg_ref = pd.read_csv(ROOT/'regression_reference_comparison.csv').sort_values('rmse', ascending=False)
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(reg_ref['model'], reg_ref['rmse'])
ax.set_xlabel('Test RMSE (lower is better)')
ax.set_title('Regression reference-model comparison')
save(fig, 'regression_reference_comparison.png')

cls_ref = pd.read_csv(ROOT/'classification_reference_comparison.csv').sort_values('pr_auc_average_precision')
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(cls_ref['model'], cls_ref['pr_auc_average_precision'])
ax.set_xlabel('Test PR-AUC / Average Precision (higher is better)')
ax.set_title('Classification reference-model comparison')
save(fig, 'classification_reference_comparison.png')

efb = json.loads((ROOT/'efb_sparse_microbenchmark.json').read_text())
fig, ax = plt.subplots(figsize=(7,5))
ax.bar(['Direct histogram', 'EFB histogram'], [efb['direct_histogram_seconds_median'], efb['efb_histogram_seconds_median']])
ax.set_ylabel('Median histogram construction time (s)')
ax.set_title(f"EFB sparse microbenchmark: {efb['n_features']} features → {efb['n_bundles']} bundle")
save(fig, 'efb_sparse_microbenchmark.png')

reg_p2 = json.loads((ROOT.parent/'regression'/'metrics.json').read_text())['metrics']['test']
reg_p3 = json.loads((ROOT/'regression_final.json').read_text())['metrics']['test']
fig, ax = plt.subplots(figsize=(7,5))
ax.bar(['Phase 2', 'Phase 3'], [reg_p2['rmse'], reg_p3['rmse']])
ax.set_ylabel('Test RMSE')
ax.set_title('Regression improvement from Phase 2 to Phase 3')
save(fig, 'regression_phase2_vs_phase3.png')

cls_p2 = json.loads((ROOT.parent/'classification'/'metrics.json').read_text())['metrics']['test']
cls_p3 = json.loads((ROOT/'classification_final.json').read_text())['metrics']['test']
fig, ax = plt.subplots(figsize=(7,5))
ax.bar(['Phase 2', 'Phase 3'], [cls_p2['pr_auc_average_precision'], cls_p3['pr_auc_average_precision']])
ax.set_ylabel('Test PR-AUC / Average Precision')
ax.set_title('Classification improvement from Phase 2 to Phase 3')
save(fig, 'classification_phase2_vs_phase3.png')

print(f'Generated {len(list(FIG.glob("*.png")))} Phase 3 figures in {FIG}')
