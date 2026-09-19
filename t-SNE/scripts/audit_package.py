from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
required=[
 '00_README_START_HERE.md','README.md','README_RUN_ON_NEW_MACHINE.md','01_RESEARCH_IDEA_AND_SCOPE.md',
 '02_TSNE_THEORY_KNOWLEDGE.md','02_PROMPT_CODE_MAPPING.md','03_IMPLEMENTATION_ROADMAP.md','04_STAGE_GATES_AND_QA.md',
 '05_AI_PROMPTING_LOG_MASTER.md','06_ENVIRONMENT_AND_DATA_GUIDE.md','08_REPRODUCIBILITY_AND_HANDOFF.md',
 '09_EVALUATION_AND_SELECTION_PROTOCOL.md','10_REFERENCES.md','11_RUN_ORDER_QUICK_GUIDE.md','12_RESULTS_AND_EVIDENCE_GUIDE.md',
 '13_LEGACY_PROJECT_AUDIT.md','14_CODE_API_REFERENCE.md','15_REPRODUCTION_PROFILES.md','16_FEATURE_COMPLETENESS_MATRIX.md',
 '17_FINAL_CASE_STUDY_AUDIT.md','18_EXECUTION_CHECKLIST.md','19_EXPECTED_ARTIFACTS_CONTRACT.md','FINAL_RESULTS_SUMMARY.md',
 'FINAL_TSNE_REPORT.md','PACKAGE_VALIDATION_REPORT.md','requirements.txt','requirements-lock.txt','requirements-notebook.txt',
 'project_manifest.json','package_validation.json','report/Chuong_Case_Study_Thuc_nghiem_tSNE_UCI_Optdigits_FINAL.docx','notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb',
 'notebook/TSNE_From_Scratch_Optdigits_FINAL_RESULTS.ipynb','data/processed/optdigits_clean.csv',
 'src/tsne_from_scratch.py','src/evaluation.py','scripts/run_case_study.py','scripts/run_notebook.py','scripts/audit_rerun_outputs.py',
 'outputs/tables/perplexity_screening_summary.csv','outputs/tables/perplexity_neighborhood_quality.csv',
 'outputs/tables/p11_multiseed_quality.csv','outputs/tables/p11_pairwise_seed_stability.csv',
 'outputs/tables/p11_primary_alternative_selection.csv','outputs/tables/final_run_summary_p40.csv',
    'outputs/tables/final_configuration_p40.csv','outputs/tables/final_neighborhood_quality_p40.csv',
    'outputs/tables/final_neighborhood_quality_summary_p40.csv',
 'outputs/embeddings/final_optdigits_embedding_p40.csv','outputs/figures/final_optdigits_tsne_p40.png',
 'outputs/reproducibility_manifest.json','outputs/empirical_audit.json','outputs/p12_independent_kl_audit.json'
]
missing=[p for p in required if not (ROOT/p).exists()]
assert not missing, f'Missing: {missing}'
# Prompt contract P00-P13
prompt_files=sorted((ROOT/'prompts').glob('P*.md'))
assert len(prompt_files)==14, f'Expected 14 prompt files, got {len(prompt_files)}'
for i in range(14):
    assert any(p.name.startswith(f'P{i:02d}_') for p in prompt_files), f'Missing P{i:02d} prompt'
# Data contract
df=pd.read_csv(ROOT/'data/processed/optdigits_clean.csv')
features=[f'Pixel_{i}' for i in range(1,65)]
assert df.shape==(5620,65)
assert df.columns.tolist()==features+['label']
assert not any(str(c).startswith('Unnamed') for c in df.columns)
# Runtime/final-status contract
status=json.loads((ROOT/'outputs/run_status.json').read_text(encoding='utf-8'))
assert status.get('P09')=='PASS' and status.get('P10')=='PASS' and status.get('P11')=='PASS' and status.get('P12')=='PASS'
assert int(status.get('primary_perplexity'))==40 and int(status.get('alternative_perplexity'))==30
validation=json.loads((ROOT/'package_validation.json').read_text(encoding='utf-8'))
assert validation.get('submission_status')=='READY FOR SUBMISSION'
print('PACKAGE AUDIT: PASS')
