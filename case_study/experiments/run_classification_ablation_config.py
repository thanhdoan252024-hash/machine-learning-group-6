import argparse,json
from case_study.experiments.run_phase3_studies import _classification_data,_evaluate_classification_validation,OUTPUT_ROOT
VARIANTS={'selected':{},'no_goss':{'use_goss':False},'no_efb':{'use_efb':False},'no_regularization':{'reg_alpha':0.0,'reg_lambda':0.0},'feature_fraction_085':{'feature_fraction':0.85},'positive_weight_5':{'scale_pos_weight':5.0}}
a=argparse.ArgumentParser();a.add_argument('variant',choices=VARIANTS);args=a.parse_args()
sel=json.loads((OUTPUT_ROOT/'classification_selection.json').read_text())['selected_params'];p=dict(sel);p.update(VARIANTS[args.variant]);D=_classification_data();r=_evaluate_classification_validation(args.variant,'ablation',args.variant,p,D)
f=OUTPUT_ROOT/'classification_ablation_checkpoint.json';d=json.loads(f.read_text()) if f.exists() else {};d[args.variant]=r;f.write_text(json.dumps(d,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
