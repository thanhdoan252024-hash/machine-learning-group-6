import argparse,json
from case_study.experiments.run_phase3_studies import _regression_screen_data,_evaluate_regression_screen,OUTPUT_ROOT
VARIANTS={
 'selected':{},'no_goss':{'use_goss':False},'no_efb':{'use_efb':False},
 'no_regularization':{'reg_alpha':0.0,'reg_lambda':0.0},'feature_fraction_08':{'feature_fraction':0.8},
 'feature_fraction_10':{'feature_fraction':1.0},
}
a=argparse.ArgumentParser(); a.add_argument('variant',choices=VARIANTS); args=a.parse_args()
sel=json.loads((OUTPUT_ROOT/'regression_selection.json').read_text())['selected_params']
params=dict(sel); params.update(VARIANTS[args.variant]); screen,_=_regression_screen_data()
r=_evaluate_regression_screen(args.variant,'ablation',args.variant,params,screen)
p=OUTPUT_ROOT/'regression_ablation_checkpoint.json'; d=json.loads(p.read_text()) if p.exists() else {}; d[args.variant]=r; p.write_text(json.dumps(d,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
