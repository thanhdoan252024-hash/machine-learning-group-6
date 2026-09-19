from __future__ import annotations
import argparse, json
from pathlib import Path
from case_study.experiments.run_phase3_studies import _regression_screen_data,_evaluate_regression_screen,_regression_base,OUTPUT_ROOT

CONFIGS={
 'baseline': ('baseline','baseline',{}),
 'num_leaves_15': ('num_leaves',15,{'num_leaves':15}),
 'learning_rate_005': ('learning_rate',0.05,{'learning_rate':0.05}),
 'max_bins_32': ('max_bins',32,{'max_bins':32}),
 'min_data_50': ('min_data_in_leaf',50,{'min_data_in_leaf':50}),
 'reg_lambda_5': ('reg_lambda',5.0,{'reg_lambda':5.0}),
 'feature_fraction_08': ('feature_fraction',0.8,{'feature_fraction':0.8}),
}
parser=argparse.ArgumentParser(); parser.add_argument('config',choices=CONFIGS); args=parser.parse_args()
family,value,changes=CONFIGS[args.config]
screen,_=_regression_screen_data(); params=_regression_base(); params.update(changes)
row=_evaluate_regression_screen(args.config,family,value,params,screen)
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); path=OUTPUT_ROOT/'regression_screen_checkpoint.json'
current=json.loads(path.read_text()) if path.exists() else {}
current[args.config]=row
path.write_text(json.dumps(current,indent=2),encoding='utf-8')
print(json.dumps(row,indent=2))
