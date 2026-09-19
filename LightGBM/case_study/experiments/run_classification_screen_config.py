import argparse,json
from case_study.experiments.run_phase3_studies import _classification_data,_classification_base,_evaluate_classification_validation,OUTPUT_ROOT
CONFIGS={
 'baseline':('baseline','baseline',{}),
 'scale_pos_2':('scale_pos_weight',2.0,{'scale_pos_weight':2.0}),
 'scale_pos_3':('scale_pos_weight',3.0,{'scale_pos_weight':3.0}),
 'scale_pos_5':('scale_pos_weight',5.0,{'scale_pos_weight':5.0}),
 'leaves_7':('num_leaves',7,{'num_leaves':7}),
 'leaves_31':('num_leaves',31,{'num_leaves':31}),
 'learning_rate_01':('learning_rate',0.10,{'learning_rate':0.10}),
 'max_bins_31':('max_bins',31,{'max_bins':31}),
 'max_bins_127':('max_bins',127,{'max_bins':127}),
 'min_child_50':('min_child_samples',50,{'min_child_samples':50}),
 'reg_lambda_5':('reg_lambda',5.0,{'reg_lambda':5.0}),
 'feature_fraction_085':('feature_fraction',0.85,{'feature_fraction':0.85}),
}
a=argparse.ArgumentParser(); a.add_argument('config',choices=CONFIGS); args=a.parse_args()
family,value,chg=CONFIGS[args.config]; p=_classification_base(); p.update(chg); D=_classification_data()
r=_evaluate_classification_validation(args.config,family,value,p,D)
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True); f=OUTPUT_ROOT/'classification_screen_checkpoint.json'; d=json.loads(f.read_text()) if f.exists() else {}; d[args.config]=r; f.write_text(json.dumps(d,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
