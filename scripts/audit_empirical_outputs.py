from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tsne_from_scratch import squared_euclidean_distances
from src.evaluation import neighbor_order_and_ranks, neighborhood_quality, knn_indices_from_distances, neighborhood_overlap_score

OUT = ROOT/'outputs'
TABLES = OUT/'tables'
EMB = OUT/'embeddings'

# Data contract
raw = pd.read_csv(ROOT/'data/processed/optdigits_clean.csv')
assert raw.shape == (5620,65)
assert not any(str(c).startswith('Unnamed') for c in raw.columns)
features = [c for c in raw.columns if c.startswith('Pixel_')]
assert len(features)==64
X = raw[features].to_numpy(np.float64)
y = raw['label'].to_numpy(np.int64)
assert np.isfinite(X).all() and set(np.unique(y)) == set(range(10))

# P09
pgrid=[5,10,20,30,40,50]
p09=pd.read_csv(TABLES/'perplexity_screening_summary.csv')
assert p09['Perplexity'].tolist()==pgrid
assert np.all(np.isfinite(p09.select_dtypes(include=[np.number]).to_numpy()))
assert np.all(p09['Best KL'] < p09['Initial KL'])
assert np.all(p09['Final Restored KL'].to_numpy() == p09['Best KL'].to_numpy())
idx_df=pd.read_csv(TABLES/'optdigits_screening_indices.csv')
idx=idx_df['original_index'].to_numpy(np.int64)
assert len(idx)==1000 and len(np.unique(idx))==1000
X_screen=X[idx]
y_screen=y[idx]
assert np.array_equal(y_screen,idx_df['label'].to_numpy(np.int64))

# Recompute P10 independently from stored embeddings
D_high=squared_euclidean_distances(X_screen)
high_order,high_ranks=neighbor_order_and_ranks(D_high)
p10_saved=pd.read_csv(TABLES/'perplexity_neighborhood_quality.csv').set_index('Perplexity')
max_p10_err=0.0
for p in pgrid:
    e=pd.read_csv(EMB/f'optdigits_tsne_perplexity_{p}.csv')
    assert e.shape==(1000,4)
    assert np.array_equal(e['original_index'].to_numpy(np.int64),idx)
    assert np.array_equal(e['label'].to_numpy(np.int64),y_screen)
    Y=e[['TSNE_1','TSNE_2']].to_numpy(np.float64)
    assert np.isfinite(Y).all()
    m=neighborhood_quality(Y,high_order,high_ranks,k_values=(5,10,20,50))
    for k,v in m.items():
        err=abs(v-float(p10_saved.loc[p,k])); max_p10_err=max(max_p10_err,err)
        assert err < 1e-12, (p,k,v,p10_saved.loc[p,k],err)

# P11 per-seed quality and pairwise stability
p11q=pd.read_csv(TABLES/'p11_multiseed_quality.csv')
p11s=pd.read_csv(TABLES/'p11_pairwise_seed_stability.csv')
candidates=[20,30,40]; seeds=[0,42,123]; ks=[5,10,20,50]
assert len(p11q)==len(candidates)*len(seeds)
max_p11_quality_err=0.0
for p in candidates:
    for seed in seeds:
        e=pd.read_csv(EMB/f'p11_p{p}_seed{seed}.csv')
        Y=e[['TSNE_1','TSNE_2']].to_numpy(np.float64)
        m=neighborhood_quality(Y,high_order,high_ranks,k_values=ks)
        row=p11q[(p11q['Perplexity']==p)&(p11q['Seed']==seed)].iloc[0]
        for col,v in m.items():
            err=abs(v-float(row[col])); max_p11_quality_err=max(max_p11_quality_err,err)
            assert err < 1e-12, (p,seed,col,v,row[col],err)

max_stab_err=0.0
for p in candidates:
    Ys={}
    for seed in seeds:
        e=pd.read_csv(EMB/f'p11_p{p}_seed{seed}.csv')
        Ys[seed]=e[['TSNE_1','TSNE_2']].to_numpy(np.float64)
    for i,a in enumerate(seeds):
        for b in seeds[i+1:]:
            for k in ks:
                na=knn_indices_from_distances(squared_euclidean_distances(Ys[a]),k)
                nb=knn_indices_from_distances(squared_euclidean_distances(Ys[b]),k)
                mean,_=neighborhood_overlap_score(na,nb)
                row=p11s[(p11s['Perplexity']==p)&(p11s['Seed A']==a)&(p11s['Seed B']==b)&(p11s['k']==k)].iloc[0]
                err=abs(mean-float(row['Mean kNN Overlap'])); max_stab_err=max(max_stab_err,err)
                assert err < 1e-12, (p,a,b,k,mean,row['Mean kNN Overlap'],err)

# Selection contract
sel=pd.read_csv(TABLES/'p11_primary_alternative_selection.csv')
primary=int(sel.loc[sel['Role']=='PRIMARY','Perplexity'].iloc[0])
alternative=int(sel.loc[sel['Role']=='ALTERNATIVE','Perplexity'].iloc[0])
assert (primary,alternative)==(40,30)

# P12 final artifact sanity
summary=pd.read_csv(TABLES/f'final_run_summary_p{primary}.csv').iloc[0]
conf=pd.read_csv(TABLES/f'final_configuration_p{primary}.csv')
hist=pd.read_csv(TABLES/f'final_optimization_history_p{primary}.csv')
final=pd.read_csv(EMB/f'final_optdigits_embedding_p{primary}.csv')
assert final.shape==(5620,4)
assert final.columns.tolist()==['original_index','TSNE_1','TSNE_2','label']
assert np.array_equal(final['original_index'].to_numpy(np.int64),np.arange(5620))
assert np.array_equal(final['label'].to_numpy(np.int64),y)
Yf=final[['TSNE_1','TSNE_2']].to_numpy(np.float64)
assert np.isfinite(Yf).all() and np.all(Yf.std(axis=0)>0)
assert np.allclose(Yf.mean(axis=0),0.0,atol=1e-10)
assert int(summary['Samples'])==5620 and int(summary['Features'])==64 and int(summary['Perplexity'])==primary
expected_summary_cols={'Samples','Features','Perplexity','Initial KL','Best KL','Final Restored KL','Best Iteration','Actual Iterations','Stopped Early','Runtime Seconds','Optimizer Seconds','Achieved Perplexity Mean','Achieved Perplexity Min','Achieved Perplexity Max','Achieved Perplexity Max Error'}
assert expected_summary_cols.issubset(set(summary.index))
conf_map=dict(zip(conf['Parameter'].astype(str),conf['Value'].astype(str)))
for key in ['Samples','Original dimensions','Embedding dimensions','Perplexity','Learning rate','Max iterations','Early exaggeration','Early exaggeration iterations','Initial momentum','Final momentum','Random state','Implementation']:
    assert key in conf_map, key
assert float(summary['Best KL']) < float(summary['Initial KL'])
assert math.isclose(float(summary['Best KL']),float(summary['Final Restored KL']),rel_tol=1e-12,abs_tol=1e-12)
assert int(summary['Actual Iterations'])==len(hist)==1000
assert hist['Iteration'].iloc[0]==1 and hist['Iteration'].iloc[-1]==1000
assert np.isfinite(hist[['KL','Gradient Norm']].to_numpy()).all()
assert (OUT/'figures'/f'final_optdigits_tsne_p{primary}.png').stat().st_size>1000
assert (OUT/'figures'/f'final_kl_convergence_p{primary}.png').stat().st_size>1000
assert (OUT/'figures'/f'final_gradient_norm_p{primary}.png').stat().st_size>1000
manifest=json.loads((OUT/'reproducibility_manifest.json').read_text())
assert manifest['primary_perplexity']==primary and manifest['alternative_perplexity']==alternative
assert math.isclose(float(manifest['final_kl']),float(summary['Final Restored KL']),rel_tol=1e-12)

# Recompute P12 full-data neighborhood quality directly from the final 5,620-sample embedding.
D_full=squared_euclidean_distances(X)
full_order,full_ranks=neighbor_order_and_ranks(D_full)
full_metrics=neighborhood_quality(Yf,full_order,full_ranks,k_values=(5,10,20,50))
full_saved=pd.read_csv(TABLES/f'final_neighborhood_quality_p{primary}.csv').set_index('k')
max_p12_full_quality_err=0.0
for k in [5,10,20,50]:
    for col,prefix in [('Trustworthiness','T'),('Continuity','C')]:
        v=float(full_metrics[f'{prefix}@{k}'])
        err=abs(v-float(full_saved.loc[k,col])); max_p12_full_quality_err=max(max_p12_full_quality_err,err)
        assert err < 1e-12, (k,col,v,full_saved.loc[k,col],err)
full_summary=pd.read_csv(TABLES/f'final_neighborhood_quality_summary_p{primary}.csv').iloc[0]
assert int(full_summary['Samples'])==5620 and int(full_summary['Perplexity'])==primary
assert math.isclose(float(full_summary['Worst Neighborhood Score']),min(full_metrics.values()),rel_tol=1e-12,abs_tol=1e-12)

report={
    'data_contract':'PASS',
    'P09':'PASS',
    'P10':'PASS',
    'P11':'PASS',
    'P12_artifact_sanity':'PASS',
    'P12_full_neighborhood_quality':'PASS',
    'primary_perplexity':primary,
    'alternative_perplexity':alternative,
    'max_p10_recompute_abs_error':max_p10_err,
    'max_p11_quality_recompute_abs_error':max_p11_quality_err,
    'max_p11_stability_recompute_abs_error':max_stab_err,
    'max_p12_full_quality_recompute_abs_error':max_p12_full_quality_err,
    'final_neighborhood_quality':{k:float(v) for k,v in full_metrics.items()},
    'final_embedding_shape':[5620,2],
    'final_kl':float(summary['Final Restored KL']),
    'full_runtime_seconds':float(summary['Runtime Seconds']),
}
(OUT/'empirical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print('EMPIRICAL OUTPUT AUDIT: PASS')
