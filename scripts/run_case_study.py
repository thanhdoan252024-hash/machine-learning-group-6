from __future__ import annotations
import argparse, gc, json, time, sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tsne_from_scratch import (
    squared_euclidean_distances,
    conditional_probability_matrix,
    symmetrize_probabilities,
    optimize_tsne_embedding,
    TSNEFromScratch,
)
from src.evaluation import (
    neighbor_order_and_ranks,
    neighborhood_quality,
    pareto_frontier,
    knn_indices_from_distances,
    neighborhood_overlap_score,
)

PGRID=[5,10,20,30,40,50]
K_VALUES=[5,10,20,50]
SEEDS=[0,42,123]

def stratified_sample_indices(y,n_total=1000,random_state=2026):
    y=np.asarray(y)
    rng=np.random.default_rng(random_state)
    labels=np.unique(y); q,r=divmod(n_total,len(labels)); out=[]
    for pos,label in enumerate(labels):
        idx=np.flatnonzero(y==label)
        n=q+(1 if pos<r else 0)
        if n>len(idx): raise ValueError(f'Class {label} does not contain enough samples.')
        out.extend(rng.choice(idx,size=n,replace=False).tolist())
    return rng.permutation(np.asarray(out,dtype=np.int64))

def load_data():
    df=pd.read_csv(ROOT/'data/processed/optdigits_clean.csv')
    features=[f'Pixel_{i}' for i in range(1,65)]
    assert df.shape==(5620,65)
    assert df.columns.tolist()==features+['label']
    assert not any(str(c).startswith('Unnamed') for c in df.columns)
    X=df[features].to_numpy(np.float64); y=df['label'].to_numpy(np.int64)
    assert X.shape==(5620,64) and np.isfinite(X).all()
    return X,y

def ensure_dirs(output_root):
    out=Path(output_root)
    if not out.is_absolute(): out=(ROOT/out).resolve()
    tables=out/'tables'; emb=out/'embeddings'; figs=out/'figures'
    for d in [out,tables,emb,figs]: d.mkdir(parents=True,exist_ok=True)
    return out,tables,emb,figs

def run_p09(X,y,tables,emb,figs):
    idx=stratified_sample_indices(y,1000,2026); Xs=X[idx]; ys=y[idx]
    pd.DataFrame({'original_index':idx,'label':ys}).to_csv(tables/'optdigits_screening_indices.csv',index=False)
    embeddings={}; rows=[]
    for p in PGRID:
        print(f'P09: perplexity={p}',flush=True)
        t=time.perf_counter()
        m=TSNEFromScratch(perplexity=p,random_state=42,n_iter=750,
                          early_exaggeration_iter=250,momentum_switch_iter=250,
                          patience=120,store_matrices=False)
        Z=m.fit_transform(Xs); runtime=time.perf_counter()-t
        assert Z.shape==(1000,2) and np.isfinite(Z).all()
        assert m.best_kl_divergence_ < m.initial_kl_divergence_
        err=float(np.max(np.abs(m.achieved_perplexities_-p)))
        embeddings[p]=Z.copy()
        rows.append({
            'Perplexity':p,'Samples':len(Xs),'Initial KL':m.initial_kl_divergence_,
            'Best KL':m.best_kl_divergence_,'Final Restored KL':m.kl_divergence_,
            'Best Iteration':m.best_iteration_,'Actual Iterations':m.n_iter_,
            'Stopped Early':m.stopped_early_,'Runtime Seconds':runtime,
            'Achieved Perplexity Min':float(m.achieved_perplexities_.min()),
            'Achieved Perplexity Mean':float(m.achieved_perplexities_.mean()),
            'Achieved Perplexity Max':float(m.achieved_perplexities_.max()),
            'Max Perplexity Abs Error':err,
        })
        pd.DataFrame({'original_index':idx,'TSNE_1':Z[:,0],'TSNE_2':Z[:,1],'label':ys}).to_csv(
            emb/f'optdigits_tsne_perplexity_{p}.csv',index=False)
        gc.collect()
    summary=pd.DataFrame(rows).sort_values('Perplexity').reset_index(drop=True)
    summary.to_csv(tables/'perplexity_screening_summary.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.ravel()
    for ax,p in zip(axes,PGRID):
        Z=embeddings[p]; ax.scatter(Z[:,0],Z[:,1],c=ys,s=8); ax.set_title(f'Perplexity={p}'); ax.grid(alpha=.15)
    fig.suptitle('Optdigits — t-SNE Perplexity Sensitivity'); fig.tight_layout()
    fig.savefig(figs/'optdigits_perplexity_grid.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    print('CHECKPOINT 9 — PASS')
    return idx,Xs,ys,embeddings,summary

def run_p10(Xs,embeddings,tables,figs):
    D=squared_euclidean_distances(Xs); ho,hr=neighbor_order_and_ranks(D)
    for k in K_VALUES:
        # Direct identity check using the exact same ordering/ranks.
        from src.evaluation import trustworthiness_from_ranks, continuity_from_ranks
        assert np.isclose(trustworthiness_from_ranks(ho,hr,ho,k),1.0)
        assert np.isclose(continuity_from_ranks(ho,ho,hr,k),1.0)
    rows=[]
    for p in PGRID:
        row={'Perplexity':p}; row.update(neighborhood_quality(embeddings[p],ho,hr,K_VALUES)); rows.append(row)
    q=pd.DataFrame(rows).sort_values('Perplexity').reset_index(drop=True)
    tc=[c for c in q if c.startswith('T@') or c.startswith('C@')]
    tcols=[c for c in tc if c.startswith('T@')]; ccols=[c for c in tc if c.startswith('C@')]
    q['Mean Trustworthiness']=q[tcols].mean(axis=1); q['Mean Continuity']=q[ccols].mean(axis=1); q['Worst Neighborhood Score']=q[tc].min(axis=1)
    q.to_csv(tables/'perplexity_neighborhood_quality.csv',index=False)
    pareto=pareto_frontier(q,tc); ps=pareto.sort_values('Worst Neighborhood Score',ascending=False)
    if len(ps)>=2: shortlist=ps.head(3).copy()
    else:
        rem=q[~q.Perplexity.isin(ps.Perplexity)].sort_values('Worst Neighborhood Score',ascending=False)
        shortlist=pd.concat([ps,rem.head(2-len(ps))],ignore_index=True)
    shortlist.to_csv(tables/'p10_provisional_candidate_shortlist.csv',index=False)

    fig=plt.figure(figsize=(9,5))
    for k in K_VALUES: plt.plot(q['Perplexity'],q[f'T@{k}'],marker='o',label=f'T@{k}')
    plt.xlabel('Perplexity'); plt.ylabel('Trustworthiness'); plt.ylim(0,1.01); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    fig.savefig(figs/'perplexity_vs_trustworthiness_multiscale.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    fig=plt.figure(figsize=(9,5))
    for k in K_VALUES: plt.plot(q['Perplexity'],q[f'C@{k}'],marker='o',label=f'C@{k}')
    plt.xlabel('Perplexity'); plt.ylabel('Continuity'); plt.ylim(0,1.01); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    fig.savefig(figs/'perplexity_vs_continuity_multiscale.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    candidates=shortlist.Perplexity.astype(int).tolist()
    print('P10 candidates:',candidates)
    print('CHECKPOINT 10 — PASS')
    return D,ho,hr,q,candidates

def run_p11(D,ho,hr,idx,candidates,tables,emb,figs):
    qrows=[]; embeddings={}
    for p in candidates:
        print(f'P11: build P for perplexity={p}',flush=True)
        cr=conditional_probability_matrix(D,p); assert cr['converged'].all(); P=symmetrize_probabilities(cr['P_cond']); embeddings[p]={}
        for seed in SEEDS:
            print(f'P11: p={p} seed={seed}',flush=True)
            t=time.perf_counter()
            o=optimize_tsne_embedding(P,n_iter=750,early_exaggeration_iter=250,momentum_switch_iter=250,patience=120,random_state=seed)
            runtime=time.perf_counter()-t; Z=o['embedding'].copy(); m=neighborhood_quality(Z,ho,hr,K_VALUES)
            row={'Perplexity':p,'Seed':seed,'Runtime Seconds':runtime,'Initial KL':o['initial_kl'],'Final KL':o['kl_divergence'],
                 'Best KL':o['best_kl_divergence'],'Best Iteration':o['best_iteration'],'Actual Iterations':o['n_iter'],'Stopped Early':o['stopped_early']}
            row.update(m); qrows.append(row); embeddings[p][seed]=Z
            pd.DataFrame({'original_index':idx,'TSNE_1':Z[:,0],'TSNE_2':Z[:,1]}).to_csv(emb/f'p11_p{p}_seed{seed}.csv',index=False)
        del P,cr; gc.collect()
    quality=pd.DataFrame(qrows).sort_values(['Perplexity','Seed']).reset_index(drop=True); quality.to_csv(tables/'p11_multiseed_quality.csv',index=False)
    srows=[]
    for p in candidates:
        for a,b in combinations(SEEDS,2):
            for k in K_VALUES:
                na=knn_indices_from_distances(squared_euclidean_distances(embeddings[p][a]),k)
                nb=knn_indices_from_distances(squared_euclidean_distances(embeddings[p][b]),k)
                mean,vals=neighborhood_overlap_score(na,nb)
                srows.append({'Perplexity':p,'Seed A':a,'Seed B':b,'k':k,'Mean kNN Overlap':mean,
                              'Min Sample Overlap':float(vals.min()),'Median Sample Overlap':float(np.median(vals))})
    stability=pd.DataFrame(srows); stability.to_csv(tables/'p11_pairwise_seed_stability.csv',index=False)
    metric_cols=[f'T@{k}' for k in K_VALUES]+[f'C@{k}' for k in K_VALUES]
    robust=[]
    for p in candidates:
        g=quality[quality.Perplexity==p]; s=stability[stability.Perplexity==p]; allv=g[metric_cols].to_numpy(float)
        robust.append({'Perplexity':p,'Mean Neighborhood Score':float(allv.mean()),'Robust Worst Neighborhood':float(allv.min()),
                       'Mean Seed Stability':float(s['Mean kNN Overlap'].mean()),'Worst Seed Stability':float(s['Mean kNN Overlap'].min()),
                       'Runtime Mean':float(g['Runtime Seconds'].mean()),'Runtime Std':float(g['Runtime Seconds'].std(ddof=0)),
                       'Final KL Mean':float(g['Final KL'].mean()),'Final KL Std':float(g['Final KL'].std(ddof=0))})
    robustness=pd.DataFrame(robust); robustness.to_csv(tables/'p11_candidate_robustness_summary.csv',index=False)
    select_cols=['Robust Worst Neighborhood','Mean Neighborhood Score','Worst Seed Stability','Mean Seed Stability']
    pool=pareto_frontier(robustness,select_cols).sort_values(
        ['Robust Worst Neighborhood','Worst Seed Stability','Mean Neighborhood Score','Mean Seed Stability','Runtime Mean'],
        ascending=[False,False,False,False,True]).reset_index(drop=True)
    primary=int(pool.iloc[0].Perplexity)
    if len(pool)>=2: altrow=pool.iloc[1]
    else:
        altrow=robustness[robustness.Perplexity!=primary].sort_values(
            ['Robust Worst Neighborhood','Worst Seed Stability','Mean Neighborhood Score','Mean Seed Stability','Runtime Mean'],
            ascending=[False,False,False,False,True]).iloc[0]
    alternative=int(altrow.Perplexity)
    pd.DataFrame([{'Role':'PRIMARY','Perplexity':primary},{'Role':'ALTERNATIVE','Perplexity':alternative}]).to_csv(tables/'p11_primary_alternative_selection.csv',index=False)

    fig=plt.figure(figsize=(9,5))
    stab=stability.groupby(['Perplexity','k'])['Mean kNN Overlap'].mean().reset_index()
    for k in K_VALUES:
        s=stab[stab.k==k]; plt.plot(s.Perplexity,s['Mean kNN Overlap'],marker='o',label=f'k={k}')
    plt.xlabel('Perplexity'); plt.ylabel('Pairwise Seed kNN Overlap'); plt.ylim(0,1.01); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    fig.savefig(figs/'p11_multiseed_stability.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    print(f'PRIMARY={primary} ALTERNATIVE={alternative}')
    print('CHECKPOINT 11 — PASS')
    return primary,alternative,quality,stability,robustness

def run_p12(X,y,primary,alternative,tables,emb,figs,out):
    n=len(X); print('P12 preflight: N^2=',n*n,'one float64 matrix MiB=',n*n*8/1024**2,flush=True)
    t0=time.perf_counter()
    D=squared_euclidean_distances(X); cr=conditional_probability_matrix(D,primary); assert cr['converged'].all()
    achieved=cr['perplexities'].copy(); P=symmetrize_probabilities(cr['P_cond'])
    del D,cr; gc.collect()

    optimizer_t0=time.perf_counter()
    o=optimize_tsne_embedding(
        P, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0,
        early_exaggeration_iter=250, initial_momentum=0.5, final_momentum=0.8,
        momentum_switch_iter=250, patience=150, random_state=42, verbose=True,
    )
    optimizer_seconds=time.perf_counter()-optimizer_t0
    runtime=time.perf_counter()-t0
    Z=o['embedding'].copy(); assert Z.shape==(5620,2) and np.isfinite(Z).all()
    assert np.isclose(o['kl_divergence'],o['best_kl_divergence'],rtol=1e-8,atol=1e-10)
    tag=f'p{primary}'
    pd.DataFrame({'original_index':np.arange(len(X)),'TSNE_1':Z[:,0],'TSNE_2':Z[:,1],'label':y}).to_csv(emb/f'final_optdigits_embedding_{tag}.csv',index=False)

    pd.DataFrame([
        {'Parameter':'Samples','Value':len(X)},
        {'Parameter':'Original dimensions','Value':X.shape[1]},
        {'Parameter':'Embedding dimensions','Value':Z.shape[1]},
        {'Parameter':'Perplexity','Value':primary},
        {'Parameter':'Learning rate','Value':200.0},
        {'Parameter':'Max iterations','Value':1000},
        {'Parameter':'Early exaggeration','Value':12.0},
        {'Parameter':'Early exaggeration iterations','Value':250},
        {'Parameter':'Initial momentum','Value':0.5},
        {'Parameter':'Final momentum','Value':0.8},
        {'Parameter':'Random state','Value':42},
        {'Parameter':'Implementation','Value':'Exact NumPy t-SNE from scratch'},
    ]).to_csv(tables/f'final_configuration_{tag}.csv',index=False)

    summary=pd.DataFrame([{
        'Samples':len(X),'Features':X.shape[1],'Perplexity':primary,
        'Initial KL':o['initial_kl'],'Best KL':o['best_kl_divergence'],
        'Final Restored KL':o['kl_divergence'],'Best Iteration':o['best_iteration'],
        'Actual Iterations':o['n_iter'],'Stopped Early':o['stopped_early'],
        'Runtime Seconds':runtime,'Optimizer Seconds':optimizer_seconds,
        'Achieved Perplexity Mean':float(achieved.mean()),
        'Achieved Perplexity Min':float(achieved.min()),
        'Achieved Perplexity Max':float(achieved.max()),
        'Achieved Perplexity Max Error':float(np.max(np.abs(achieved-primary))),
    }])
    summary.to_csv(tables/f'final_run_summary_{tag}.csv',index=False)
    pd.DataFrame({'Iteration':np.arange(1,len(o['kl_history'])+1),'KL':o['kl_history'],'Gradient Norm':o['gradient_norm_history']}).to_csv(tables/f'final_optimization_history_{tag}.csv',index=False)

    # Directly evaluate the final 5,620-sample embedding, not only the screening subset.
    # Recompute high-dimensional ranks after optimization to keep peak memory bounded during t-SNE.
    print('P12: evaluating final full-data neighborhood preservation', flush=True)
    D_eval=squared_euclidean_distances(X)
    high_order,high_ranks=neighbor_order_and_ranks(D_eval)
    del D_eval; gc.collect()
    fq=neighborhood_quality(Z,high_order,high_ranks,K_VALUES)
    final_quality=pd.DataFrame([{
        'k':k,'Trustworthiness':fq[f'T@{k}'],'Continuity':fq[f'C@{k}']
    } for k in K_VALUES])
    final_quality.to_csv(tables/f'final_neighborhood_quality_{tag}.csv',index=False)
    final_quality_summary=pd.DataFrame([{
        'Perplexity':primary,'Samples':len(X),
        'Mean Trustworthiness':float(final_quality['Trustworthiness'].mean()),
        'Mean Continuity':float(final_quality['Continuity'].mean()),
        'Worst Neighborhood Score':float(final_quality[['Trustworthiness','Continuity']].to_numpy().min()),
    }])
    final_quality_summary.to_csv(tables/f'final_neighborhood_quality_summary_{tag}.csv',index=False)
    del high_order,high_ranks; gc.collect()

    fig=plt.figure(figsize=(10,8))
    for digit in range(10):
        m=y==digit; plt.scatter(Z[m,0],Z[m,1],s=9,alpha=.7,label=str(digit))
    plt.title(f'Optdigits — t-SNE From Scratch | Perplexity={primary}'); plt.legend(title='Digit',ncol=2); plt.grid(alpha=.15); plt.tight_layout()
    fig.savefig(figs/f'final_optdigits_tsne_{tag}.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    fig=plt.figure(figsize=(9,4)); plt.plot(np.arange(1,len(o['kl_history'])+1),o['kl_history']); plt.axvline(250,ls='--'); plt.xlabel('Iteration'); plt.ylabel('KL(P||Q)'); plt.tight_layout(); fig.savefig(figs/f'final_kl_convergence_{tag}.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    fig=plt.figure(figsize=(9,4)); plt.plot(np.arange(1,len(o['gradient_norm_history'])+1),o['gradient_norm_history']); plt.axvline(250,ls='--'); plt.xlabel('Iteration'); plt.ylabel('Gradient norm'); plt.tight_layout(); fig.savefig(figs/f'final_gradient_norm_{tag}.png',dpi=170,bbox_inches='tight'); plt.close(fig)
    manifest={'project':'t-SNE From Scratch on UCI Optdigits','n_samples':5620,'n_features':64,
              'primary_perplexity':primary,'alternative_perplexity':alternative,'final_random_state':42,
              'final_kl':float(o['kl_divergence']),'runtime_seconds':runtime,'optimizer_seconds':optimizer_seconds,
              'final_neighborhood_quality':{f'T@{k}':float(fq[f'T@{k}']) for k in K_VALUES} | {f'C@{k}':float(fq[f'C@{k}']) for k in K_VALUES},
              'label_usage':'not used in fitting; only stratified screening and post-hoc visualization'}
    (out/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('CHECKPOINT 12 — PASS')
    return summary,Z

def main():
    ap=argparse.ArgumentParser(description='Reproduce the exact t-SNE Optdigits case study.')
    ap.add_argument('--profile',choices=['quick','screening','full'],default='quick')
    ap.add_argument('--output-root',default='rerun_outputs',help='Output directory relative to package root unless absolute.')
    args=ap.parse_args()
    X,y=load_data(); out,tables,emb,figs=ensure_dirs(args.output_root)
    if args.profile=='quick':
        print('Quick profile: data contract + tiny t-SNE smoke run')
        m=TSNEFromScratch(perplexity=10,n_iter=80,early_exaggeration_iter=25,patience=30,random_state=42,store_matrices=False)
        Z=m.fit_transform(X[:80]); assert Z.shape==(80,2) and m.kl_divergence_<m.initial_kl_divergence_
        (out/'run_status.json').write_text(json.dumps({'profile':'quick','status':'PASS'},indent=2),encoding='utf-8')
        print('QUICK PROFILE: PASS'); return
    idx,Xs,ys,screen_embeddings,_=run_p09(X,y,tables,emb,figs)
    D,ho,hr,_,candidates=run_p10(Xs,screen_embeddings,tables,figs)
    if args.profile=='screening':
        (out/'run_status.json').write_text(json.dumps({'profile':'screening','P09':'PASS','P10':'PASS','status':'PASS'},indent=2),encoding='utf-8')
        print('SCREENING PROFILE: PASS'); return
    primary,alternative,_,_,_=run_p11(D,ho,hr,idx,candidates,tables,emb,figs)
    summary,_=run_p12(X,y,primary,alternative,tables,emb,figs,out)
    status={'profile':'full','P09':'PASS','P10':'PASS','P11':'PASS','P12':'PASS','primary_perplexity':primary,'alternative_perplexity':alternative,'final_kl':float(summary.iloc[0]['Final Restored KL']),'status':'PASS'}
    (out/'run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('FULL PROFILE: PASS')

if __name__=='__main__':
    main()
