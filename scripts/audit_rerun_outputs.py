from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tsne_from_scratch import squared_euclidean_distances
from src.evaluation import neighbor_order_and_ranks, neighborhood_quality, knn_indices_from_distances, neighborhood_overlap_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root', default='rerun_outputs')
    args=ap.parse_args()
    OUT=(ROOT/args.output_root).resolve()
    T=OUT/'tables'; E=OUT/'embeddings'; F=OUT/'figures'
    raw=pd.read_csv(ROOT/'data/processed/optdigits_clean.csv')
    feats=[f'Pixel_{i}' for i in range(1,65)]
    X=raw[feats].to_numpy(float); y=raw['label'].to_numpy(int)

    p09=T/'perplexity_screening_summary.csv'
    if not p09.exists():
        raise SystemExit(f'Missing P09 outputs under {OUT}')
    s09=pd.read_csv(p09); pgrid=[5,10,20,30,40,50]
    assert s09['Perplexity'].astype(int).tolist()==pgrid
    assert np.all(s09['Best KL'] < s09['Initial KL'])
    assert np.allclose(s09['Best KL'], s09['Final Restored KL'], rtol=1e-10, atol=1e-12)

    idx_df=pd.read_csv(T/'optdigits_screening_indices.csv'); idx=idx_df['original_index'].to_numpy(int)
    assert len(idx)==1000 and len(np.unique(idx))==1000
    Xs=X[idx]
    D=squared_euclidean_distances(Xs); ho,hr=neighbor_order_and_ranks(D)
    q=pd.read_csv(T/'perplexity_neighborhood_quality.csv').set_index('Perplexity')
    for p in pgrid:
        e=pd.read_csv(E/f'optdigits_tsne_perplexity_{p}.csv')
        Y=e[['TSNE_1','TSNE_2']].to_numpy(float)
        m=neighborhood_quality(Y,ho,hr,k_values=(5,10,20,50))
        for c,v in m.items():
            assert abs(v-float(q.loc[p,c])) < 1e-9

    result={'P09':'PASS','P10':'PASS'}
    sel=T/'p11_primary_alternative_selection.csv'
    if sel.exists():
        sq=pd.read_csv(T/'p11_multiseed_quality.csv'); ss=pd.read_csv(T/'p11_pairwise_seed_stability.csv')
        candidates=sorted(sq['Perplexity'].astype(int).unique().tolist()); seeds=sorted(sq['Seed'].astype(int).unique().tolist())
        assert seeds==[0,42,123]
        for p in candidates:
            Ys={}
            for seed in seeds:
                e=pd.read_csv(E/f'p11_p{p}_seed{seed}.csv'); Ys[seed]=e[['TSNE_1','TSNE_2']].to_numpy(float)
                m=neighborhood_quality(Ys[seed],ho,hr,k_values=(5,10,20,50))
                row=sq[(sq.Perplexity==p)&(sq.Seed==seed)].iloc[0]
                for c,v in m.items(): assert abs(v-float(row[c]))<1e-9
            for ia,a in enumerate(seeds):
                for b in seeds[ia+1:]:
                    for k in [5,10,20,50]:
                        na=knn_indices_from_distances(squared_euclidean_distances(Ys[a]),k)
                        nb=knn_indices_from_distances(squared_euclidean_distances(Ys[b]),k)
                        mean,_=neighborhood_overlap_score(na,nb)
                        row=ss[(ss.Perplexity==p)&(ss['Seed A']==a)&(ss['Seed B']==b)&(ss.k==k)].iloc[0]
                        assert abs(mean-float(row['Mean kNN Overlap']))<1e-9
        s=pd.read_csv(sel); primary=int(s.loc[s.Role=='PRIMARY','Perplexity'].iloc[0]); alt=int(s.loc[s.Role=='ALTERNATIVE','Perplexity'].iloc[0])
        assert primary!=alt
        result.update({'P11':'PASS','primary':primary,'alternative':alt})

        final_csv=E/f'final_optdigits_embedding_p{primary}.csv'
        if final_csv.exists():
            fdf=pd.read_csv(final_csv); assert fdf.shape==(5620,4)
            Y=fdf[['TSNE_1','TSNE_2']].to_numpy(float)
            assert np.isfinite(Y).all() and np.all(Y.std(axis=0)>0)
            summ=pd.read_csv(T/f'final_run_summary_p{primary}.csv').iloc[0]
            hist=pd.read_csv(T/f'final_optimization_history_p{primary}.csv')
            assert int(summ['Samples'])==5620 and int(summ['Features'])==64
            assert float(summ['Best KL']) < float(summ['Initial KL'])
            assert math.isclose(float(summ['Best KL']),float(summ['Final Restored KL']),rel_tol=1e-9,abs_tol=1e-10)
            assert len(hist)==int(summ['Actual Iterations'])
            assert {'Optimizer Seconds','Achieved Perplexity Min','Achieved Perplexity Max'}.issubset(set(summ.index))
            Df=squared_euclidean_distances(X); fho,fhr=neighbor_order_and_ranks(Df)
            fm=neighborhood_quality(Y,fho,fhr,k_values=(5,10,20,50))
            fq=pd.read_csv(T/f'final_neighborhood_quality_p{primary}.csv').set_index('k')
            for k in [5,10,20,50]:
                assert abs(fm[f'T@{k}']-float(fq.loc[k,'Trustworthiness']))<1e-9
                assert abs(fm[f'C@{k}']-float(fq.loc[k,'Continuity']))<1e-9
            for fp in [F/f'final_optdigits_tsne_p{primary}.png',F/f'final_kl_convergence_p{primary}.png',F/f'final_gradient_norm_p{primary}.png']:
                assert fp.exists() and fp.stat().st_size>1000
            result['P12']='PASS'
    print(json.dumps(result,indent=2))
    print('RERUN OUTPUT AUDIT: PASS')

if __name__=='__main__':
    main()
