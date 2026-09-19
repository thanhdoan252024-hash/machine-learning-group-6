"""Neighborhood-preservation evaluation helpers for the t-SNE case study."""
import numpy as np
from .tsne_from_scratch import squared_euclidean_distances

def knn_indices_from_distances(D,k):
    D=np.asarray(D,dtype=np.float64)
    if D.ndim!=2 or D.shape[0]!=D.shape[1] or not np.isfinite(D).all(): raise ValueError("Invalid D.")
    n=D.shape[0]
    if not 1<=k<n: raise ValueError("k must be in [1,N-1].")
    W=D.copy(); np.fill_diagonal(W,np.inf)
    return np.argsort(W,axis=1,kind='mergesort')[:,:k]

def neighborhood_overlap_score(a,b):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape: raise ValueError("Neighbor matrices must match.")
    n,k=a.shape; out=np.empty(n,dtype=np.float64)
    for i in range(n): out[i]=len(set(a[i].tolist()).intersection(b[i].tolist()))/k
    return float(out.mean()),out

def neighbor_order_and_ranks(D):
    D=np.asarray(D,dtype=np.float64)
    if D.ndim!=2 or D.shape[0]!=D.shape[1] or not np.isfinite(D).all(): raise ValueError("Invalid D.")
    n=D.shape[0]; W=D.copy(); np.fill_diagonal(W,np.inf)
    order=np.argsort(W,axis=1,kind='mergesort')
    ranks=np.empty((n,n),dtype=np.int32); rows=np.arange(n)[:,None]
    ranks[rows,order]=np.arange(1,n+1,dtype=np.int32)[None,:]
    return order,ranks

def trustworthiness_from_ranks(high_order,high_ranks,low_order,k):
    n=high_order.shape[0]
    if not 1<=k<n/2: raise ValueError("Require 1 <= k < N/2.")
    low_knn=low_order[:,:k]; rows=np.arange(n)[:,None]
    r=high_ranks[rows,low_knn]; penalty=np.sum((r-k)*(r>k))
    return float(1.0 - 2.0*penalty/(n*k*(2*n-3*k-1)))

def continuity_from_ranks(high_order,low_order,low_ranks,k):
    n=high_order.shape[0]
    if not 1<=k<n/2: raise ValueError("Require 1 <= k < N/2.")
    high_knn=high_order[:,:k]; rows=np.arange(n)[:,None]
    r=low_ranks[rows,high_knn]; penalty=np.sum((r-k)*(r>k))
    return float(1.0 - 2.0*penalty/(n*k*(2*n-3*k-1)))

def neighborhood_quality(embedding,high_order,high_ranks,k_values=(5,10,20,50)):
    D=squared_euclidean_distances(embedding); low_order,low_ranks=neighbor_order_and_ranks(D)
    out={}
    for k in k_values:
        out[f'T@{k}']=trustworthiness_from_ranks(high_order,high_ranks,low_order,k)
        out[f'C@{k}']=continuity_from_ranks(high_order,low_order,low_ranks,k)
    return out

def pareto_frontier(df, metric_columns, atol=1e-12):
    vals=df[metric_columns].to_numpy(dtype=np.float64); keep=np.ones(len(df),dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i==j: continue
            if np.all(vals[j]>=vals[i]-atol) and np.any(vals[j]>vals[i]+atol): keep[i]=False; break
    return df.loc[keep].copy().reset_index(drop=True)
