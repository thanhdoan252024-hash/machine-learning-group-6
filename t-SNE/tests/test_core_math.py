import numpy as np
from src.tsne_from_scratch import *

def test_distance_matches_bruteforce():
    rng=np.random.default_rng(0); X=rng.normal(size=(8,5)); D=squared_euclidean_distances(X)
    B=np.array([[np.sum((X[i]-X[j])**2) for j in range(8)] for i in range(8)])
    assert np.allclose(D,B,atol=1e-10)

def test_probability_and_perplexity():
    d=np.array([0.,1.,4.,9.,16.]); p,H,perp=entropy_and_probabilities(d,0.5,0)
    assert np.isclose(p.sum(),1); assert p[0]==0; assert 1<=perp<=4

def test_P_Q_KL_identity_and_gradient():
    rng=np.random.default_rng(1); X=rng.normal(size=(12,4)); D=squared_euclidean_distances(X)
    cr=conditional_probability_matrix(D,3.0,tol=1e-7,max_iter=80); assert cr['converged'].all()
    P=symmetrize_probabilities(cr['P_cond']); Y=initialize_embedding(12,2,3,0.1)
    Q,num,_=low_dimensional_affinities(Y)
    assert np.isclose(P.sum(),1) and np.isclose(Q.sum(),1)
    assert np.all(P>=0) and np.all(Q>=0)
    assert np.allclose(P,P.T,atol=1e-12) and np.allclose(Q,Q.T,atol=1e-12)
    assert np.allclose(np.diag(P),0,atol=1e-15) and np.allclose(np.diag(Q),0,atol=1e-15)
    assert abs(kl_divergence(P,P))<1e-12
    # Q/KL are invariant to a global translation of the low-dimensional embedding.
    Qt,_,_=low_dimensional_affinities(Y+np.array([7.5,-3.25]))
    assert np.allclose(Q,Qt,atol=1e-12)
    g=tsne_gradient(P,Q,num,Y); assert np.allclose(g.sum(axis=0),0,atol=1e-12)
    # Finite differences on several coordinates, not only a single element.
    eps=1e-6
    for i,j in [(0,0),(3,1),(7,0),(11,1)]:
        yp=Y.copy(); ym=Y.copy(); yp[i,j]+=eps; ym[i,j]-=eps
        qp,_,_=low_dimensional_affinities(yp); qm,_,_=low_dimensional_affinities(ym)
        numg=(kl_divergence(P,qp)-kl_divergence(P,qm))/(2*eps)
        assert np.isclose(g[i,j],numg,rtol=1e-4,atol=1e-6)
