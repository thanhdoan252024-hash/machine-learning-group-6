"""Exact educational t-SNE from scratch using NumPy.

This module intentionally avoids sklearn.manifold.TSNE and automatic differentiation.
It is designed for auditability and mathematical validation, not production-scale speed.
"""
from __future__ import annotations
import numpy as np


def squared_euclidean_distances(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2: raise ValueError("X must be a 2D array.")
    if X.shape[0] < 2 or X.shape[1] < 1: raise ValueError("X has invalid shape.")
    if not np.isfinite(X).all(): raise ValueError("X contains NaN/Inf.")
    norms = np.sum(X * X, axis=1, keepdims=True)
    D = norms + norms.T - 2.0 * (X @ X.T)
    D = 0.5 * (D + D.T)
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def entropy_and_probabilities(distance_row, beta, self_index):
    d = np.asarray(distance_row, dtype=np.float64)
    if d.ndim != 1 or d.size < 2: raise ValueError("distance_row must be 1D with >=2 values.")
    if not np.isfinite(d).all() or np.any(d < -1e-12): raise ValueError("Invalid distances.")
    if not np.isfinite(beta) or beta <= 0: raise ValueError("beta must be finite and >0.")
    if not isinstance(self_index, (int, np.integer)) or not 0 <= self_index < d.size:
        raise ValueError("Invalid self_index.")
    mask = np.ones(d.size, dtype=bool); mask[self_index] = False
    nd = np.maximum(d[mask], 0.0)
    log_w = -beta * nd
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    s = w.sum()
    if not np.isfinite(s) or s <= 0: raise FloatingPointError("Cannot normalize affinities.")
    p = np.zeros(d.size, dtype=np.float64)
    p[mask] = w / s
    positive = p > 0
    H = float(-np.sum(p[positive] * np.log(p[positive])))
    perp = float(np.exp(H))
    return p, H, perp


def binary_search_beta(distance_row, target_perplexity, self_index, tol=1e-5, max_iter=60):
    d = np.asarray(distance_row, dtype=np.float64)
    if d.ndim != 1 or d.size < 3: raise ValueError("distance_row must be 1D with >=3 values.")
    if not (1.0 < float(target_perplexity) <= d.size - 1):
        raise ValueError("target_perplexity must satisfy 1 < perplexity <= n_samples-1.")
    target_H = float(np.log(target_perplexity))
    beta = 1.0
    beta_min, beta_max = -np.inf, np.inf
    converged = False
    p = None; H = None; perp = None; beta_used = beta
    for iteration in range(1, int(max_iter)+1):
        beta_used = beta
        p, H, perp = entropy_and_probabilities(d, beta_used, self_index)
        err = H - target_H
        if abs(err) <= tol:
            converged = True
            break
        if err > 0:  # too diffuse => increase beta
            beta_min = beta_used
            beta = beta_used * 2.0 if np.isinf(beta_max) else 0.5*(beta_used + beta_max)
        else:        # too concentrated => decrease beta
            beta_max = beta_used
            beta = beta_used / 2.0 if np.isinf(beta_min) else 0.5*(beta_used + beta_min)
        if not np.isfinite(beta) or beta <= 0: raise FloatingPointError("Invalid beta during search.")
    return {'probabilities':p, 'beta':float(beta_used), 'entropy':float(H),
            'perplexity':float(perp), 'iterations':int(iteration), 'converged':bool(converged)}


def conditional_probability_matrix(D, target_perplexity, tol=1e-5, max_iter=60):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]: raise ValueError("D must be square.")
    if not np.isfinite(D).all() or np.any(D < -1e-12): raise ValueError("Invalid D.")
    n = D.shape[0]
    if not (1.0 < float(target_perplexity) <= n-1): raise ValueError("Invalid perplexity.")
    P_cond = np.zeros((n,n), dtype=np.float64)
    betas = np.empty(n); perps = np.empty(n); entropies = np.empty(n)
    iterations = np.empty(n, dtype=np.int64); converged = np.empty(n, dtype=bool)
    for i in range(n):
        r = binary_search_beta(D[i], target_perplexity, i, tol=tol, max_iter=max_iter)
        P_cond[i] = r['probabilities']; betas[i] = r['beta']; perps[i] = r['perplexity']
        entropies[i] = r['entropy']; iterations[i] = r['iterations']; converged[i] = r['converged']
    return {'P_cond':P_cond,'betas':betas,'perplexities':perps,'entropies':entropies,
            'iterations':iterations,'converged':converged}


def symmetrize_probabilities(P_cond):
    P_cond = np.asarray(P_cond, dtype=np.float64)
    if P_cond.ndim != 2 or P_cond.shape[0] != P_cond.shape[1]: raise ValueError("P_cond must be square.")
    if not np.isfinite(P_cond).all() or np.any(P_cond < 0): raise ValueError("Invalid P_cond.")
    n = P_cond.shape[0]
    P = (P_cond + P_cond.T) / (2.0*n)
    np.fill_diagonal(P, 0.0)
    return P


def initialize_embedding(n_samples, n_components=2, random_state=42, scale=1e-4):
    if n_samples < 2 or n_components < 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid initialization arguments.")
    rng = np.random.default_rng(random_state)
    return rng.normal(0.0, scale, size=(n_samples,n_components)).astype(np.float64)


def low_dimensional_affinities(Y):
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] < 2 or not np.isfinite(Y).all(): raise ValueError("Invalid Y.")
    D = squared_euclidean_distances(Y)
    numerator = 1.0 / (1.0 + D)
    np.fill_diagonal(numerator, 0.0)
    z = numerator.sum()
    if not np.isfinite(z) or z <= 0: raise FloatingPointError("Invalid Student-t normalizer.")
    Q = numerator / z
    return Q, numerator, D


def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float64); Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2: raise ValueError("P and Q must have same 2D shape.")
    if not np.isfinite(P).all() or not np.isfinite(Q).all() or np.any(P<0) or np.any(Q<0):
        raise ValueError("Invalid probability matrices.")
    mask = P > 0
    if np.any(Q[mask] <= 0): raise FloatingPointError("Q <= 0 where P > 0.")
    return float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))))


def tsne_gradient(P, Q, numerator, Y):
    P = np.asarray(P,dtype=np.float64); Q=np.asarray(Q,dtype=np.float64)
    numerator=np.asarray(numerator,dtype=np.float64); Y=np.asarray(Y,dtype=np.float64)
    if P.ndim!=2 or P.shape[0]!=P.shape[1] or Q.shape!=P.shape or numerator.shape!=P.shape:
        raise ValueError("P/Q/numerator shapes are invalid.")
    if Y.ndim!=2 or Y.shape[0]!=P.shape[0]: raise ValueError("Y shape mismatch.")
    if any(not np.isfinite(a).all() for a in [P,Q,numerator,Y]): raise ValueError("NaN/Inf detected.")
    A = (P - Q) * numerator
    grad = 4.0 * (A.sum(axis=1, keepdims=True) * Y - A @ Y)
    if not np.isfinite(grad).all(): raise FloatingPointError("Gradient contains NaN/Inf.")
    return grad


def optimize_tsne_embedding(P, n_components=2, learning_rate=200.0, n_iter=750,
                            early_exaggeration=12.0, early_exaggeration_iter=250,
                            initial_momentum=0.5, final_momentum=0.8, momentum_switch_iter=250,
                            min_gain=0.01, patience=120, min_kl_improvement=1e-7,
                            random_state=42, init_scale=1e-4, verbose=False):
    P = np.asarray(P,dtype=np.float64)
    if P.ndim!=2 or P.shape[0]!=P.shape[1]: raise ValueError("P must be square.")
    if not np.isfinite(P).all() or np.any(P<0): raise ValueError("Invalid P.")
    if not np.isclose(P.sum(),1.0,atol=1e-10) or not np.allclose(P,P.T,atol=1e-12):
        raise ValueError("P must be symmetric and sum to 1.")
    if not (0 <= early_exaggeration_iter < n_iter): raise ValueError("Invalid exaggeration schedule.")
    n = P.shape[0]
    Y = initialize_embedding(n,n_components,random_state,init_scale)
    previous_update = np.zeros_like(Y); gains=np.ones_like(Y)
    kl_history=[]; grad_history=[]; best_kl=np.inf; best_Y=None; best_iteration=None
    no_improvement=0; stopped_early=False; actual_iterations=0
    Q0,_,_=low_dimensional_affinities(Y); initial_kl=kl_divergence(P,Q0)
    for iteration in range(1,int(n_iter)+1):
        actual_iterations=iteration
        Y_current = Y.copy()
        Q,numerator,_ = low_dimensional_affinities(Y_current)
        current_kl = kl_divergence(P,Q)
        kl_history.append(current_kl)
        in_exaggeration = iteration <= early_exaggeration_iter
        P_work = P*early_exaggeration if in_exaggeration else P
        grad = tsne_gradient(P_work,Q,numerator,Y_current)
        grad_history.append(float(np.linalg.norm(grad)))
        # Track best state only after exaggeration; current_kl corresponds exactly to Y_current.
        if not in_exaggeration:
            if current_kl < best_kl - min_kl_improvement:
                best_kl=current_kl; best_Y=Y_current.copy(); best_iteration=iteration; no_improvement=0
            else:
                no_improvement += 1
            if patience is not None and patience>0 and no_improvement >= patience:
                stopped_early=True
                if verbose: print(f"Early stopping at iteration {iteration}")
                break
        momentum = initial_momentum if iteration < momentum_switch_iter else final_momentum
        sign_changed = np.sign(grad) != np.sign(previous_update)
        gains = np.where(sign_changed, gains+0.2, gains*0.8)
        gains = np.maximum(gains, min_gain)
        update = momentum*previous_update - learning_rate*gains*grad
        Y = Y_current + update
        Y -= Y.mean(axis=0, keepdims=True)
        previous_update = update
        if not np.isfinite(Y).all(): raise FloatingPointError("Embedding diverged.")
        if verbose and (iteration==1 or iteration%50==0 or iteration==early_exaggeration_iter):
            print(f"Iter {iteration:4d} | KL={current_kl:.6f} | grad={grad_history[-1]:.6f}")
    # Evaluate state after final update when loop exhausted normally.
    Q_last,_,_=low_dimensional_affinities(Y); last_kl=kl_divergence(P,Q_last)
    if actual_iterations > early_exaggeration_iter and last_kl < best_kl:
        best_kl=last_kl; best_Y=Y.copy(); best_iteration=actual_iterations
    if best_Y is None:
        best_Y=Y.copy(); best_kl=last_kl; best_iteration=actual_iterations
    Q_best,_,_=low_dimensional_affinities(best_Y); restored_kl=kl_divergence(P,Q_best)
    return {'embedding':best_Y,'initial_kl':float(initial_kl),'kl_divergence':float(restored_kl),
            'best_kl_divergence':float(best_kl),'last_kl_before_restore':float(last_kl),
            'best_iteration':int(best_iteration),'kl_history':np.asarray(kl_history),
            'gradient_norm_history':np.asarray(grad_history),'n_iter':int(actual_iterations),
            'stopped_early':bool(stopped_early),'final_gains':gains.copy()}


class TSNEFromScratch:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=750,
                 early_exaggeration=12.0, early_exaggeration_iter=250,
                 initial_momentum=0.5, final_momentum=0.8, momentum_switch_iter=250,
                 min_gain=0.01, perplexity_tol=1e-5, perplexity_max_iter=60,
                 patience=120, min_kl_improvement=1e-7, init_scale=1e-4,
                 random_state=42, verbose=False, store_matrices=True):
        self.n_components=n_components; self.perplexity=perplexity; self.learning_rate=learning_rate
        self.n_iter=n_iter; self.early_exaggeration=early_exaggeration
        self.early_exaggeration_iter=early_exaggeration_iter; self.initial_momentum=initial_momentum
        self.final_momentum=final_momentum; self.momentum_switch_iter=momentum_switch_iter
        self.min_gain=min_gain; self.perplexity_tol=perplexity_tol
        self.perplexity_max_iter=perplexity_max_iter; self.patience=patience
        self.min_kl_improvement=min_kl_improvement; self.init_scale=init_scale
        self.random_state=random_state; self.verbose=verbose; self.store_matrices=store_matrices
        self.is_fitted_=False
    def fit(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim!=2 or X.shape[0]<3 or not np.isfinite(X).all(): raise ValueError("Invalid X.")
        n,f=X.shape
        if not (1.0 < self.perplexity <= n-1): raise ValueError("Invalid perplexity.")
        D=squared_euclidean_distances(X)
        cr=conditional_probability_matrix(D,self.perplexity,self.perplexity_tol,self.perplexity_max_iter)
        if not cr['converged'].all(): raise RuntimeError("Perplexity matching did not converge for all rows.")
        P=symmetrize_probabilities(cr['P_cond'])
        opt=optimize_tsne_embedding(P,n_components=self.n_components,learning_rate=self.learning_rate,
            n_iter=self.n_iter,early_exaggeration=self.early_exaggeration,
            early_exaggeration_iter=self.early_exaggeration_iter,initial_momentum=self.initial_momentum,
            final_momentum=self.final_momentum,momentum_switch_iter=self.momentum_switch_iter,
            min_gain=self.min_gain,patience=self.patience,min_kl_improvement=self.min_kl_improvement,
            random_state=self.random_state,init_scale=self.init_scale,verbose=self.verbose)
        self.n_samples_=n; self.n_features_in_=f; self.betas_=cr['betas']; self.achieved_perplexities_=cr['perplexities']
        self.perplexity_search_iterations_=cr['iterations']; self.embedding_=opt['embedding']; self.kl_history_=opt['kl_history']
        self.gradient_norm_history_=opt['gradient_norm_history']; self.initial_kl_divergence_=opt['initial_kl']
        self.best_kl_divergence_=opt['best_kl_divergence']; self.kl_divergence_=opt['kl_divergence']
        self.best_iteration_=opt['best_iteration']; self.n_iter_=opt['n_iter']; self.stopped_early_=opt['stopped_early']
        if self.store_matrices:
            self.distance_matrix_=D; self.conditional_probabilities_=cr['P_cond']; self.P_=P
        else:
            self.distance_matrix_=None; self.conditional_probabilities_=None; self.P_=None
        self.is_fitted_=True
        return self
    def fit_transform(self,X):
        return self.fit(X).embedding_.copy()
