from pathlib import Path
import sys, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.tsne_from_scratch import TSNEFromScratch
p=ROOT/'data/processed/optdigits_clean.csv'; df=pd.read_csv(p); features=[f'Pixel_{i}' for i in range(1,65)]
X=df[features].to_numpy(float)[:80]
model=TSNEFromScratch(perplexity=10,n_iter=80,early_exaggeration_iter=25,patience=30,random_state=42,store_matrices=False)
Y=model.fit_transform(X)
assert Y.shape==(80,2) and np.isfinite(Y).all() and model.kl_divergence_ < model.initial_kl_divergence_
print('QUICK SMOKE TEST: PASS', {'initial_kl':model.initial_kl_divergence_,'final_kl':model.kl_divergence_,'iterations':model.n_iter_})
