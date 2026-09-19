from pathlib import Path
import pandas as pd, numpy as np

def test_clean_dataset_contract():
    p=Path(__file__).resolve().parents[1]/'data/processed/optdigits_clean.csv'
    df=pd.read_csv(p); features=[f'Pixel_{i}' for i in range(1,65)]
    assert df.shape==(5620,65); assert df.columns.tolist()==features+['label']
    assert not any(str(c).startswith('Unnamed') for c in df.columns)
    X=df[features].to_numpy(float); assert X.shape==(5620,64); assert np.isfinite(X).all()
