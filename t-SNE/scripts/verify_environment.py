from __future__ import annotations
from pathlib import Path
import json, platform, sys

ROOT = Path(__file__).resolve().parents[1]
required = {
    'data/processed/optdigits_clean.csv',
    'src/tsne_from_scratch.py',
    'src/evaluation.py',
    'notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb',
}
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    raise SystemExit(f'Missing required package files: {missing}')

mods = {}
for name in ['numpy','pandas','matplotlib','pytest','nbformat']:
    try:
        mod = __import__(name)
        mods[name] = getattr(mod, '__version__', 'unknown')
    except Exception as exc:
        mods[name] = f'NOT AVAILABLE: {exc!r}'

try:
    import psutil
    mem_gib = psutil.virtual_memory().total / (1024**3)
except Exception:
    mem_gib = None

info = {
    'python': sys.version,
    'platform': platform.platform(),
    'executable': sys.executable,
    'package_root': str(ROOT),
    'versions': mods,
    'physical_ram_gib_if_detected': mem_gib,
    'full_run_note': 'Exact N=5620 t-SNE uses multiple 5620x5620 arrays; 4 GiB free RAM minimum recommended, 8 GiB+ preferred.',
}
print(json.dumps(info, indent=2))
if sys.maxsize <= 2**32:
    raise SystemExit('A 64-bit Python interpreter is required for the full run.')
print('ENVIRONMENT CHECK: PASS')
