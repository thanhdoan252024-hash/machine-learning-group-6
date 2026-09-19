from __future__ import annotations
import argparse, os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'notebook/TSNE_From_Scratch_Optdigits_P00_P13_With_AI_Prompting_Log.ipynb'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=['quick','screening','full'], default='quick')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--output-dir', default='rerun_outputs/notebooks')
    args = ap.parse_args()

    out_dir = ROOT/args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir/f'TSNE_From_Scratch_Optdigits_{args.profile.upper()}_RERUN.ipynb'

    os.environ['TSNE_EXECUTION_PROFILE'] = args.profile
    os.environ['TSNE_OUTPUT_ROOT'] = str((ROOT/'rerun_outputs').resolve())
    nb = nbformat.read(SRC, as_version=4)
    client = NotebookClient(nb, timeout=args.timeout, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    nbformat.write(nb, out)
    print(f'EXECUTED NOTEBOOK SAVED: {out}')
    print('NOTEBOOK EXECUTION: PASS')

if __name__ == '__main__':
    main()
