from __future__ import annotations
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'CHECKSUMS.sha256'
if not manifest.exists():
    raise SystemExit('CHECKSUMS.sha256 not found.')
failed=[]; checked=0
for line in manifest.read_text(encoding='utf-8').splitlines():
    line=line.strip()
    if not line or line.startswith('#'): continue
    digest,rel=line.split('  ',1)
    p=ROOT/rel
    if not p.exists(): failed.append((rel,'missing')); continue
    h=hashlib.sha256(p.read_bytes()).hexdigest(); checked+=1
    if h!=digest: failed.append((rel,'hash mismatch'))
if failed:
    raise SystemExit(f'CHECKSUM VERIFICATION FAILED: {failed[:10]}')
print(f'CHECKSUM VERIFICATION: PASS ({checked} files)')
