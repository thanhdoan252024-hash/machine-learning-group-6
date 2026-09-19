from __future__ import annotations
from pathlib import Path
import os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ)
env["PYTHONDONTWRITEBYTECODE"] = "1"
commands = [
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
    [sys.executable, str(ROOT/"scripts/audit_package.py")],
    [sys.executable, str(ROOT/"scripts/audit_empirical_outputs.py")],
    [sys.executable, str(ROOT/"scripts/quick_smoke_test.py")],
]
for cmd in commands:
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)
print("\nALL CHECKS: PASS")
