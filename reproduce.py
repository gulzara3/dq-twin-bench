"""Run the whole analysis: tables, figures, then the checks of the quoted numbers.

Usage:
    python reproduce.py            # figures at 600 dpi
    python reproduce.py --dpi 200  # faster
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
dpi = sys.argv[sys.argv.index("--dpi") + 1] if "--dpi" in sys.argv else "600"
steps = [
    ("Tables", ["analysis/make_tables.py"]),
    ("Figures", ["analysis/make_figures.py", "--dpi", dpi]),
    ("Checks of quoted numbers", ["analysis/verify_claims.py", "--quiet"]),
]
for title, cmd in steps:
    print(f"\n=== {title} ===", flush=True)
    code = subprocess.call([sys.executable, *cmd], cwd=HERE)
    if code:
        sys.exit(code)
print("\nAll steps finished.")
