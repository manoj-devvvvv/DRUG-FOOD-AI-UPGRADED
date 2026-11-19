"""Make the project import layout available when pytest runs from dfi/."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Tests must not depend on Gemini or PubChem network access.
os.environ.setdefault("DFI_ONLINE_RESOLVER", "0")
