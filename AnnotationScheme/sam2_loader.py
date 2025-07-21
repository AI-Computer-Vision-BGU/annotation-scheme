"""
Ensures that `import sam2 ...` works when this repo is run from its root folder.
It does NOT modify anything inside the SAM-2 submodule.
"""
from pathlib import Path
import sys

SAM2_ROOT = Path(__file__).resolve().parents[1] / "segmentanything"
if str(SAM2_ROOT) not in sys.path:      # idempotent
    sys.path.insert(0, str(SAM2_ROOT))
