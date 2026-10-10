import os
import sys
from pathlib import Path

# Must be set before any src.* import so src.paths enters demo mode.
os.environ.setdefault("DEMO_MODE", "1")

# Repository modules are imported as src.* from project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
