import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tests must stay offline: keep ChromaDB's anonymous telemetry from phoning home.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
