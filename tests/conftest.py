"""Add project directories to sys.path for tests."""
import sys
from pathlib import Path

# Add the project root so app/ can be imported (same as uv run behavior)
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
