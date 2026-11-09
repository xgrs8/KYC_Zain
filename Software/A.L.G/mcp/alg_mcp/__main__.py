"""``python -m alg_mcp`` entry point."""
import sys
from pathlib import Path

# The MCP package lives below the application package so it can be installed
# independently while still reusing A.L.G's label engine and schemas.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .cli import main

raise SystemExit(main(sys.argv[1:]))
