"""Launcher for the MCP server; works from any current working directory."""
import sys
from pathlib import Path

MCP_DIR = Path(__file__).resolve().parent
ALG_DIR = MCP_DIR.parent
sys.path.insert(0, str(MCP_DIR))
sys.path.insert(1, str(ALG_DIR))

from alg_mcp.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
