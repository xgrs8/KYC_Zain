"""The way Claude Desktop / Cursor / Cline start a local server: a real subprocess speaking JSON-RPC on stdio."""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LAUNCHER = str(Path(__file__).resolve().parents[1] / "run_alg_mcp.py")


def test_stdio_subprocess_from_another_folder(ws, inbox, tmp_path):
    # register samples with the CLI exactly like a user would
    env = {**os.environ, "ALG_WORKSPACE": str(ws), "ALG_MCP_AGENT": "stdio-agent"}
    params = StdioServerParameters(command=sys.executable, args=[LAUNCHER, "serve", "--transport", "stdio"],
                                   env=env, cwd=str(tmp_path))                  # not the A.L.G folder on purpose

    async def go():
        import subprocess
        subprocess.run([sys.executable, LAUNCHER, "add", str(inbox)], check=True, env=env, cwd=str(tmp_path),
                       capture_output=True)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as client:
                init = await client.initialize()
                assert init.serverInfo.name == "alg-labeler" and "get_protocol FIRST" in init.instructions
                assert len((await client.list_tools()).tools) == 16
                # no agent argument: the default agent comes from ALG_MCP_AGENT
                res = await client.call_tool("next_sample", {"name": "TAX_03"})
                assert not res.isError, res.content
                info = json.loads(res.content[0].text)
                assert info["sample_id"].startswith("TAX-")
                res = await client.call_tool("view_sample", {"sample_id": info["sample_id"]})
                assert [c.type for c in res.content] == ["text", "image"]

    asyncio.run(go())
