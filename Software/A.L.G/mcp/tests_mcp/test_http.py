import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.sse import sse_client

import contextlib

try:                                   # the client helper was renamed between SDK releases
    from mcp.client.streamable_http import streamable_http_client
    NEW_CLIENT = True
except ImportError:                    # pragma: no cover
    from mcp.client.streamable_http import streamablehttp_client
    NEW_CLIENT = False

from alg_mcp.config import Config
from alg_mcp.http_app import build_http_app, new_token
from alg_mcp.server import build_server

TOKEN = new_token()


@contextlib.asynccontextmanager
async def connect(url: str, headers: dict | None = None):
    """An initialized MCP client session over streamable HTTP (works with both SDK helper names)."""
    if NEW_CLIENT:
        async with httpx.AsyncClient(headers=headers or {}, timeout=httpx.Timeout(30.0, read=60.0)) as hc:
            async with streamable_http_client(url, http_client=hc) as (r, w, _):
                async with ClientSession(r, w) as client:
                    await client.initialize()
                    yield client
    else:                                                  # pragma: no cover
        async with streamablehttp_client(url, headers=headers) as (r, w, _):
            async with ClientSession(r, w) as client:
                await client.initialize()
                yield client


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Served:
    def __init__(self, ws, inbox, transport="http", token=TOKEN):
        cfg = Config.from_env(workspace=ws)
        mcp, self.svc = build_server(cfg)
        self.svc.add_paths([str(inbox)])
        self.port = free_port()
        app = build_http_app(mcp, token, transport)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                break
            time.sleep(0.05)
        assert self.server.started, "server did not start"
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def served(ws, inbox):
    s = Served(ws, inbox)
    yield s
    s.close()


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "pytest", "version": "0"}}}
HDRS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def test_health_is_open_and_everything_else_needs_auth(served):
    assert httpx.get(f"{served.base}/healthz").text == "ok"
    r = httpx.post(f"{served.base}/mcp", json=INIT, headers=HDRS)
    assert r.status_code == 401 and "bearer" in r.headers["www-authenticate"].lower()
    r = httpx.post(f"{served.base}/mcp", json=INIT, headers={**HDRS, "Authorization": "Bearer wrong-token-wrong-token"})
    assert r.status_code == 401
    r = httpx.post(f"{served.base}/s/not-the-secret/mcp", json=INIT, headers=HDRS)
    assert r.status_code == 401
    r = httpx.get(f"{served.base}/s/{TOKEN}x/mcp")
    assert r.status_code == 401                                    # a longer/different secret must not match the prefix


def test_valid_credentials_reach_the_server(served):
    r = httpx.post(f"{served.base}/mcp", json=INIT, headers={**HDRS, "Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "alg-labeler"
    r = httpx.post(f"{served.base}/s/{TOKEN}/mcp", json=INIT, headers=HDRS)
    assert r.status_code == 200 and "tools" in r.json()["result"]["capabilities"]


def test_host_allow_list_accepts_funnel_names_and_rejects_others(served):
    ok = httpx.post(f"{served.base}/s/{TOKEN}/mcp", json=INIT, headers={**HDRS, "Host": "mypc.tail1234.ts.net"})
    assert ok.status_code == 200                                   # what a Tailscale Funnel request looks like
    bad = httpx.post(f"{served.base}/s/{TOKEN}/mcp", json=INIT, headers={**HDRS, "Host": "evil.example.com"})
    assert bad.status_code == 421


def test_real_mcp_client_over_bearer_header(served):
    async def go():
        async with connect(f"{served.base}/mcp", {"Authorization": f"Bearer {TOKEN}"}) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"next_sample", "view_sample", "submit_label"} <= names
            res = await client.call_tool("next_sample", {"agent": "http-agent", "name": "TAX_03"})
            sid = json.loads(res.content[0].text)["sample_id"]
            res = await client.call_tool("view_sample", {"sample_id": sid, "agent": "http-agent"})
            assert [c.type for c in res.content] == ["text", "image"]

    asyncio.run(go())


def test_real_mcp_client_over_secret_url(served):
    async def go():
        async with connect(f"{served.base}/s/{TOKEN}/mcp") as client:                  # no headers at all
            res = await client.call_tool("queue_status", {})
            assert json.loads(res.content[0].text)["total"] == 4

    asyncio.run(go())


def test_sse_transport_with_secret_url(ws, inbox):
    s = Served(ws, inbox, transport="sse")
    try:
        async def go():
            async with sse_client(f"{s.base}/s/{TOKEN}/sse") as (r, w):
                async with ClientSession(r, w) as client:
                    await client.initialize()
                    assert len((await client.list_tools()).tools) == 16

        asyncio.run(go())
    finally:
        s.close()


def test_short_tokens_are_refused():
    with pytest.raises(ValueError, match="too short"):
        build_http_app(build_server(Config.from_env(workspace=__import__("tempfile").mkdtemp()))[0], "abc")
