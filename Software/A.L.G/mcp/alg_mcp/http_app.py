"""HTTP deployment: wraps the SDK's ASGI app with authentication and a Host allow-list.

Two ways to authenticate (both use the same token):
  * Bearer header   ``Authorization: Bearer <token>``        (Claude Code, Cursor, Cline, mcp-remote ...)
  * Secret URL      ``https://host/s/<token>/mcp``            (clients that only accept a URL: ChatGPT
                                                               "no authentication", Claude.ai connectors)
Anything else gets 401. ``GET /healthz`` answers "ok" without auth and reveals nothing else.
"""
import fnmatch
import hmac
import json
import secrets

from mcp.server.fastmcp import FastMCP

DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1", "*.ts.net")


def new_token() -> str:
    return secrets.token_urlsafe(32)


def _host_only(host_header: str) -> str:
    h = (host_header or "").strip().lower()
    if h.startswith("["):                       # [::1]:8765
        return h.split("]")[0] + "]"
    return h.split(":")[0]


class Guard:
    """Pure ASGI middleware (keeps streaming responses and the SDK's lifespan intact)."""

    def __init__(self, app, token: str | None, allow_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS):
        if token is not None and len(token) < 16:
            raise ValueError("token is too short: use at least 16 characters (alg-mcp token makes a good one)")
        self.app = app
        self.token = token
        self.allow_hosts = tuple(h.lower() for h in allow_hosts)
        self.prefix = f"/s/{token}" if token else None

    async def _reply(self, send, status: int, body: dict | str, headers: list | None = None) -> None:
        payload = (body if isinstance(body, str) else json.dumps(body)).encode()
        ctype = b"text/plain" if isinstance(body, str) else b"application/json"
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", ctype), (b"content-length", str(len(payload)).encode())]
                    + (headers or [])})
        await send({"type": "http.response.body", "body": payload})

    def _host_ok(self, host: str) -> bool:
        return any(fnmatch.fnmatch(host, pat) for pat in self.allow_hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":                 # lifespan / websocket: pass through untouched
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if not self._host_ok(_host_only(headers.get("host", ""))):
            return await self._reply(send, 421, {"error": "host not allowed"})
        path = scope.get("path", "")
        if path == "/healthz":
            return await self._reply(send, 200, "ok")
        if self.token is None:                      # --no-auth (local use only)
            return await self.app(scope, receive, send)
        authed = False
        if path == self.prefix or path.startswith(self.prefix + "/"):
            authed = True
            new_path = path[len(self.prefix):] or "/"
            scope = dict(scope, path=new_path, raw_path=new_path.encode())
        else:
            auth = headers.get("authorization", "")
            if auth[:7].lower() == "bearer " and hmac.compare_digest(auth[7:].strip().encode(), self.token.encode()):
                authed = True
        if not authed:
            return await self._reply(send, 401, {"error": "unauthorized"},
                                     [(b"www-authenticate", b'Bearer realm="alg-mcp"')])
        return await self.app(scope, receive, send)


def build_http_app(mcp: FastMCP, token: str | None, transport: str = "http",
                   allow_hosts: tuple[str, ...] = ()) -> Guard:
    """transport: 'http' (streamable HTTP at /mcp, the modern default) or 'sse' (legacy /sse)."""
    hosts = tuple(DEFAULT_ALLOWED_HOSTS) + tuple(allow_hosts)
    if transport == "sse":
        # the SSE endpoint event must contain the secret prefix so that the client's POSTs stay authenticated
        inner = mcp.sse_app(mount_path=f"/s/{token}" if token else None)
    else:
        inner = mcp.streamable_http_app()
    return Guard(inner, token, hosts)
