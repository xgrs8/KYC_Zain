"""``python -m alg_mcp <command>`` (run from this MCP folder)."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import Config
from .http_app import new_token


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", help="workspace folder (default: the A.L.G app workspace)")
    p.add_argument("--agent", help="default agent name (e.g. claude-desktop)")
    p.add_argument("--redundancy", type=int, choices=(1, 2), help="2 = two different agents must agree")
    p.add_argument("--output", choices=("ai_labeled", "labeled"), help="where AI labels are written (default ai_labeled)")
    p.add_argument("--no-require-view", action="store_true", help="do not insist on view_sample (upload mode)")
    p.add_argument("--expose-truth", action="store_true", help="let agents call get_truth (breaks blind labeling)")


def _cfg(a) -> Config:
    cfg = Config.from_env(workspace=a.workspace, agent=a.agent, redundancy=a.redundancy, output=a.output)
    if getattr(a, "no_require_view", False):
        cfg.require_view = False
    if getattr(a, "expose_truth", False):
        cfg.expose_truth = True
    return cfg


def _token_file(cfg: Config) -> Path:
    return cfg.workspace / "alg_mcp_token.txt"


def _say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------- commands
def cmd_serve(a) -> int:
    from .server import build_server
    cfg = _cfg(a)
    mcp, _svc = build_server(cfg, host=a.host, port=a.port)
    if a.transport == "stdio":
        mcp.run("stdio")                      # nothing may be printed to stdout in this mode
        return 0
    import uvicorn
    from .http_app import build_http_app
    token = None
    if not a.no_auth:
        token = a.token or os.environ.get("ALG_MCP_TOKEN")
        if not token and _token_file(cfg).exists():
            token = _token_file(cfg).read_text(encoding="utf-8").strip()
        if not token:
            print("HTTP mode needs a token. Run:  python -m alg_mcp token --save   (or pass --token)", file=sys.stderr)
            return 2
    elif a.host not in ("127.0.0.1", "localhost", "::1"):
        print("--no-auth is only allowed when binding to 127.0.0.1", file=sys.stderr)
        return 2
    extra = tuple(h.strip() for h in (os.environ.get("ALG_MCP_ALLOW_HOSTS") or "").split(",") if h.strip())
    app = build_http_app(mcp, token, "sse" if a.transport == "sse" else "http", tuple(a.allow_host or ()) + extra)
    path = "/sse" if a.transport == "sse" else "/mcp"
    err = lambda *x: print(*x, file=sys.stderr, flush=True)
    err(f"A.L.G MCP server | transport={a.transport} | workspace={cfg.workspace}")
    err(f"  local URL      : http://{a.host}:{a.port}{path}" + ("   (send header: Authorization: Bearer <token>)" if token else ""))
    if token:
        err(f"  secret-URL form: https://<your-host>.ts.net/s/{token}{path}   <- treat like a password, do not share")
    err("  health check   : /healthz     stop with Ctrl+C")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_add(a) -> int:
    from .service import Service
    svc = Service(_cfg(a))
    if a.manifest:
        import csv
        with open(a.manifest, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    r = svc.add_sample(row["doc_type"], row["front"], row.get("back") or None, row.get("name") or None)
                    _say(("added " if r["added"] else "have  ") + r["sid"])
                except (ValueError, KeyError) as e:
                    _say(f"problem: {row}: {e}")
        return 0
    res = svc.add_paths(a.paths, a.type)
    _say(json.dumps(res, ensure_ascii=False, indent=2))
    return 1 if res["problems"] and not res["added"] else 0


def cmd_status(a) -> int:
    from .service import Service
    svc = Service(_cfg(a))
    _say(json.dumps(svc.queue_status(), ensure_ascii=False, indent=2))
    st = svc.store.snapshot()
    waiting = [s for s in st["samples"].values() if s["status"] == "needs_human"]
    for s in waiting:
        _say(f"needs human: {s['sid']} ({s['name']}): " + (s["last_error"] or
             ", ".join(d["field"] for d in s["disagreements"])))
    ai = [p for p in _cfg(a).out_dir.glob("*.label.ai.json")]
    counts: dict[str, int] = {}
    for p in ai:
        state = json.loads(p.read_text(encoding="utf-8"))["review"]["state"]
        counts[state] = counts.get(state, 0) + 1
    _say("AI labels by review state: " + (json.dumps(counts) if counts else "none yet"))
    return 0


def cmd_token(a) -> int:
    cfg = _cfg(a)
    tok = new_token()
    if a.save:
        _token_file(cfg).write_text(tok + "\n", encoding="utf-8")
        _say(f"saved to {_token_file(cfg)} (serve reads it automatically)")
    _say(tok)
    return 0


def cmd_review(a) -> int:
    from . import review
    path, n = review.review_sheet(_cfg(a), a.n, a.seed, a.only, Path(a.out) if a.out else None)
    _say(f"wrote {path} with {n} labels: open it in a browser")
    return 0


def cmd_decide(a, state: str) -> int:
    from . import review
    _say(json.dumps(review.set_review(_cfg(a), a.bases, state), indent=2))
    return 0


def cmd_promote(a) -> int:
    from . import review
    try:
        res = review.promote(_cfg(a), a.auto_ok, a.approved, a.dry_run)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    _say(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_compare(a) -> int:
    from . import review
    res = review.compare(_cfg(a), Path(a.ai_dir) if a.ai_dir else None, Path(a.human_dir) if a.human_dir else None)
    _say(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_resolve(a) -> int:
    from .service import Service, ToolError
    try:
        _say(json.dumps(Service(_cfg(a)).resolve(a.sample, a.pick), ensure_ascii=False))
    except ToolError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


def cmd_doctor(a) -> int:
    ok_all = True

    def line(ok: bool, text: str) -> None:
        nonlocal ok_all
        ok_all &= ok
        _say(("PASS  " if ok else "FAIL  ") + text)

    line(sys.version_info >= (3, 10), f"python {sys.version.split()[0]} (need 3.10+)")
    try:
        from importlib.metadata import version
        line(version("mcp").startswith("1."), f"mcp SDK {version('mcp')} (tested with 1.30; 2.x is not supported)")
    except Exception as e:  # noqa: BLE001
        line(False, f"mcp SDK missing: {e}")
    for mod in ("PIL", "uvicorn", "alg.label", "rfalib.schema", "rfalib.mrz"):
        try:
            __import__(mod)
            line(True, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            line(False, f"import {mod}: {e}")
    cfg = _cfg(a)
    try:
        cfg.workspace.mkdir(parents=True, exist_ok=True)
        probe = cfg.workspace / ".alg_mcp_probe"
        probe.write_text("x")
        probe.unlink()
        line(True, f"workspace writable: {cfg.workspace}")
    except OSError as e:
        line(False, f"workspace not writable: {e}")
    from .service import Service
    st = Service(cfg).store.snapshot()
    line(True, f"{len(st['samples'])} samples registered | AI output folder: {cfg.out_dir}")
    have = _token_file(cfg).exists() or bool(os.environ.get("ALG_MCP_TOKEN"))
    line(True, "HTTP token: " + ("found" if have else "none yet (only needed for http/sse mode: "
                                 "python -m alg_mcp token --save)"))

    async def handshake() -> int:
        from mcp.shared.memory import create_connected_server_and_client_session
        from .server import build_server
        server, _ = build_server(cfg)
        async with create_connected_server_and_client_session(server._mcp_server) as client:   # noqa: SLF001
            return len((await client.list_tools()).tools)

    try:
        line(True, f"in-process MCP handshake OK, {asyncio.run(handshake())} tools exposed")
    except Exception as e:  # noqa: BLE001
        line(False, f"MCP handshake failed: {e}")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------- parser
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alg-mcp", description="MCP server and tools for AI labeling with A.L.G")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the MCP server")
    _common(s)
    s.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", help="secret for http/sse (or env ALG_MCP_TOKEN, or token file)")
    s.add_argument("--allow-host", action="append", help="extra allowed Host name (repeatable), e.g. mypc.tail1234.ts.net")
    s.add_argument("--no-auth", action="store_true", help="no token (127.0.0.1 only)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("add", help="register images to be labeled")
    _common(s)
    s.add_argument("paths", nargs="*", help="image files or folders")
    s.add_argument("--type", choices=("ID", "TAX", "LIC"), help="document type (otherwise taken from names like ID_01_front.png)")
    s.add_argument("--manifest", help="CSV with columns doc_type,front,back[,name]")
    s.set_defaults(fn=cmd_add)

    for name, fn, help_ in (("status", cmd_status, "queue and review counts"),
                            ("doctor", cmd_doctor, "check the installation")):
        s = sub.add_parser(name, help=help_)
        _common(s)
        s.set_defaults(fn=fn)

    s = sub.add_parser("token", help="make a random token for http mode")
    _common(s)
    s.add_argument("--save", action="store_true", help="store it in the workspace (serve reads it)")
    s.set_defaults(fn=cmd_token)

    s = sub.add_parser("review", help="write an HTML sheet to audit AI labels")
    _common(s)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--only", choices=("needs_review", "auto_ok", "all"), default="needs_review")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_review)

    for name, state in (("approve", "approved"), ("reject", "rejected")):
        s = sub.add_parser(name, help=f"mark AI labels as {state}")
        _common(s)
        s.add_argument("bases", nargs="+", help="label names such as ID_01 LIC_21")
        s.set_defaults(fn=lambda a, st=state: cmd_decide(a, st))

    s = sub.add_parser("promote", help="copy reviewed AI labels into labeled/ (what you train on)")
    _common(s)
    s.add_argument("--auto-ok", action="store_true", help="labels where every check passed")
    s.add_argument("--approved", action="store_true", help="labels you approved")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("compare", help="AI labels vs your own labels of the same images")
    _common(s)
    s.add_argument("--ai-dir")
    s.add_argument("--human-dir")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("resolve", help="settle a sample where agents disagreed: pick one agent's version")
    _common(s)
    s.add_argument("sample", help="sample id or name")
    s.add_argument("--pick", required=True, help="agent whose answers to keep")
    s.set_defaults(fn=cmd_resolve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
