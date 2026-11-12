"""Server configuration. Every setting can come from an environment variable or a CLI flag.

Environment variables
  ALG_WORKSPACE           workspace folder (default: the A.L.G app workspace, ~/Documents/RFALG)
  ALG_MCP_AGENT           default agent name (handy for local stdio clients: one client = one agent)
  ALG_MCP_REDUNDANCY      1 (default) or 2: how many different agents must agree on a sample
  ALG_MCP_OUTPUT          ai_labeled (default, safe) or labeled (straight into the app's output folder)
  ALG_MCP_REQUIRE_VIEW    1 (default): an agent must have viewed each side through view_sample
  ALG_MCP_EXPOSE_TRUTH    0 (default): agents never see the truth sidecar values
  ALG_MCP_LEASE_MINUTES   45: how long a claimed sample is reserved for one agent
  ALG_MCP_TOKEN           secret for HTTP mode (Bearer header or secret URL path)
  ALG_MCP_ALLOW_HOSTS     extra allowed Host names for HTTP mode, comma separated
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _flag(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def default_workspace() -> Path:
    env = os.environ.get("ALG_WORKSPACE")
    if env:
        return Path(env).expanduser()
    try:                                    # the same place the desktop app uses
        from alg.appsettings import resolve_workspace
        return Path(resolve_workspace())
    except Exception:                       # noqa: BLE001
        from rfalib.constants import default_workspace as dw
        return Path(dw())


@dataclass
class Config:
    workspace: Path
    agent: str | None = None
    redundancy: int = 1
    output: str = "ai_labeled"
    require_view: bool = True
    expose_truth: bool = False
    lease_minutes: int = 45
    max_unreadable_ratio: float = 0.4
    merge_width: int = 1600             # = rfalib.constants.MERGE_DEFAULT_WIDTH and the GATE Block
    merge_gap: int = 24
    jpeg: bool = False

    # ------------------------------------------------------------------ paths
    @property
    def out_dir(self) -> Path:
        return self.workspace / self.output

    @property
    def human_dir(self) -> Path:
        return self.workspace / "labeled"

    @property
    def state_path(self) -> Path:
        return self.workspace / "alg_mcp_state.json"

    @property
    def audit_path(self) -> Path:
        return self.workspace / "alg_mcp_audit.jsonl"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        env = os.environ
        ws = Path(overrides.pop("workspace", None) or default_workspace()).expanduser()
        cfg = cls(
            workspace=ws,
            agent=overrides.pop("agent", None) or env.get("ALG_MCP_AGENT") or None,
            redundancy=int(overrides.pop("redundancy", None) or env.get("ALG_MCP_REDUNDANCY") or 1),
            output=overrides.pop("output", None) or env.get("ALG_MCP_OUTPUT") or "ai_labeled",
            require_view=_flag(env.get("ALG_MCP_REQUIRE_VIEW"), True),
            expose_truth=_flag(env.get("ALG_MCP_EXPOSE_TRUTH"), False),
            lease_minutes=int(env.get("ALG_MCP_LEASE_MINUTES") or 45),
        )
        for key, value in overrides.items():           # explicit CLI flags win over env vars
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        settings = ws / "settings.json"                # reuse the app's merge settings when present
        if settings.exists():
            try:
                data = json.loads(settings.read_text(encoding="utf-8"))
                cfg.merge_width = int(data.get("merge_width", cfg.merge_width))
                cfg.merge_gap = int(data.get("merge_gap", cfg.merge_gap))
                cfg.jpeg = bool(data.get("jpeg_option", cfg.jpeg))
            except (ValueError, OSError):
                pass
        if cfg.redundancy not in (1, 2):
            raise ValueError("redundancy must be 1 or 2")
        if cfg.output not in ("ai_labeled", "labeled"):
            raise ValueError("output must be 'ai_labeled' or 'labeled'")
        return cfg
