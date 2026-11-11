"""MCP wiring (official ``mcp`` SDK, FastMCP). All logic lives in service.py.

Compatibility choices
  * Every tool returns plain JSON *text* (``structured_output=False``): some clients drop image blocks
    when ``structuredContent`` is also present, and plain text works in every client and local model.
  * ``view_sample`` returns text + an image block. Clients/models that cannot take images from tools
    (reported for ChatGPT's connector) must be given the image another way: see README.md "upload mode".
  * stdio for local agents; streamable HTTP (stateless, JSON responses) or SSE for remote ones.
"""
import functools
import json
import time
from typing import Optional

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from . import __version__
from .config import Config
from .protocol import INSTRUCTIONS, PROTOCOL
from .service import Service

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
SUBMIT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)


def _audit(cfg: Config, tool: str, kwargs: dict, ok: bool, error: str | None, started: float) -> None:
    try:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": tool, "ok": ok,
               "agent": kwargs.get("agent") or cfg.agent, "sample": kwargs.get("sample_id"),
               "ms": round((time.time() - started) * 1000)}
        if error:
            row["error"] = error[:300]
        with open(cfg.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass                                   # auditing must never break labeling


def build_server(cfg: Config, host: str = "127.0.0.1", port: int = 8765) -> tuple[FastMCP, Service]:
    svc = Service(cfg)
    mcp = FastMCP(
        "alg-labeler", instructions=INSTRUCTIONS, host=host, port=port,
        streamable_http_path="/mcp", sse_path="/sse", message_path="/messages/",
        stateless_http=True, json_response=True, log_level="WARNING",
        # Host-header checking is done by our own guard (http_app.py), which understands *.ts.net names.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    def audited(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            started, error = time.time(), None
            try:
                return fn(*args, **kwargs)
            except Exception as e:             # noqa: BLE001
                error = str(e)
                raise
            finally:
                _audit(cfg, fn.__name__, kwargs, error is None, error, started)
        return wrapper

    def tool(annotations):
        def deco(fn):
            return mcp.tool(annotations=annotations, structured_output=False)(audited(fn))
        return deco

    # ------------------------------------------------------------------ read-only helpers
    @tool(READ)
    def get_protocol() -> dict:
        """Return the labeling rules. Call this FIRST in every conversation and follow it exactly.
        Core rule: every field is the exact printed text or [UNREADABLE]; never guess."""
        return svc.get_protocol()

    @tool(READ)
    def get_schema(doc_type: str) -> dict:
        """Fields of a document type (ID, TAX or LIC): key, Arabic label, side, format hint and a fictional
        example. Hints show the expected format only; always type what is printed."""
        return svc.get_schema(doc_type)

    @tool(READ)
    def queue_status() -> dict:
        """Counts of samples by status and document type, and who currently holds a claim."""
        return svc.queue_status()

    @tool(READ)
    def list_samples(status: Optional[str] = None, doc_type: Optional[str] = None,
                     limit: int = 50, offset: int = 0) -> dict:
        """List registered samples (status: waiting, done, skipped, needs_human)."""
        return svc.list_samples(status, doc_type, limit, offset)

    # ------------------------------------------------------------------ claiming and viewing
    @tool(WRITE)
    def next_sample(agent: Optional[str] = None, doc_type: Optional[str] = None, name: Optional[str] = None) -> dict:
        """Claim the next sample to label (reserved for you for a while). Pass agent='<your model name>' and use the
        same name in every call. Optional: doc_type to filter, or name (e.g. 'LIC_21') to claim that sample.
        Calling it again returns the sample you already hold."""
        return svc.next_sample(agent, doc_type, name)

    @mcp.tool(annotations=READ, structured_output=False)
    @audited
    def view_sample(sample_id: str, side: str = "front", region: Optional[list[float]] = None,
                    max_side: int = 1600, agent: Optional[str] = None) -> list:
        """Look at a claimed sample. side: 'front' or 'back' (an ID is ONE merged image but has both sides;
        view both). To read small print, ZOOM: region=[x0,y0,x1,y1] as fractions 0..1 of the image, e.g.
        [0,0.5,1,1] is the bottom half. Zoom before you declare any field unreadable."""
        data, info = svc.view_sample(sample_id, side, region, max_side, agent)
        return [json.dumps(info, ensure_ascii=False), Image(data=data, format="jpeg")]

    # ------------------------------------------------------------------ writing the label
    @tool(WRITE)
    def set_fields(sample_id: str, values: Optional[dict[str, str]] = None, unreadable: Optional[dict[str, str]] = None,
                   format_exceptions: Optional[dict[str, str]] = None, agent: Optional[str] = None) -> dict:
        """Set many fields at once. values={key: exact printed text}. unreadable={key: reason_code} for fields
        you cannot read with certainty (reasons: covered, cut_off, blur, glare, low_light, too_small, damaged,
        other). A value is exact printed text or nothing: no '?', no 'N/A', no partial text. If printed text
        breaks the expected format, still type it and add format_exceptions={key: 'why it differs'}."""
        return svc.set_fields(sample_id, values, unreadable, format_exceptions, agent)

    @tool(WRITE)
    def set_field(sample_id: str, key: str, value: str, format_exception: Optional[str] = None,
                  agent: Optional[str] = None) -> dict:
        """Set ONE field to the exact printed text (same rules as set_fields)."""
        exc = {key: format_exception} if format_exception else None
        return svc.set_fields(sample_id, {key: value}, None, exc, agent)

    @tool(WRITE)
    def mark_unreadable(sample_id: str, key: str, reason: str, note: Optional[str] = None,
                        agent: Optional[str] = None) -> dict:
        """Mark a field [UNREADABLE]: use it when you cannot read EVERY character with certainty (covered, cut_off,
        blur, glare, low_light, too_small, damaged, other; 'other' needs a note). Never guess instead."""
        return svc.mark_unreadable(sample_id, key, reason, note, agent)

    @tool(WRITE)
    def set_conditions(sample_id: str, tags: list[str], agent: Optional[str] = None) -> dict:
        """Capture-condition tags for this image: glare, blur, low-res, other. (Tags are also added
        automatically from the unreadable reasons blur, glare and too_small.)"""
        return svc.set_conditions(sample_id, tags, agent)

    # ------------------------------------------------------------------ checking
    @tool(READ)
    def validate_sample(sample_id: str, agent: Optional[str] = None) -> dict:
        """Check your draft: which fields are readable/unreadable/missing, format warnings, MRZ validity and
        consistency, and everything that blocks submit_label. Run it before submitting."""
        return svc.validate_sample(sample_id, agent)

    @tool(READ)
    def mrz_check(sample_id: Optional[str] = None, line1: Optional[str] = None, line2: Optional[str] = None,
                  line3: Optional[str] = None, agent: Optional[str] = None) -> dict:
        """ID only. Verify MRZ lines (30 chars each, check digits) and compare them with the visual fields.
        Pass sample_id to check your saved lines, or line1/line2/line3 to test candidates. It never shows the
        'right' MRZ: re-read the image (zoom) when it reports a problem. 'problems' block submit_label;
        'warnings' (composite check digit only) need a re-check and confirm=true. Never change a character to
        make a check pass: if you are certain a line is what is printed, keep it and give a format_exception."""
        return svc.mrz_check(sample_id, line1, line2, line3, agent)

    @tool(READ)
    def get_truth(sample_id: str, agent: Optional[str] = None) -> dict:
        """Disabled by default (blind labeling): reference values are never shown to agents."""
        return svc.get_truth(sample_id, agent)

    # ------------------------------------------------------------------ finishing
    @tool(SUBMIT)
    def submit_label(sample_id: str, agent: Optional[str] = None, confirm: bool = False,
                     notes: Optional[str] = None) -> dict:
        """Finish the sample. Blocked until every field is set (text or unreadable), format warnings are decided,
        every side was viewed and the MRZ passes its blocking checks (or has a format_exception). If it returns
        needs_confirmation, re-inspect the flagged fields and call again with confirm=true. Labels are written to
        a review folder, not straight into the dataset."""
        return svc.submit_label(sample_id, agent, confirm, notes)

    @tool(WRITE)
    def skip_sample(sample_id: str, reason: str, agent: Optional[str] = None) -> dict:
        """Skip an image that is not a usable document of its type (a reason is required)."""
        return svc.skip_sample(sample_id, reason, agent)

    @tool(WRITE)
    def release_sample(sample_id: str, agent: Optional[str] = None, discard_draft: bool = False) -> dict:
        """Give a claimed sample back to the queue (your draft is kept unless discard_draft=true)."""
        return svc.release_sample(sample_id, agent, discard_draft)

    # ------------------------------------------------------------------ resources and prompts
    @mcp.resource("alg://protocol", mime_type="text/plain")
    def protocol_resource() -> str:
        """The labeling protocol."""
        return PROTOCOL

    @mcp.resource("alg://schema/{doc_type}", mime_type="application/json")
    def schema_resource(doc_type: str) -> str:
        """Schema of ID, TAX or LIC."""
        return json.dumps(svc.get_schema(doc_type), ensure_ascii=False, indent=2)

    @mcp.prompt(name="label_samples", description="Label the next samples following the A.L.G protocol")
    def label_samples(agent: str = "agent", count: int = 5) -> str:
        return (f"You are the labeler '{agent}'. First call get_protocol and follow it exactly. Then label up to "
                f"{count} samples: next_sample(agent='{agent}') -> view_sample for every side (zoom into small "
                "text) -> set_fields / mark_unreadable -> mrz_check (ID) -> validate_sample -> submit_label. "
                "Every field is the exact printed text or [UNREADABLE]; never guess. Stop when next_sample "
                "reports no sample, and finish with a short summary: how many labeled, how many fields "
                "unreadable, and anything you were unsure about.")

    mcp.alg_version = __version__            # handy in tests / doctor
    return mcp, svc
