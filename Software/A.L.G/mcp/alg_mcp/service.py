"""The labeling service: all rules live here, independent of MCP, so they can be tested directly.

Design rules
  * Agents only ever see samples that were registered by the human (``alg-mcp add``); no tool takes a path.
  * A label is either the exact printed text or [UNREADABLE]. The service rejects the usual dodges
    ("N/A", "?", partial text, notes in brackets...) and routes them to ``mark_unreadable``.
  * Format warnings of the app (regex per field) are blocking for agents unless the agent states a
    reason (``format_exception``); the exception is logged exactly like the app's overridden warnings.
  * Agents never see the generator's reference ("truth") values. The service compares afterwards and
    writes the result into the audit file for the human reviewer.
  * Everything an agent submits is written with the app's own ``LabelSession`` (same merge, same
    validation, same sidecar) into ``ai_labeled/`` by default, plus ``<base>.label.ai.json``.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from alg.label import LabelSession
from rfalib import naming as NM
from rfalib import schema as S
from rfalib.constants import CONDITION_TAGS, DOC_TYPES
from rfalib.normalize import same

from . import images, mrz_tools
from .config import Config
from .protocol import (EVASIONS, HINTS, PROTOCOL, REASON_TO_CONDITION, REASONS, UNREADABLE)
from .store import Store, new_draft, new_sample

AGENT_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,40}$")
_SOURCE_RE = re.compile(r"^(ID|TAX|LIC)_(\d+)", re.I)
_PAIR_RE = re.compile(r"^(?P<stem>.+?)_(?P<side>front|back)$", re.I)
_UNCERTAIN_RE = re.compile(r"[?\u061F\u2026\ufffd]|\.{3}")
_NOTE_RE = re.compile(r"[\[\]{}]")


class ToolError(Exception):
    """An error the agent can act on (becomes an MCP tool error with this message)."""


def _now() -> float:
    return time.time()


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.workspace.mkdir(parents=True, exist_ok=True)
        self.store = Store(cfg.state_path)

    # ================================================================== helpers
    def _agent(self, agent: str | None) -> str:
        name = (agent or self.cfg.agent or "").strip()
        if not name:
            raise ToolError("Say who you are: pass agent='<your model name>' (for example 'claude', "
                            "'chatgpt', 'glm-5v'). Use the same name in every call.")
        if not AGENT_RE.match(name):
            raise ToolError("agent must be 1-40 characters: letters, digits, _ . : -")
        return name

    @staticmethod
    def _find(st: dict, ref: str) -> dict:
        ref = (ref or "").strip()
        samples = st["samples"]
        if ref in samples:
            return samples[ref]
        hits = [s for s in samples.values() if s["name"].lower() == ref.lower()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ToolError(f"name {ref!r} is ambiguous: use the sample_id")
        raise ToolError(f"unknown sample {ref!r}: call next_sample to get one")

    def _keys(self, doc_type: str) -> tuple[str, ...]:
        return S.label_keys(doc_type)

    @staticmethod
    def _sides(doc_type: str) -> list[str]:
        return ["front", "back"] if doc_type == "ID" else ["front"]

    def _expire(self, st: dict, now: float) -> None:
        for s in st["samples"].values():
            for a in [a for a, c in s["claims"].items() if c["expires"] < now]:
                del s["claims"][a]

    def _active_others(self, s: dict, agent: str, now: float) -> set[str]:
        return {a for a, c in s["claims"].items() if c["expires"] > now and a != agent}

    def _draft(self, st: dict, s: dict, agent: str) -> dict:
        """The agent's draft for this sample; requires (or re-acquires) an active claim."""
        now = _now()
        claim = s["claims"].get(agent)
        if claim and claim["expires"] > now:
            claim["expires"] = now + self.cfg.lease_minutes * 60
        else:
            done_by = {x["agent"] for x in s["submissions"]}
            if s["status"] != "waiting" or agent in done_by:
                raise ToolError(f"sample {s['sid']} is not open for you (status: {s['status']}). Call next_sample.")
            if len(self._active_others(s, agent, now)) + len(s["submissions"]) >= self.cfg.redundancy:
                raise ToolError(f"sample {s['sid']} is claimed by another agent. Call next_sample.")
            s["claims"][agent] = {"at": now, "expires": now + self.cfg.lease_minutes * 60}
        draft = s["drafts"].setdefault(agent, new_draft())
        draft["updated"] = now
        return draft

    # ================================================================== schema / protocol
    def get_protocol(self) -> dict:
        return {"protocol": PROTOCOL, "unreadable_token": UNREADABLE, "reason_codes": REASONS,
                "condition_tags": list(CONDITION_TAGS), "redundancy": self.cfg.redundancy,
                "require_view": self.cfg.require_view}

    def get_schema(self, doc_type: str) -> dict:
        doc_type = (doc_type or "").upper()
        if doc_type not in DOC_TYPES:
            raise ToolError(f"doc_type must be one of {list(DOC_TYPES)}")
        fields = []
        for i, meta in enumerate(S.fields_meta(doc_type), 1):
            what, example, side = HINTS.get(doc_type, {}).get(meta["key"], ("as printed", "", None))
            fields.append({"order": i, "key": meta["key"], "label_ar": meta.get("ar") or meta.get("label_ar", ""),
                           "rtl": bool(meta.get("rtl")), "side": side, "format": what, "example": example,
                           "regex": meta["pattern"]})
        return {"doc_type": doc_type, "sides": self._sides(doc_type), "field_count": len(fields),
                "fields": fields, "note": "Examples are fictional and only show the format. "
                                          "Never copy an example: type what is printed."}

    # ================================================================== queue
    def queue_status(self) -> dict:
        st = self.store.snapshot()
        now = _now()
        by_status: dict[str, int] = {}
        by_type: dict[str, dict[str, int]] = {}
        claims = []
        for s in st["samples"].values():
            by_status[s["status"]] = by_status.get(s["status"], 0) + 1
            t = by_type.setdefault(s["doc_type"], {})
            t[s["status"]] = t.get(s["status"], 0) + 1
            for a, c in s["claims"].items():
                if c["expires"] > now:
                    claims.append({"sample": s["name"], "agent": a, "minutes_left": round((c["expires"] - now) / 60)})
        return {"total": len(st["samples"]), "by_status": by_status, "by_doc_type": by_type,
                "active_claims": claims, "redundancy": self.cfg.redundancy,
                "output_folder": str(self.cfg.out_dir)}

    def list_samples(self, status: str | None = None, doc_type: str | None = None,
                     limit: int = 50, offset: int = 0) -> dict:
        st = self.store.snapshot()
        now = _now()
        rows = []
        for s in st["samples"].values():
            if status and s["status"] != status:
                continue
            if doc_type and s["doc_type"] != doc_type.upper():
                continue
            rows.append({"sample_id": s["sid"], "name": s["name"], "doc_type": s["doc_type"], "status": s["status"],
                         "claimed_by": sorted(a for a, c in s["claims"].items() if c["expires"] > now),
                         "submissions": len(s["submissions"])})
        limit = max(1, min(int(limit), 200))
        return {"total": len(rows), "offset": offset, "samples": rows[offset: offset + limit]}

    def _payload(self, s: dict, agent: str, resumed: bool) -> dict:
        draft = s["drafts"].get(agent) or new_draft()
        done = sorted(draft["fields"])
        keys = self._keys(s["doc_type"])
        return {
            "sample_id": s["sid"], "name": s["name"], "doc_type": s["doc_type"],
            "sides": self._sides(s["doc_type"]),
            "fields_to_label": list(keys), "already_set": done,
            "still_missing": [k for k in keys if k not in draft["fields"]],
            "resumed": resumed, "lease_minutes": self.cfg.lease_minutes,
            "next_step": ("view_sample for every side in 'sides' (zoom with region=[x0,y0,x1,y1] for small text), "
                          "then set_fields / mark_unreadable, validate_sample, submit_label. "
                          "Call get_schema(doc_type) for format hints."),
        }

    def next_sample(self, agent: str | None = None, doc_type: str | None = None, name: str | None = None) -> dict:
        agent = self._agent(agent)
        now = _now()
        with self.store.tx() as st:
            self._expire(st, now)
            wanted = (name or "").strip().lower()
            for s in st["samples"].values():                         # resume what this agent already holds
                c = s["claims"].get(agent)
                if c and s["status"] == "waiting" and (not wanted or wanted in (s["name"].lower(), s["sid"].lower())):
                    c["expires"] = now + self.cfg.lease_minutes * 60
                    return self._payload(s, agent, resumed=True)
            for s in st["samples"].values():
                if s["status"] != "waiting":
                    continue
                if doc_type and s["doc_type"] != doc_type.upper():
                    continue
                if wanted and wanted not in (s["name"].lower(), s["sid"].lower()):
                    continue
                if agent in {x["agent"] for x in s["submissions"]}:
                    continue
                if len(self._active_others(s, agent, now)) + len(s["submissions"]) >= self.cfg.redundancy:
                    continue
                s["claims"][agent] = {"at": now, "expires": now + self.cfg.lease_minutes * 60}
                s["drafts"].setdefault(agent, new_draft())
                return self._payload(s, agent, resumed=bool(s["drafts"][agent]["fields"]))
        return {"sample_id": None, "message": "No sample is available for you right now (queue empty, or the "
                                              "remaining ones are claimed or already done by you)."}

    def release_sample(self, sample_id: str, agent: str | None = None, discard_draft: bool = False) -> dict:
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            s["claims"].pop(agent, None)
            if discard_draft:
                s["drafts"].pop(agent, None)
        return {"released": s["sid"], "draft_kept": not discard_draft}

    # ================================================================== viewing
    def view_sample(self, sample_id: str, side: str = "front", region=None, max_side: int = 1600,
                    agent: str | None = None) -> tuple[bytes, dict]:
        agent = self._agent(agent)
        side = (side or "front").lower()
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            if side not in self._sides(s["doc_type"]):
                raise ToolError(f"{s['doc_type']} has these sides: {self._sides(s['doc_type'])}")
            draft = self._draft(st, s, agent)
            path = s["front"] if side == "front" else s["back"]
            try:
                data, info = images.render_view(path, region=region, max_side=max_side)
            except ValueError as e:
                raise ToolError(str(e))
            except FileNotFoundError:
                raise ToolError(f"the image file for {s['name']} ({side}) is missing on the server")
            draft["views"][side] = draft["views"].get(side, 0) + 1
            if region is not None:
                draft["zoom_views"] += 1
        info.update({"sample_id": s["sid"], "side": side,
                     "views_so_far": dict(draft["views"]), "zoom_views": draft["zoom_views"]})
        return data, info

    # ================================================================== editing
    @staticmethod
    def _check_value(key: str, raw: Any) -> tuple[str | None, str | None]:
        """-> (clean_value, error). Rejects the usual ways of dodging the readable/unreadable rule."""
        if not isinstance(raw, str):
            return None, "value must be text"
        v = raw.strip()
        if not v:
            return None, "empty value: type the printed text or use mark_unreadable"
        if any(c in v for c in "\r\n\t"):
            return None, "line breaks and tabs are not allowed"
        if UNREADABLE in v:
            if v == UNREADABLE:
                return None, "use mark_unreadable(sample_id, key, reason) so the reason is recorded"
            return None, "mixing text and [UNREADABLE] is not allowed: the whole field is either exact text or unreadable"
        if v.lower() in EVASIONS:
            return None, "that is not printed text: use mark_unreadable with a reason"
        if _UNCERTAIN_RE.search(v):
            return None, "characters like ? … ... mean you are unsure: if any character is uncertain the WHOLE field is unreadable"
        if _NOTE_RE.search(v):
            return None, "brackets look like a note, not printed text: type only what is printed (or mark_unreadable)"
        return v, None

    def set_fields(self, sample_id: str, values: dict | None = None, unreadable: dict | None = None,
                   format_exceptions: dict | None = None, agent: str | None = None) -> dict:
        agent = self._agent(agent)
        values = values or {}
        unreadable = unreadable or {}
        format_exceptions = format_exceptions or {}
        accepted, with_exception, rejected, pending, marked = [], [], {}, {}, []
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            keys = set(self._keys(s["doc_type"]))
            for key, raw in values.items():
                if key not in keys:
                    rejected[key] = f"unknown field for {s['doc_type']}; valid keys: {sorted(keys)}"
                    continue
                v, err = self._check_value(key, raw)
                if err:
                    rejected[key] = err
                    continue
                draft["fields"][key] = v
                draft["reasons"].pop(key, None)
                draft["notes"].pop(key, None)
                warn = S.field_warnings(s["doc_type"], {key: v}).get(key)
                exc = (format_exceptions.get(key) or "").strip()
                if warn and len(exc) >= 8:
                    draft["exceptions"][key] = exc
                    draft["pending_format"].pop(key, None)
                    with_exception.append(key)
                elif warn:
                    draft["pending_format"][key] = warn
                    draft["exceptions"].pop(key, None)
                    pending[key] = (f"{warn} If this is exactly what is printed, call again with "
                                    f"format_exceptions={{'{key}': '<why it differs>'}}; otherwise re-read it.")
                else:
                    draft["pending_format"].pop(key, None)
                    draft["exceptions"].pop(key, None)
                    accepted.append(key)
            if s["doc_type"] == "ID":
                # A line that passes the format but fails an MRZ check digit has no format warning to except,
                # yet the agent may be certain it is what is printed (cards with a wrong check digit exist).
                # Then the exception is the honest way out; the label goes to human review.
                for key in self._mrz_exceptions(draft, format_exceptions):
                    if key in accepted:
                        accepted.remove(key)
                    if key not in with_exception:
                        with_exception.append(key)
            for key, why in format_exceptions.items():                 # exception for an already stored value
                if key in values or key not in draft["pending_format"]:
                    continue
                if len((why or "").strip()) >= 8:
                    draft["exceptions"][key] = why.strip()
                    draft["pending_format"].pop(key, None)
                    with_exception.append(key)
            for key, reason in unreadable.items():
                res = self._mark(s, draft, key, reason, None)
                (rejected.__setitem__(key, res["error"]) if "error" in res else marked.append(key))
            return {"sample_id": s["sid"], "accepted": accepted, "accepted_with_format_exception": with_exception,
                    "needs_format_decision": pending, "marked_unreadable": marked, "rejected": rejected,
                    "progress": self._progress(s, draft)}

    @staticmethod
    def _mrz_exceptions(draft: dict, format_exceptions: dict) -> list[str]:
        """Record format exceptions for MRZ lines while the MRZ fails a blocking check. Returns the keys."""
        lines = [draft["fields"].get(k) for k in mrz_tools.LINE_KEYS]
        if any(v is None or v == UNREADABLE for v in lines):
            return []
        if not mrz_tools.check_lines(tuple(lines))["problems"]:          # type: ignore[arg-type]
            return []
        done = []
        for key in mrz_tools.LINE_KEYS:
            why = (format_exceptions.get(key) or "").strip()
            if len(why) >= 8 and key not in draft["pending_format"]:
                draft["exceptions"][key] = why
                done.append(key)
        return done

    def _mark(self, s: dict, draft: dict, key: str, reason: str, note: str | None) -> dict:
        if key not in self._keys(s["doc_type"]):
            return {"error": f"unknown field for {s['doc_type']}"}
        reason = (reason or "").strip().lower()
        if reason not in REASONS:
            return {"error": f"reason must be one of {list(REASONS)}"}
        if reason == "other" and len((note or "").strip()) < 5:
            return {"error": "reason 'other' needs a note (at least 5 characters)"}
        draft["fields"][key] = UNREADABLE
        draft["reasons"][key] = reason
        if note:
            draft["notes"][key] = note.strip()
        draft["pending_format"].pop(key, None)
        draft["exceptions"].pop(key, None)
        cond = REASON_TO_CONDITION.get(reason)
        if cond and cond not in draft["conditions"]:
            draft["conditions"].append(cond)
        return {"ok": True}

    def mark_unreadable(self, sample_id: str, key: str, reason: str, note: str | None = None,
                        agent: str | None = None) -> dict:
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            res = self._mark(s, draft, key, reason, note)
            if "error" in res:
                raise ToolError(res["error"])
            return {"sample_id": s["sid"], "key": key, "value": UNREADABLE, "reason": reason,
                    "progress": self._progress(s, draft)}

    def set_conditions(self, sample_id: str, tags: list[str], agent: str | None = None) -> dict:
        agent = self._agent(agent)
        bad = [t for t in tags if t not in CONDITION_TAGS]
        if bad:
            raise ToolError(f"unknown tags {bad}; allowed: {list(CONDITION_TAGS)}")
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            draft["conditions"] = list(dict.fromkeys(tags))
            return {"sample_id": s["sid"], "conditions": draft["conditions"]}

    # ================================================================== validation
    def _progress(self, s: dict, draft: dict) -> dict:
        keys = self._keys(s["doc_type"])
        fields = draft["fields"]
        resolved = [k for k in keys if k in fields and k not in draft["pending_format"]]
        unread = [k for k in resolved if fields[k] == UNREADABLE]
        return {"resolved": len(resolved), "total": len(keys), "readable": len(resolved) - len(unread),
                "unreadable": len(unread), "missing": [k for k in keys if k not in fields]}

    def _mrz(self, s: dict, draft: dict) -> dict | None:
        if s["doc_type"] != "ID":
            return None
        lines = [draft["fields"].get(k) for k in mrz_tools.LINE_KEYS]
        if any(v is None or v == UNREADABLE for v in lines):
            return {"checked": False, "note": "MRZ not fully readable: no MRZ check"}
        res = mrz_tools.check_lines(tuple(lines))             # type: ignore[arg-type]
        res["checked"] = True
        return res

    def _report(self, s: dict, draft: dict) -> dict:
        keys = self._keys(s["doc_type"])
        prog = self._progress(s, draft)
        blocking: list[str] = []
        if prog["missing"]:
            blocking.append(f"fields not set yet: {prog['missing']}")
        if draft["pending_format"]:
            blocking.append("fields with a format warning need a decision (re-read, or give a format_exception): "
                            + ", ".join(draft["pending_format"]))
        if self.cfg.require_view:
            missing_views = [sd for sd in self._sides(s["doc_type"]) if not draft["views"].get(sd)]
            if missing_views:
                blocking.append(f"you have not viewed these sides with view_sample: {missing_views}")
        mrz = self._mrz(s, draft)
        excepted = any(k in draft["exceptions"] for k in mrz_tools.LINE_KEYS)
        if mrz and mrz.get("checked") and mrz["problems"] and not excepted:
            msg = "MRZ is invalid: " + "; ".join(mrz["problems"]) + "."
            if mrz.get("repair_hint"):
                msg += " " + mrz["repair_hint"]
            # Never steer the agent towards [UNREADABLE] or towards "fixing" a character it read correctly:
            # the label must be what is printed.
            blocking.append(msg + " Zoom into the MRZ and type exactly what is printed. If you are sure your "
                                  "line matches the card, keep it and give a format_exception for that line "
                                  "(it goes to human review). Use mark_unreadable only if you really cannot "
                                  "read the line.")
        flags: list[dict] = []
        if mrz and mrz.get("checked") and mrz.get("warnings") and not excepted:
            flags += [{"kind": "mrz_check_digit", "check": "composite_check_digit", "fields": ["mrz_line2"],
                       "detail": w} for w in mrz["warnings"]]
        if s["doc_type"] == "ID":
            flags += [{"kind": "mrz_consistency", **f} for f in mrz_tools.consistency(draft["fields"])]
        total = len(keys)
        if prog["resolved"] == total and prog["unreadable"] == total:
            flags.append({"kind": "all_unreadable", "detail": "every field is unreadable: is this a usable "
                                                              "document? If not, use skip_sample with a reason."})
        elif total and prog["unreadable"] / total > self.cfg.max_unreadable_ratio:
            flags.append({"kind": "many_unreadable", "detail": f"{prog['unreadable']} of {total} fields unreadable: "
                                                               "zoom in before giving up on a field."})
        if draft["exceptions"]:
            flags.append({"kind": "format_exceptions", "detail": "format exceptions will be sent to human review: "
                                                                 + ", ".join(draft["exceptions"])})
        return {"progress": prog, "blocking": blocking, "flags": flags, "mrz": mrz}

    def validate_sample(self, sample_id: str, agent: str | None = None) -> dict:
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            rep = self._report(s, draft)
            states = {}
            for k in self._keys(s["doc_type"]):
                if k in draft["pending_format"]:
                    states[k] = "needs_format_decision"
                elif k not in draft["fields"]:
                    states[k] = "empty"
                elif draft["fields"][k] == UNREADABLE:
                    states[k] = f"unreadable ({draft['reasons'].get(k, '?')})"
                else:
                    states[k] = "readable"
            return {"sample_id": s["sid"], "field_states": states, "progress": rep["progress"],
                    "views": dict(draft["views"]), "zoom_views": draft["zoom_views"], "mrz": rep["mrz"],
                    "format_warnings": dict(draft["pending_format"]), "flags": rep["flags"],
                    "blocking": rep["blocking"], "ready_to_submit": not rep["blocking"],
                    "conditions": draft["conditions"]}

    def mrz_check(self, sample_id: str | None = None, line1: str | None = None, line2: str | None = None,
                  line3: str | None = None, agent: str | None = None) -> dict:
        if any(v is not None for v in (line1, line2, line3)):
            if None in (line1, line2, line3):
                raise ToolError("pass all three lines (line1, line2, line3) or none to check the saved draft")
            res = mrz_tools.check_lines((line1.strip(), line2.strip(), line3.strip()))   # type: ignore[union-attr]
            return {"checked": "the three lines you passed", **res}
        if not sample_id:
            raise ToolError("pass sample_id (to check your saved MRZ lines) or line1/line2/line3")
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            if s["doc_type"] != "ID":
                raise ToolError("only ID documents have an MRZ")
            out = self._mrz(s, draft) or {}
            out["consistency_with_visual_fields"] = mrz_tools.consistency(draft["fields"]) or "no mismatch found"
            return out

    # ================================================================== submit
    def submit_label(self, sample_id: str, agent: str | None = None, confirm: bool = False,
                     notes: str | None = None) -> dict:
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            draft = self._draft(st, s, agent)
            rep = self._report(s, draft)
            if rep["blocking"]:
                return {"status": "blocked", "sample_id": s["sid"], "blocking": rep["blocking"],
                        "hint": "Fix these, then call submit_label again (validate_sample shows everything)."}
            if rep["flags"] and not confirm:
                return {"status": "needs_confirmation", "sample_id": s["sid"], "flags": rep["flags"],
                        "hint": "Re-inspect the flagged fields (zoom in). If your values are right, call "
                                "submit_label again with confirm=true; otherwise fix them first."}
            submission = {"agent": agent, "at": _stamp(), "fields": dict(draft["fields"]),
                          "reasons": dict(draft["reasons"]), "notes": dict(draft["notes"]),
                          "exceptions": dict(draft["exceptions"]), "conditions": list(draft["conditions"]),
                          "views": dict(draft["views"]), "zoom_views": draft["zoom_views"],
                          "flags": rep["flags"], "agent_note": (notes or "").strip() or None}
            s["submissions"].append(submission)
            s["claims"].pop(agent, None)
            s["drafts"].pop(agent, None)
            if len(s["submissions"]) < self.cfg.redundancy:
                return {"status": "submitted_waiting_for_second_agent", "sample_id": s["sid"],
                        "message": f"Saved. Redundancy is {self.cfg.redundancy}: a different agent must label "
                                   "this sample too; the label is written when both agree."}
            return self._finalize(s)

    def _disagreements(self, subs: list[dict]) -> list[dict]:
        out = []
        for key in subs[0]["fields"]:
            vals = {x["agent"]: x["fields"].get(key) for x in subs}
            first = subs[0]["fields"].get(key)
            ok = all((v == UNREADABLE) == (first == UNREADABLE) and (v == UNREADABLE or same(v, first))
                     for v in vals.values())
            if not ok:
                out.append({"field": key, "values": vals})
        return out

    def _finalize(self, s: dict) -> dict:
        subs = s["submissions"][: self.cfg.redundancy]
        if len(subs) > 1:
            dis = self._disagreements(subs)
            if dis:
                s["status"], s["disagreements"] = "needs_human", dis
                return {"status": "needs_human", "sample_id": s["sid"],
                        "message": f"The agents disagree on {len(dis)} field(s): a human must resolve it "
                                   "(alg-mcp resolve).", "fields": [d["field"] for d in dis]}
        try:
            result = self._write_outputs(s, subs)
        except (ValueError, OSError, ToolError) as e:
            # Keep the agents' work and park the sample for a human (raising would roll the state back).
            s["status"], s["last_error"] = "needs_human", str(e)
            return {"status": "needs_human", "sample_id": s["sid"],
                    "message": f"Your answers are saved, but the label file could not be written ({e}). "
                               "A human must run: alg-mcp resolve."}
        s["status"], s["result"], s["last_error"] = "done", result, None
        return {"status": "done", "sample_id": s["sid"], "label_base": result["base"],
                "review_state": result["review_state"], "review_reasons": result["review_reasons"],
                "written_to": str(self.cfg.out_dir)}

    # ================================================================== outputs
    def _exists(self, directory: Path, base: str) -> bool:
        return any((directory / f"{base}{suf}").exists()
                   for suf in (".png", ".jpg", ".label.json", ".label.sidecar.json", ".label.ai.json"))

    def _pick_base(self, s: dict) -> str:
        m = _SOURCE_RE.match(s["name"])
        if m and m.group(1).upper() == s["doc_type"]:
            base = NM.doc_name(s["doc_type"], int(m.group(2)))
            if self._exists(self.cfg.out_dir, base):
                raise ToolError(f"{base} already exists in {self.cfg.out_dir}: not overwriting")
            return base
        while True:
            base = NM.doc_name(s["doc_type"], NM.take_next(self.cfg.workspace, s["doc_type"]))
            if not self._exists(self.cfg.out_dir, base) and not self._exists(self.cfg.human_dir, base):
                return base

    def _write_outputs(self, s: dict, subs: list[dict]) -> dict:
        cfg = self.cfg
        final = subs[0]
        agents = [x["agent"] for x in subs]
        labeler = "ai:" + "+".join(agents)
        base = self._pick_base(s)
        tmp = cfg.workspace / ".alg_mcp_tmp" / s["sid"]
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            sess = LabelSession(s["doc_type"], base, s["front"], s["back"], tmp, labeler=labeler)
            for key in self._keys(s["doc_type"]):
                v = final["fields"][key]
                sess.mark_unreadable(key) if v == UNREADABLE else sess.set_value(key, v)
            conditions = []
            for x in subs:
                conditions += [c for c in x["conditions"] if c in CONDITION_TAGS and c not in conditions]
            sess.conditions = conditions
            sess.warnings_overridden = sorted(final["exceptions"])
            sess.generate(merge_width=cfg.merge_width, merge_gap=cfg.merge_gap, jpeg=cfg.jpeg, labeler=labeler)
            truth = sess.truth_values()
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            for f in (tmp / "labeled").iterdir():
                shutil.move(str(f), str(cfg.out_dir / f.name))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        audit, state, reasons = self._audit(s, subs, truth, base, labeler)
        audit_path = cfg.out_dir / f"{base}.label.ai.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"base": base, "dir": str(cfg.out_dir), "audit": str(audit_path),
                "agents": agents, "review_state": state, "review_reasons": reasons}

    def _audit(self, s: dict, subs: list[dict], truth: dict[str, str], base: str,
               labeler: str) -> tuple[dict, str, list[str]]:
        final = subs[0]
        keys = self._keys(s["doc_type"])
        reasons: list[str] = []
        fields: dict[str, dict] = {}
        unread = 0
        for key in keys:
            v = final["fields"][key]
            entry: dict[str, Any] = {"value": v, "status": "unreadable" if v == UNREADABLE else "readable"}
            if v == UNREADABLE:
                unread += 1
                entry["reason"] = final["reasons"].get(key)
                if final["notes"].get(key):
                    entry["note"] = final["notes"][key]
            if key in final["exceptions"]:
                entry["format_exception"] = final["exceptions"][key]
            if len(subs) > 1:
                entry["agents_agree"] = True              # disagreements never reach this point
            if key in truth:
                if v == UNREADABLE:
                    entry["reference"] = "agent_unreadable"
                    reasons.append(f"{key}: agent says unreadable but a reference value exists (human decides)")
                elif same(v, truth[key]):
                    entry["reference"] = "agrees"
                else:
                    entry["reference"] = "disagrees"
                    reasons.append(f"{key}: disagrees with the generator's reference value")
            fields[key] = entry
        if final["exceptions"]:
            reasons.append("format exception(s): " + ", ".join(sorted(final["exceptions"])))
        if not truth and len(subs) < 2:
            reasons.append("no reference values and only one agent: needs human review")
        soft = [f for f in final["flags"]
                if f.get("kind") in ("mrz_check_digit", "mrz_consistency", "many_unreadable", "all_unreadable")]
        reasons += [f"{f['kind']}: {f.get('detail', '')}" for f in soft]
        state = "auto_ok" if not reasons else "needs_review"
        audit = {
            "base": base, "photo": f"{base}.png", "doc_type": s["doc_type"], "labeler": labeler,
            "created_at": _stamp(), "redundancy": self.cfg.redundancy,
            "agents": [x["agent"] for x in subs], "reference_available": bool(truth),
            "fields": fields, "conditions": final["conditions"],
            "diligence": {x["agent"]: {"views": x["views"], "zoom_views": x["zoom_views"]} for x in subs},
            "agent_notes": {x["agent"]: x["agent_note"] for x in subs if x.get("agent_note")},
            "unreadable_count": unread,
            "review": {"state": state, "reasons": reasons, "decided_by": None, "decided_at": None},
        }
        return audit, state, reasons

    def skip_sample(self, sample_id: str, reason: str, agent: str | None = None) -> dict:
        agent = self._agent(agent)
        reason = (reason or "").strip()
        if len(reason) < 5:
            raise ToolError("skip needs a reason (at least 5 characters)")
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            self._draft(st, s, agent)
            tmp = self.cfg.workspace / ".alg_mcp_tmp" / (s["sid"] + "-skip")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            try:
                sess = LabelSession(s["doc_type"], s["name"], s["front"], s["back"], tmp, labeler=f"ai:{agent}")
                sess.skip(reason, labeler=f"ai:{agent}")
                self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
                for f in (tmp / "labeled").iterdir():
                    shutil.move(str(f), str(self.cfg.out_dir / f.name))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            s["status"], s["skip"] = "skipped", {"agent": agent, "reason": reason, "at": _stamp()}
            s["claims"].pop(agent, None)
            s["drafts"].pop(agent, None)
            return {"status": "skipped", "sample_id": s["sid"], "reason": reason}

    def get_truth(self, sample_id: str, agent: str | None = None) -> dict:
        if not self.cfg.expose_truth:
            raise ToolError("reference values are hidden from agents on purpose (blind labeling). "
                            "The server compares your labels with them afterwards.")
        agent = self._agent(agent)
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            self._draft(st, s, agent)
            sess = LabelSession(s["doc_type"], s["name"], s["front"], s["back"], self.cfg.workspace)
            return {"sample_id": s["sid"], "reference": sess.truth_values()}

    # ================================================================== admin (CLI only, never exposed as tools)
    def add_sample(self, doc_type: str, front: str, back: str | None = None, name: str | None = None) -> dict:
        doc_type = doc_type.upper()
        if doc_type not in DOC_TYPES:
            raise ValueError(f"doc type must be one of {list(DOC_TYPES)}")
        front_p = Path(front).expanduser().resolve()
        back_p = Path(back).expanduser().resolve() if back else None
        for p in (front_p, back_p):
            if p is not None and (not p.is_file() or not images.is_image_file(p)):
                raise ValueError(f"not an image file (png/jpg): {p}")
        if doc_type == "ID" and back_p is None:
            raise ValueError("ID samples need a front and a back image")
        if doc_type != "ID" and back_p is not None:
            raise ValueError(f"{doc_type} samples have a single image")
        stem = name or (_PAIR_RE.match(front_p.stem).group("stem") if _PAIR_RE.match(front_p.stem) else front_p.stem)
        sample = new_sample(doc_type, stem, str(front_p), str(back_p) if back_p else None)
        with self.store.tx() as st:
            if sample["sid"] in st["samples"]:
                return {"sid": sample["sid"], "added": False}
            st["samples"][sample["sid"]] = sample
        return {"sid": sample["sid"], "added": True, "name": stem}

    def add_paths(self, paths: list[str], doc_type: str | None = None) -> dict:
        """Add files/folders. Types come from the file names (ID_01_front.png, TAX_03.png, LIC_07.png)
        unless ``doc_type`` is given. ID images are paired by *_front / *_back."""
        files: list[Path] = []
        for p in paths:
            pp = Path(p).expanduser()
            files += sorted(f for f in pp.rglob("*") if f.is_file() and images.is_image_file(f)) if pp.is_dir() else [pp]
        added, skipped, problems = 0, 0, []
        pairs: dict[tuple[str, str], dict[str, Path]] = {}
        todo: list[tuple[str, str, str | None, str | None]] = []       # (doc_type, front, back, name)
        for f in files:
            pair = _PAIR_RE.match(f.stem)
            src = _SOURCE_RE.match(f.stem)
            kind = (doc_type or (src.group(1) if src else "")).upper()
            if kind == "ID":
                if not pair:
                    problems.append(f"{f.name}: ID images must be named *_front / *_back")
                    continue
                pairs.setdefault((str(f.parent), pair.group("stem")), {})[pair.group("side").lower()] = f
            elif kind in DOC_TYPES:
                todo.append((kind, str(f), None, None))
            else:
                problems.append(f"{f.name}: cannot tell the document type: pass --type ID|TAX|LIC")
        for (_, stem), sides in pairs.items():
            if set(sides) != {"front", "back"}:
                problems.append(f"{stem}: needs both _front and _back (found {sorted(sides)})")
            else:
                todo.append(("ID", str(sides["front"]), str(sides["back"]), stem))
        natural = lambda t: [int(x) if x.isdigit() else x.lower()                       # noqa: E731
                             for x in re.split(r"(\d+)", t[3] or Path(t[1]).stem)]
        for kind, front, back, name in sorted(todo, key=natural):       # a predictable queue order
            try:
                r = self.add_sample(kind, front, back, name)
                added += r["added"]
                skipped += not r["added"]
            except ValueError as e:
                problems.append(f"{Path(front).name}: {e}")
        return {"added": added, "already_registered": skipped, "problems": problems}

    def resolve(self, sample_id: str, pick_agent: str) -> dict:
        """Human decision for a needs_human sample: take one agent's version."""
        with self.store.tx() as st:
            s = self._find(st, sample_id)
            if s["status"] != "needs_human":
                raise ToolError(f"{s['name']} is not waiting for a human decision (status {s['status']})")
            chosen = [x for x in s["submissions"] if x["agent"] == pick_agent]
            if not chosen:
                raise ToolError(f"no submission from {pick_agent}; have {[x['agent'] for x in s['submissions']]}")
            ordered = chosen + [x for x in s["submissions"] if x["agent"] != pick_agent]
            s["status"] = "waiting"
            result = self._write_outputs(s, ordered[:1])
            s["status"], s["result"], s["disagreements"] = "done", result, []
            return {"status": "done", "base": result["base"]}
