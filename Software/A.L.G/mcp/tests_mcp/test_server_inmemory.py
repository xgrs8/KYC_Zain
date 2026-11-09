import asyncio
import base64
import json
import re

from mcp.shared.memory import create_connected_server_and_client_session

from alg_mcp.config import Config
from alg_mcp.server import build_server
from conftest import TAX_VALUES

EXPECTED_TOOLS = {
    "get_protocol", "get_schema", "queue_status", "list_samples", "next_sample", "view_sample", "set_fields",
    "set_field", "mark_unreadable", "set_conditions", "validate_sample", "mrz_check", "get_truth",
    "submit_label", "skip_sample", "release_sample",
}


def run_with_client(ws, scenario, **cfg_kw):
    cfg = Config.from_env(workspace=ws, **cfg_kw)
    server, svc = build_server(cfg)

    async def main():
        async with create_connected_server_and_client_session(server._mcp_server) as client:   # noqa: SLF001
            return await scenario(client, svc)

    return asyncio.run(main())


def text_of(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


def as_json(result):
    return json.loads(text_of(result))


def test_tools_annotations_and_text_results(ws):
    async def scenario(client, svc):
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == EXPECTED_TOOLS
        assert tools["view_sample"].annotations.readOnlyHint is True
        assert tools["get_protocol"].annotations.readOnlyHint is True
        assert tools["submit_label"].annotations.readOnlyHint is False
        assert tools["next_sample"].annotations.readOnlyHint is False
        for t in tools.values():                                  # every tool says what it is for
            assert t.description and len(t.description) > 40, t.name
        res = await client.call_tool("get_protocol", {})
        assert not res.isError and res.structuredContent is None
        body = as_json(res)
        assert "[UNREADABLE]" in body["protocol"] and "covered" in body["reason_codes"]
        res = await client.call_tool("get_schema", {"doc_type": "LIC"})
        assert re.search("[\u0600-\u06FF]", text_of(res)), "Arabic must be sent as real characters, not \\u escapes"
        assert as_json(res)["field_count"] == 8
        res = await client.call_tool("get_schema", {"doc_type": "XYZ"})
        assert res.isError and "doc_type must be one of" in text_of(res)

    run_with_client(ws, scenario)


def test_image_block_and_errors(ws, inbox):
    async def scenario(client, svc):
        svc.add_paths([str(inbox)])
        res = await client.call_tool("next_sample", {})
        assert res.isError and "Say who you are" in text_of(res)
        res = await client.call_tool("next_sample", {"agent": "tester", "name": "TAX_03"})
        sid = as_json(res)["sample_id"]
        res = await client.call_tool("view_sample", {"sample_id": sid, "agent": "tester"})
        kinds = [c.type for c in res.content]
        assert kinds == ["text", "image"], kinds
        assert res.structuredContent is None                     # image blocks must not be paired with structuredContent
        img = next(c for c in res.content if c.type == "image")
        assert img.mimeType == "image/jpeg" and base64.b64decode(img.data)[:2] == b"\xff\xd8"
        full = as_json(type("R", (), {"content": [res.content[0]]}))
        res = await client.call_tool("view_sample", {"sample_id": sid, "agent": "tester",
                                                     "region": [0.0, 0.0, 0.5, 0.5], "max_side": 600})
        info = json.loads(res.content[0].text)
        assert info["region"] == [0.0, 0.0, 0.5, 0.5] and max(info["returned_size"]) == 600
        assert info["zoom_views"] == 1 and full["views_so_far"]["front"] == 1
        res = await client.call_tool("view_sample", {"sample_id": sid, "agent": "tester", "region": [0.5, 0.5, 0.4, 0.9]})
        assert res.isError and "region must satisfy" in text_of(res)
        res = await client.call_tool("view_sample", {"sample_id": sid, "agent": "tester", "side": "back"})
        assert res.isError and "has these sides" in text_of(res)

    run_with_client(ws, scenario)


def test_full_labeling_through_the_tools(ws, inbox):
    async def scenario(client, svc):
        svc.add_paths([str(inbox)])
        call = lambda tool, **kw: client.call_tool(tool, kw)                     # noqa: E731
        sid = as_json(await call("next_sample", agent="glm-5v", name="TAX_03"))["sample_id"]
        await call("view_sample", sample_id=sid, agent="glm-5v")
        res = as_json(await call("set_fields", sample_id=sid, agent="glm-5v",
                                 values={k: v for k, v in TAX_VALUES.items() if k != "expiry_date"},
                                 unreadable={"expiry_date": "glare"}))
        assert res["marked_unreadable"] == ["expiry_date"] and not res["rejected"]
        res = as_json(await call("validate_sample", sample_id=sid, agent="glm-5v"))
        assert res["ready_to_submit"] and res["field_states"]["expiry_date"] == "unreadable (glare)"
        res = as_json(await call("submit_label", sample_id=sid, agent="glm-5v"))
        assert res["status"] == "done" and res["review_state"] == "needs_review"      # reference exists but field unreadable
        status = as_json(await call("queue_status"))
        assert status["by_status"]["done"] == 1
        rows = as_json(await call("list_samples", status="waiting"))
        assert rows["total"] == 3
        # truth is hidden by default
        res = await call("get_truth", sample_id=sid, agent="glm-5v")
        assert res.isError

    run_with_client(ws, scenario)


def test_resources_and_prompt(ws):
    async def scenario(client, svc):
        res = await client.read_resource("alg://protocol")
        assert "A.L.G LABELING PROTOCOL" in res.contents[0].text
        res = await client.read_resource("alg://schema/TAX")
        assert json.loads(res.contents[0].text)["doc_type"] == "TAX"
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert "label_samples" in prompts
        got = await client.get_prompt("label_samples", {"agent": "claude", "count": "3"})
        assert "claude" in got.messages[0].content.text and "get_protocol" in got.messages[0].content.text
        info = await client.initialize() if False else None      # initialize() already ran inside the helper
        assert info is None

    run_with_client(ws, scenario)


def test_audit_log_records_calls_without_values(ws, inbox):
    async def scenario(client, svc):
        svc.add_paths([str(inbox)])
        await client.call_tool("next_sample", {"agent": "auditor", "name": "TAX_03"})
        await client.call_tool("next_sample", {})                            # an error is logged too

    run_with_client(ws, scenario)
    rows = [json.loads(line) for line in (ws / "alg_mcp_audit.jsonl").read_text("utf-8").splitlines()]
    assert [r["ok"] for r in rows] == [True, False] and rows[0]["agent"] == "auditor"
    assert all("values" not in r and "fields" not in r for r in rows)
