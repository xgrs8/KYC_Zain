# A.L.G MCP server

Lets AI agents (Claude, ChatGPT, GLM, local models) label documents **through the same rules as the A.L.G desktop app**.
It reuses the app's own label engine (`alg.label.LabelSession`), schemas (`rfalib.schema`) and MRZ code (`rfalib.mrz`), so
the files it writes are byte-for-byte the same kind the app writes. It does **not** need PySide6 and it does **not**
modify any existing A.L.G file.

```
agent (Claude / ChatGPT / GLM / local model)
   │  MCP: stdio (local)  or  HTTPS (remote, via Tailscale Funnel)
   ▼
alg_mcp  ──►  rules (readable / [UNREADABLE]), claims, MRZ checks, blind reference check
   │
   ▼
<workspace>/ai_labeled/  X.png  X.label.json  X.label.sidecar.json  X.label.ai.json     (review first)
   │  alg-mcp review / approve / promote
   ▼
<workspace>/labeled/     what you train on (the same folder the desktop app writes to)
```

## Design rules (why it is built this way)

1. **AI labels never go straight into your dataset.** They land in `ai_labeled/`. You promote them after review/audit.
2. **Every field is the exact printed text or `[UNREADABLE]`.** The server rejects `N/A`, `?`, `…`, partial text, `[blurry]` notes and
   `text [UNREADABLE]`; the agent must use `mark_unreadable` with a reason code (`covered, cut_off, blur, glare, low_light, too_small, damaged, other`).
3. **The app's format warnings are blocking** (e.g. `capital` must be `1,234,567 دينار عراقي`). An agent may still type what is
   printed, but it must give a `format_exception` reason; it is logged in `warnings_overridden` exactly like the app does, and the label goes to human review.
4. **Blind labeling.** If a `*.truth.json` (generator reference values) exists, agents never see it. The server compares afterwards and
   writes `agrees / disagrees / agent_unreadable` per field into `X.label.ai.json`. Disagreements and "unreadable although a reference exists" go to human review.
5. **MRZ is checked in code** (30 chars, check digits, and consistency with the visual fields). The server never reveals the "right" MRZ; it only says what is wrong,
   and it never asks an agent to change a character it read correctly or to mark a readable line unreadable: the label is what is printed. Printed cards
   can carry a wrong check digit (81 of the 90 IDs in this project's ground truth have a composite check digit that the ICAO formula does not give), so a
   composite-only mismatch asks for a re-check and a confirmation, and an agent that is certain of a line that fails another check keeps it with a `format_exception`.
6. **Known formats of a print run.** A truth sidecar may declare `known_deviations`: the formats a print run really uses (this project's
   print run has 13-digit national numbers, «ل.س» on the licences, a placeholder in the ID surname row, a non-ICAO composite digit ...;
   see [the A.L.G README](../README.md#labeling-the-photographed-print-run)). A value that fits the declared format needs no
   `format_exception`, and the MRZ checks the print run fails on every card (`names_vs_mrz`, `composite_check_digit`) are not applied.
   Only patterns and check names are used, never a value, so labeling stays blind; the audit lists the fields that used a declared format.
7. **Agents can only touch samples you registered.** No tool takes a file path. You register images with the CLI.
8. **Optional redundancy:** `--redundancy 2` needs two *different* agents to agree on every field; disagreement → `needs_human`.
9. **Audit trail:** `alg_mcp_audit.jsonl` (every tool call, no values), `X.label.ai.json` (agents, reasons, views/zooms, flags), sidecar `labeler = "ai:<agent>"`.

Research behind these choices: vision-language models *guess* instead of abstaining (arXiv 2605.27750), AI labels need a human-checked sample
(alt-test 2501.10970, PAC labeling 2506.10908), AI pre-labels + human review saves 68–81 % of the time (GAIDA 2610.26362), and TeleOCR's own
data engine uses multi-model consensus with humans on disagreements (2608.12898).

## Quick start (Windows PowerShell)

```powershell
cd C:\path\to\KYC_Zain\Software\A.L.G\mcp
python -m venv .venv-mcp
.\.venv-mcp\Scripts\Activate.ps1
python -m pip install -r requirements-mcp.txt   # mcp 1.x + pillow (no PySide6 needed)

python run_alg_mcp.py doctor                   # everything should say PASS
python run_alg_mcp.py add D:\inbox             # names like ID_01_front.png, ID_01_back.png, TAX_02.png, LIC_03.png
python run_alg_mcp.py add --type LIC D:\lic_photos   # any other names: say the type
python run_alg_mcp.py status
```

Optional: `--workspace D:\RFALG` on every command, or `setx ALG_WORKSPACE D:\RFALG`. The default is the app's own workspace.
`run_alg_mcp.py` works from any folder (use its full path in client configs). You can also run `python -m alg_mcp ...` from this `mcp` folder.

## Connect a client

Use the **same agent name every time** per client (`ALG_MCP_AGENT`), it is how claims, redundancy and the audit work.
All clients must use a **vision** model: the agent has to *see* the image.

### Local clients (stdio): Claude Desktop, Cursor, Cline, Claude Code, local models

Claude Desktop `%APPDATA%\Claude\claude_desktop_config.json` (Cline's `cline_mcp_settings.json` and Cursor's `.cursor/mcp.json` use the same shape):

```json
{
  "mcpServers": {
    "alg-labeler": {
      "command": "C:\\path\\to\\KYC_Zain\\Software\\A.L.G\\mcp\\.venv-mcp\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\KYC_Zain\\Software\\A.L.G\\mcp\\run_alg_mcp.py", "serve", "--transport", "stdio"],
      "env": { "ALG_WORKSPACE": "D:\\RFALG", "ALG_MCP_AGENT": "claude-desktop" }
    }
  }
}
```

Claude Code: `claude mcp add alg-labeler -e ALG_MCP_AGENT=claude-code -e ALG_WORKSPACE=D:\RFALG -- C:\path\...\python.exe C:\path\...\run_alg_mcp.py serve --transport stdio`
(flags can differ by version: `claude mcp add --help`).

Codex desktop: open **Settings → MCP servers → Add server**, choose **STDIO**, and use the Python executable and launcher paths above. Set `ALG_MCP_AGENT` to `codex-desktop` and `ALG_WORKSPACE` to the workspace used by the A.L.G desktop app, then save and restart the server. The same server can also be configured through Codex CLI or `config.toml`.

Local models (Ollama, LM Studio, ...) work through any MCP-capable client (Cline, Continue, Open WebUI, ...) as long as the model supports
**tool calling and images**. Small models do better with the single-field tools `set_field` / `mark_unreadable`.
In Cline you can pre-approve the read-only tools so you are not asked on every call.

Start prompt (or pick the MCP prompt `label_samples`): *"You are labeler 'claude-desktop'. Call get_protocol first and follow it exactly. Label the next 5 samples:
next_sample → view_sample (every side, zoom into small text) → set_fields / mark_unreadable → mrz_check (ID) → validate_sample → submit_label.
Never guess; unsure = [UNREADABLE]."*

### Remote clients (HTTPS): ChatGPT, Claude.ai, GLM clients, anything that takes a URL

1. Make a token and start the server (binds to 127.0.0.1 only):
   ```powershell
   python run_alg_mcp.py token --save
   python run_alg_mcp.py serve --transport http --port 8765
   ```
   It prints the URLs. Two ways to authenticate with the **same token**:
   * header `Authorization: Bearer <token>` (Claude Code, Cursor, Cline: `"url": ..., "headers": {...}`)
   * secret URL `https://<machine>.<tailnet>.ts.net/s/<token>/mcp` (clients that only take a URL: ChatGPT "no authentication", Claude.ai)
2. Expose it with Tailscale Funnel (needs Tailscale with HTTPS certificates and Funnel enabled for your tailnet/node):
   ```powershell
   tailscale funnel --bg 8765          # flags differ by version: tailscale funnel --help
   tailscale funnel status             # shows the public https://<machine>.<tailnet>.ts.net URL
   curl https://<machine>.<tailnet>.ts.net/healthz      # -> ok
   # stop sharing:  tailscale funnel reset      (and Ctrl+C the server)
   ```
   `*.ts.net` Host names are accepted automatically; for a custom domain add `--allow-host your.domain`.
3. Add the connector:
   * **ChatGPT** (paid plan, web, Developer mode): Settings → Connectors/Apps → Advanced → Developer mode → create a connector with the secret URL, authentication "none". Menu names change.
     ChatGPT asks you to confirm write calls (read-only tools are marked as such).
   * **Claude.ai**: Settings → Connectors → Add custom connector → the secret URL.
   * **Claude Code**: `claude mcp add --transport http alg-remote https://<host>/mcp --header "Authorization: Bearer <token>"`.
   * **GLM**: through any MCP-capable client with a GLM vision model. **GLM-5.3 (Max) is text-only** (per Artificial Analysis at the time of writing), so it cannot label images;
     use GLM-5V-Turbo, or GLM-5.3-Flash if your provider lists image input.
4. SSE instead of streamable HTTP (very old clients only): `serve --transport sse` and use `.../s/<token>/sse`.

> **Images and ChatGPT.** Users report that ChatGPT's connector does not hand MCP *image* results to the model (they arrive empty).
> Test it: ask ChatGPT to `view_sample` and describe what it sees. If it cannot, use **upload mode**: start the server with
> `--no-require-view`, attach the image files to the chat (name them like the samples, e.g. `LIC_21.png`) and tell it:
> *"The attached file name is the sample name. Call next_sample(name=<file name>), read the image from the chat, then set_fields … submit_label."*
> Claude (Desktop, Code, claude.ai) receives MCP images; I have not tested every client version.

### Security checklist for Funnel
* Funnel makes the port reachable from the **public internet**. The token is the only key: 43 random characters; never paste the secret URL in public places. The chat provider stores the connector URL, so rotate (`token --save`, restart) if in doubt.
* Never run HTTP mode with `--no-auth` except on 127.0.0.1. Never commit `alg_mcp_token.txt` or the workspace.
* The server exposes only registered samples, no file paths, no reference values. Stop Funnel (`tailscale funnel reset`) when you are not labeling.

## Daily workflow

```powershell
python run_alg_mcp.py add D:\new_images
# agents label (stdio client, or the HTTP server + Funnel for remote chatbots)
python run_alg_mcp.py status                          # queue, who is working, needs_human samples, AI labels by review state
python run_alg_mcp.py review --only needs_review --n 40     # opens-in-a-browser HTML sheet: image + fields + why it was flagged
python run_alg_mcp.py review --only auto_ok --n 100 --seed 7   # AUDIT a random sample of the "all checks passed" labels
python run_alg_mcp.py reject  LIC_21 ID_05            # wrong ones
python run_alg_mcp.py approve LIC_22                  # reviewed and fine
python run_alg_mcp.py promote --auto-ok --approved    # copy into labeled/ (never overwrites, rejected never move)
python run_alg_mcp.py promote --auto-ok --dry-run     # see what would move
```

`auto_ok` means: format rules met, no format exceptions, every MRZ check digit valid and the MRZ consistent, no disagreement with the reference values, and nothing marked unreadable
although a reference exists. It is **not** proof of correctness: audit a random sample (0 errors in 100 audited fields means the error rate is below ~3 % with 95 % confidence).
Labels without reference values and a single agent are always `needs_review`.

**Calibrate before you trust it** (what the research recommends): label ~50 images yourself in A.L.G (cover all 3 document types and your conditions), let the AI label the
same images, then `python run_alg_mcp.py compare`. It matches by image and reports per field: `agree`, `differ`, `ai_over_abstains` (said unreadable, you read it),
`ai_over_reads` (said readable, you said unreadable: the dangerous one). Improve the instructions from the mistakes you see, repeat.

**Two agents** (e.g. Claude + GLM): `serve --redundancy 2` (or `ALG_MCP_REDUNDANCY=2`) and give them different agent names. A label is written when both agree
on every field; otherwise the sample is `needs_human`: `python run_alg_mcp.py resolve <sample> --pick claude`, or label it by hand in the app.

## Rules the server enforces

| Situation | What happens |
|---|---|
| Value empty / `N/A` / `?` / `…` / `[note]` / text + `[UNREADABLE]` | rejected, told to use `mark_unreadable` |
| `[UNREADABLE]` without a valid reason code (`other` needs a note) | rejected |
| Value breaks the app's regex for that field | stored as *pending*; `submit_label` blocked until re-read or `format_exception` (≥ 8 chars) given |
| A field is missing | `submit_label` blocked |
| A side was never viewed (`view_sample`), unless `--no-require-view` | blocked |
| ID: MRZ line not 30 chars, bad character, wrong fixed part (`IDIRQ`, `IRQ`, sex), or a wrong check digit of the document number / birth date / expiry date | blocked; the message says which characters to re-read. If the agent is certain the line is what is printed, it keeps it and gives a `format_exception` for that line (→ human review). Never "fix" it, never mark a readable line unreadable |
| ID: only the composite check digit (last character of line 2) is wrong | `needs_confirmation` (`mrz_check_digit` flag): re-check that character, then `confirm=true`; the label goes to human review |
| ID: MRZ disagrees with a visual field (serial, national id when line 1 has optional data, dates, sex, names) | `needs_confirmation`: agent must re-inspect, then `confirm=true`. Names are compared by their consonants, because the MRZ spells them with vowels (`الربيعي` → `ALRUBAIE`) |
| Many fields unreadable, or all of them | `needs_confirmation` (zoom first / skip the sample) |
| Another agent holds the sample | not offered; claims expire after 45 min (`ALG_MCP_LEASE_MINUTES`) |
| Several photos of one document carry the same number (`ID_01_glare`, `ID_01_blur`) | the first keeps `ID_01`, the next ones take the next free number (nothing is overwritten; the sidecar keeps the source). `tools/prepare_photos.py` gives every photo its own number |

## Tools

| Tool | Purpose |
|---|---|
| `get_protocol`, `get_schema`, `queue_status`, `list_samples` | read-only information |
| `next_sample(agent, doc_type?, name?)` | claim a sample (idempotent; `name` claims a specific one) |
| `view_sample(sample_id, side, region?, max_side?)` | the image; `region=[x0,y0,x1,y1]` (0..1) zooms |
| `set_fields`, `set_field`, `mark_unreadable`, `set_conditions` | write the draft |
| `validate_sample`, `mrz_check` | check before submitting |
| `submit_label(confirm?)`, `skip_sample(reason)`, `release_sample` | finish / skip / give back |
| `get_truth` | off by default (blind labeling); `--expose-truth` enables it |

All tools return plain JSON text (works in every client); `view_sample` returns text + a JPEG image block.

## Settings

CLI flags or environment variables: `--workspace` / `ALG_WORKSPACE`, `--agent` / `ALG_MCP_AGENT`, `--redundancy` / `ALG_MCP_REDUNDANCY` (1 or 2),
`--output ai_labeled|labeled` / `ALG_MCP_OUTPUT` (`labeled` skips the review folder: not recommended), `--no-require-view` / `ALG_MCP_REQUIRE_VIEW=0`,
`--expose-truth` / `ALG_MCP_EXPOSE_TRUTH=1`, `ALG_MCP_LEASE_MINUTES`, `ALG_MCP_TOKEN`, `ALG_MCP_ALLOW_HOSTS`.
The merge width/gap and JPEG option are read from the app's `settings.json` in the workspace, so merged ID images match the desktop app.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Say who you are` | pass `agent=...` or set `ALG_MCP_AGENT` in the client config |
| `421 host not allowed` | custom domain: `--allow-host your.domain` (`*.ts.net` is built in) |
| `401 unauthorized` | token typo, or the secret URL is missing `/s/<token>` |
| Funnel shows 502 | the server is not running or runs on another port than `tailscale funnel` |
| Agent cannot see the image | the model is not a vision model, or the client drops MCP images: use a vision model / upload mode |
| `mcp 2.x` import error | `pip install "mcp>=1.30,<2"` (the 2.x API was renamed; not supported yet) |
| `could not lock ...state.lock` | no server running? delete `alg_mcp_state.lock` |
| Agent marks too much unreadable | run `compare`; add rules from its mistakes; remind it to zoom |
| Slow or "payload too large" images | the agent can ask for `max_side` 1000-1200; the default is 1600 |
| Name clash between app and agent labels | both take names from the same counters in `<workspace>/state.json`: do not label *external photos* in the app and with an agent at the same moment |
| A sample is stuck as `needs_human` | `python run_alg_mcp.py status` shows why; `resolve --pick <agent>` or label it in the app |

## Tests

```powershell
pip install -r requirements-mcp.txt
python -m pytest tests_mcp -q      # 73 tests: rules, MRZ, redundancy, blind comparison, stdio subprocess, HTTP auth, SSE, CLI/review
```
They run against the real A.L.G label engine. Not covered by the tests: the real ChatGPT / Claude / GLM clients and Windows (the tests ran on Linux).

## Repository hygiene

The parent A.L.G `.gitignore` excludes the MCP virtual environment, token,
state and audit files, review output, and AI-labeled data. Keep generated
labels, images, and secrets out of version control.
