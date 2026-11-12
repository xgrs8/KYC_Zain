# GATE Block — Block 1 of the KYC Document Agent

The GATE Block decides what happens to every photo a customer uploads **before any expensive model
runs**. It answers three questions, in this order, and writes down why:

1. **Is it the right document?** (Gate 1, the check gate: pattern recognition against templates of the
   Iraqi national ID front and back, the tax card and the business licence.)
2. **How good is the photo?** (the quality gate: blur, camera shake, glare, shadow, darkness, contrast,
   noise, cut-off, a finger or an object on the document, camera angle, curl, too few pixels — for the
   whole document and for every field that will be read.) The measurements are written as JSON.
3. **What now?** Laya reads a compact JSON summary of the measurements and chooses a branch of the
   decision tree; Python checks the choice against the measurements and routes:

| Outcome | Next block | Example |
|---|---|---|
| `accept` | TeleOCR (Block 3) | sharp, evenly lit, complete |
| `enhance:<task>` | DocRes (Block 2) with **one** of its five tasks: `deblurring`, `deshadowing`, `dewarping`, `appearance`, `binarization` | a bit of shadow → `enhance:deshadowing` |
| `reject:<reason>` | the customer, with a fixed Iraqi-Arabic message | `reject:blur` → «الصورة مغوشة من فضلك ارجع التقط الصورة بدون تغويش» |
| `review:<reason>` | a person | the gate could not tell (rare) |

A national ID is uploaded as two photos. Right after Gate 1 has confirmed that the front slot holds a
front and the back slot a back, the two are merged exactly like A.L.G merges them (front over a 24 px
white band over the back, both sides at a common width with LANCZOS; with the default width of 1600 the
result is pixel-identical to the reference merge `ID_23.png`). The quality gate then measures each half
in the pixels of the original upload, so enlarging a small photo for the merge cannot hide that it holds
few pixels.

No model writes text to the customer. Laya returns a choice; Python looks up the message for that
choice in `config/messages_ar.yaml`.

```
upload ─► intake ─► Gate 1: right document? ──no──► reject:no_document / wrong_document / wrong_side
                          │ yes
                          ▼
              (ID) merge front + back
                          │
                          ▼
              quality gate ─► measurements (JSON report) ─► compact state ─► Laya
                                                                              │ choice + probabilities
                                                                              ▼
                                  Python: safety rules, permitted branches, decision tree, message
                                                                              │
                     ┌───────────────────────────┬───────────────────────────┼────────────────────┐
                     ▼                           ▼                           ▼                    ▼
                 TeleOCR                 DocRes (one task)            retake message           review
                                                │
                                                ▼
                                gate.recheck(enhanced image): did it help?
                                (at most 2 DocRes rounds, then accept / retake)
```

## Folder layout

Every concern has its own file, and every number the gate uses has its own configuration file.

| File | What it does |
|---|---|
| `gate_block/pipeline.py` | the `Gate` object: `process_image`, `process_id`, `process_merged_id`, `process_submission`, `recheck` |
| `gate_block/imageio.py` | intake: decoding (EXIF rotation, HEIC if `pillow-heif` is installed), size and blank checks |
| `gate_block/recognition.py` | Gate 1: is it one of the templates, which side, where is it (ORB features + homography, outline search) |
| `gate_block/templates.py` | loads `templates/` and pre-computes every template's references (blur and motion rulers, text size …) |
| `gate_block/id_merge.py` | the A.L.G front/back merge, and splitting an already merged ID |
| `gate_block/quality.py` | the quality gate: every defect for the document and for every field |
| `gate_block/measures.py` | the measuring instruments themselves (pure numpy / OpenCV, no thresholds) |
| `gate_block/geometry.py` | homographies, outlines, perspective, warping |
| `gate_block/report.py` | the JSON quality report of a whole document |
| `gate_block/laya_state.py` | the compact JSON state handed to Laya |
| `gate_block/laya_client.py` | Laya's questions and the backends (`LayaBackend`, `StaticBackend`) |
| `gate_block/decision_tree.py` | the tree: reasons for a retake, DocRes tasks, and the evidence for each |
| `gate_block/rules.py` | the deterministic rules: safety net, fallback and baseline |
| `gate_block/router.py` | Laya recommends, Python validates and routes |
| `gate_block/messages.py` | the Iraqi-Arabic messages |
| `gate_block/config.py` | loads and validates every file in `config/` |
| `gate_block/synth.py` | simulated phone photos (tests, demo, training data) |
| `gate_block/cli.py` | the command line (`python -m gate_block …`) |
| `gate_block/tools/` | calibration, evaluation, Laya training data and fine-tuning, packing the trained head |
| `config/*.yaml` | thresholds, rules, decision tree, messages, routing, Laya, merge, recognition, intake, paths |
| `templates/<doc>/` | one clean reference image per side and `layout.json` (outline, fields, critical fields) |
| `training/` | the Kaggle notebook and scripts: Laya's training data on Hugging Face, whole-model training (SFT, then RL), checks, publishing |
| `tests/` | about 400 tests (instruments, measurements, rules, router, Laya contract, pipeline, CLI, training scripts) |
| `docs/` | how it works, what is measured, Laya, failure modes |

## Install

Python 3.10 or newer.

```powershell
cd "GATE Block"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m gate_block doctor          # self-test: config, templates, recognition, merge, decisions
```

Laya is optional for running the gate (without it the deterministic rules decide), and needed for the
full design. In the environment where Laya is installed (`pip install laya==0.4.31`):

```powershell
pip install -r requirements.txt
python -m gate_block doctor --laya "D:\path\to\gate_laya"
```

`gate_laya` is the fine-tuned checkpoint (see [Laya](#laya)). The trained head,
`gate_laya_head.safetensors` (30 MB), is attached to the repository's release **gate-block-v1.0.0**
(weights stay out of git); it is put back on top of your multilingual Laya folder with one command, which
refuses a base whose encoder differs from the one it was trained on:

```powershell
python -m gate_block laya-assemble --base "D:\Desktop\Ai\Models\Laya\Multilingual" --head gate_laya_head.safetensors --out gate_laya
```

A model trained with the Kaggle notebook (`training/`) is a complete checkpoint and needs no assembly:
`--laya RADAXGRS/laya-kyc-gate` (or the folder it was downloaded to).

## Use it

Command line (prints a short JSON result; `--out` writes the complete one, `--full` prints it):

```powershell
python -m gate_block check photo.jpg --expect tax_card --laya gate_laya
python -m gate_block id front.jpg back.jpg --workdir out --laya gate_laya
python -m gate_block submission --id-front f.jpg --id-back b.jpg --tax-card t.jpg --business-license l.jpg --workdir out --laya gate_laya
python -m gate_block recheck docres_output.png --previous result.json --laya gate_laya
python -m gate_block overlay photo.jpg --out overlay.png       # draw what the gate found
python -m gate_block tree                                      # the decision tree, messages and Laya's questions
python -m gate_block demo --laya gate_laya                     # simulated photos through the whole gate
```

Python:

```python
from gate_block import Gate, LayaBackend

gate = Gate(backend=LayaBackend("gate_laya"))       # Gate() alone: the rules decide
r = gate.process_id("front.jpg", "back.jpg", workdir="out")
r.route            # accept | enhance | reject | review
r.next_block       # teleocr | docres | retake | human_review
r.status_code      # e.g. "enhance:deshadowing", "reject:blur"
r.message_ar       # the Iraqi-Arabic message for a retake
r.docres_request   # {"image": ..., "task": "deshadowing", "round": 1, ...} for Block 2
r.teleocr_request  # {"image": ..., "kind": "merged_id", "document_type": "national_id"} for Block 3
r.report           # the full quality JSON: per-defect values, severities, per-field quality
r.decision         # who decided (laya / hard_rule / rules ...), Laya's probabilities, the trace
after = gate.recheck("docres_out.png", r)           # after DocRes: did it help? what next?
```

## Laya

Laya is a non-autoregressive decision model: it receives a state and a fixed set of typed questions and
returns a calibrated probability for every option in one forward pass. The gate asks three questions —
`decision` (accept / enhance / reject), `reject_reason` (the ten retake reasons) and `docres_task` (the
five DocRes tasks) — about a compact state such as

```json
{"doc":"tax_card","blur":"moderate","shadow":"mild","crit":{"taxpayer_number":"blur:moderate"},"oth":["blur:mild"],
 "allowed":["enhance:deblurring","enhance:deshadowing","reject:blur"]}
```

Only what is wrong is listed; a measurement that could not be made is listed under `unk`, never treated
as fine; after a DocRes pass `round` and `done` say which tasks were already applied. `allowed` lists the
branches of the decision tree the measurements permit (Python computes them from the thresholds and the
safety rules). Laya makes the judgement call between them — a mild defect: OCR as it is, or restore
first? a moderate one: restore, or ask for a new photo? which task, which retake reason? — and Python
checks the choice against the same conditions before routing.

The base multilingual Laya answers the reason and task questions sensibly out of the box but accepts
badly blurred or glared photos, so the gate ships a **fine-tuned decision head** (the encoder is frozen
and unchanged). Only the head is trained, so the trained part is a small file; `laya-assemble` puts it
back on top of your multilingual checkpoint and refuses a different base. Details, training data and
results: [docs/LAYA.md](docs/LAYA.md).

To train the **whole model** (encoder included) on a GPU, from a fresh multilingual checkpoint and the
published data (`RADAXGRS/kyc-gate-laya-decisions` on Hugging Face), run
[`training/kaggle_train_laya.ipynb`](training/README.md) on Kaggle: SFT teaches the decision, RL with
proper-scoring-rule rewards trains the confidence, and the model is published only if it passes the
held-out checks (no false accept, decision accuracy >= 98 %, calibrated confidence).

Whatever Laya says, Python checks it before anything happens: a choice the measurements do not permit
(accepting a photo with a moderate defect, enhancing with a task that has no evidence, rejecting a clean
photo) is replaced by the rules' decision and recorded as `rules_override`; the safety rules (document
cut off, a field lost under glare or a finger, damage DocRes cannot repair) always win. A stuck or
untrained model therefore cannot push a bad photo to OCR, reject a good one, or send everything to review.

## Configuration

| File | Holds |
|---|---|
| `config/quality_thresholds.yaml` | the mild / moderate / severe limit of every measurement |
| `config/rules.yaml` | when to enhance, when a defect is beyond repair, field loss |
| `config/decision_tree.yaml` | priority of the retake reasons and of the DocRes tasks |
| `config/messages_ar.yaml` | the Iraqi-Arabic message of every reason (edit the wording freely) |
| `config/routing.yaml` | DocRes rounds, what happens when enhancement fails, review on/off |
| `config/laya.yaml` | minimum confidence, who decides when Laya's answer is not usable |
| `config/id_merge.yaml` | merge width (1600) and gap (24) |
| `config/recognition.yaml`, `config/image.yaml`, `config/paths.yaml` | Gate 1, intake, template folder |

Every file is validated when the gate starts; a wrong value stops it with a clear error rather than
deciding with a broken setting. `python -m gate_block calibrate <folder>` suggests thresholds from your
own labelled photos ([docs/MEASUREMENTS.md](docs/MEASUREMENTS.md)).

## Tests

```powershell
python -m pytest tests -q
```

The suite checks the instruments on inputs with a known answer, every defect on simulated photos of all
four templates, the rules and the router against every backend (including a broken, a stuck and a random
one), the Laya contract, the merge against the reference image, the full pipeline and the command line.
Beyond the suite, the gate was run on the real photos and on the project dataset — see
[docs/LAYA.md](docs/LAYA.md#end-to-end-results).

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the flow, the decision, the JSON contract with DocRes and TeleOCR
* [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) — what each quality measurement is, how it is made, its thresholds, templates, calibration
* [docs/LAYA.md](docs/LAYA.md) — the Laya contract, fine-tuning, results
* [docs/FAILURE_MODES.md](docs/FAILURE_MODES.md) — documented failure modes and what the gate does about them

## Disclosure

Models: Laya 0.4.31 (multilingual checkpoint `convaiinnovations/laya-multilingual`, Apache-2.0, encoder
`jhu-clsp/mmBERT-base`) as the decision model, with a decision head fine-tuned here. No other model runs
in the gate; DocRes and TeleOCR are the next blocks. Training data: generated by this block
(`laya-data`), labelled by its own rules, published with its provenance. Libraries: OpenCV, NumPy,
Pillow, PyYAML (and PyTorch / safetensors / huggingface_hub for fine-tuning).
AI assistance: the code, tests and documentation of this block were written with Claude (Anthropic) as a
coding assistant, under the direction of the project author, who defined the design.
