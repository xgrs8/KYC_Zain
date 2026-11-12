# Laya in the GATE Block

## The contract

Laya (`laya` 0.4.31) is a non-autoregressive decision model: a text encoder (the multilingual checkpoint
uses `jhu-clsp/mmBERT-base`) with a small decision head. It receives a **state** and a set of typed
**questions** and returns, in one forward pass, a probability for every option of every question. It
writes no text.

The state is built by `gate_block/laya_state.py` from the quality report: words, not numbers, and only
what is wrong.

```json
{"doc":"national_id","round":1,"done":["deshadowing"],"blur":"moderate","crit":{"name":"blur:moderate"},"oth":["blur:mild"],
 "allowed":["enhance:deblurring","reject:blur"]}
```

| Key | Meaning |
|---|---|
| `doc` | document type |
| `cut` `res` `blur` `dark` `contrast` `shadow` `glare` `curl` `noise` | severity of each defect that is present (`persp` and `obstr` only from moderate) |
| `glare_all` | the ink is washed out over most of the document (overexposure rather than a reflection) |
| `unk` | defects that could not be measured — never silently "none" |
| `crit` | critical fields with a problem, `cause:severity` |
| `oth` | problems of the other fields, worst first, at most four |
| `round`, `done` | after a DocRes pass: which round, which tasks were already applied |
| `allowed` | the branches the measurements permit (`rules.permitted`): `accept`, `enhance:<task>`, `reject:<reason>`; with a safety rule only `reject:<its reason>` |

The questions (`gate_block/laya_client.py`; a checkpoint fine-tuned for the gate learns the options by
position — never reorder them):

| Question | Options |
|---|---|
| `decision` | accept · enhance · reject |
| `reject_reason` | blur · glare · dark · overexposed · shadow · cut_off · angle · too_small · obstructed · other |
| `docres_task` | dewarping · deshadowing · appearance · deblurring · binarization |

## What Python does with the answer

See [ARCHITECTURE.md](ARCHITECTURE.md#5-the-decision-laya_statepy-laya_clientpy-routerpy-rulespy-decision_treepy). In short: the answer is used only if it is
complete, not truncated and confident enough (`config/laya.yaml`, `min_decision_confidence: 0.55`); the
chosen branch must be **permitted** by the measurements (accept only without a moderate defect, enhance
only with a task whose defect was measured and not applied yet, reject only for a measured reason); the
safety rules always win. Otherwise the deterministic rules decide and the result says so
(`decided_by`: `laya`, `hard_rule`, `rules_fallback`, `rules_override`).

## Why Laya is given the permitted branches

The decision depends on *combinations*: which defect, how severe, whether DocRes has a task for it,
whether a critical field is affected, which tasks were already applied. A probe on the frozen multilingual
encoder showed that its features capture those combinations only partly: a linear read-out of the
decision reached about 82 % on held-out states from the plain JSON. With the `allowed` list in the state —
the branches the thresholds and safety rules permit, computed by Python — the same read-out reaches 99 %.
So Python states the hard constraints, and Laya makes the choices between them that need judgement: a
mild defect (OCR as it is, or restore first?), a moderate one (restore, or a new photo?), which task,
which reason. Those are also the choices a later retraining on OCR outcomes can improve. The router
enforces exactly the same conditions afterwards, and a photo with nothing to choose from is decided by
the rules without asking Laya.

## Why Laya must be fine-tuned

Asked zero-shot, the base multilingual Laya does not judge severity: it answers `accept` for almost
everything. On the 625 validation rows of [`datasets/laya/v1`](../datasets/laya/README.md) it said
`accept` for 591 of them, among them 246 of the 277 photos that must be retaken and 277 of the 280 that
need DocRes, so its main decision agreed with the rules on 16 % of the rows (reject reason 77 %, DocRes
task 68 %). On the ten real phone photos it accepted nine, six of the seven bad ones among them.

Python's validation stops every answer the measurements do not permit: none of those accepts of a photo
that must be retaken would reach OCR, because a safety rule or the rules decide instead. But then Laya
decides nothing, and where the measurements leave a real choice (a mild defect: OCR as it is, or restore
first?) the base model nearly always answers `accept`.

So `--laya` needs a checkpoint fine-tuned on the gate's decisions. The fine-tuning is done outside this
repository, and no model weights are kept here; without `--laya` the deterministic rules decide, which is
the default. `python -m gate_block doctor --laya CHECKPOINT` asks a checkpoint three probe states the gate
really builds and says whether its answers agree with the rules; a checkpoint that gives the same answer
for every state has not been fine-tuned for the gate.

## Training data (`python -m gate_block laya-data`)

Each row is a state, the three questions and the expected answers, in the format `laya.train.finetune`
reads as it is. The labels are the decisions of the gate's deterministic rules (`gate_block/rules.py`) —
the same rules that are the fallback — on:

* **photos**: real image files measured by the gate (renders of the synthetic print run, and phone photos
  of one printed licence);
* **simulated photos**: every template under random combinations of blur, camera shake, shadow, glare,
  darkness, cut-off, a finger, curl, distance, angle and phone-like tone curves, measured by the gate;
* **synthetic states**: random defect severities and field problems for every document side, labelled
  by the rules (they cover combinations that are rare in photos, and every reason and task);
* **task-choice states** (`--synthetic-enhance`): synthetic states kept only where the rules enhance and
  two or more DocRes tasks are permitted, so the order of the tasks and the tasks already applied are
  well covered;

with each state's DocRes round and already-applied tasks. Identical states are kept a limited number of
times, and a state is never split between training and validation, so the validation split measures
states a model trained on the rows has not seen.

The rows of this version are in [`datasets/laya/`](../datasets/laya/README.md): 3 367 for training and
625 for validation, with their counts and provenance. `datasets/laya/publish_dataset.py` publishes them as
a Hugging Face dataset with a card written from the data.

```powershell
python -m gate_block laya-data --out laya_data --photos PHOTOS --simulate 300 --synthetic 3200 --synthetic-enhance 1700 --seed 0
python -m gate_block export-laya PHOTOS --labels labels.csv --out laya_export   # rows from your own photos
```

Regenerate the rows after any change to the rules, the decision tree or the thresholds: the labels follow
them. `export-laya --labels` takes labels from a CSV instead of the rules (for example from TeleOCR
outcomes); those are the labels that can make Laya better than the rules.

With a fine-tuned checkpoint:

```powershell
python -m gate_block doctor --laya CHECKPOINT
python -m gate_block evaluate --simulate 300 --seed 7 --laya CHECKPOINT     # how often Laya agrees with the rules
```

## End-to-end results with the rules deciding

The whole gate as it runs without `--laya` (CPU):

| Group | Correct |
|---|---|
| real phone photos of the printed licence (table, keyboard, hand-held, blur, clutter, a finger, glare, low light, partial, steep angle) | 10 / 10 |
| print-run renders: 12 identities × (merged ID, tax card, licence) | 36 / 36 |
| simulated photos of the four templates (blur, camera shake, exposure, shadow, glare, cut-off, distance, a finger, curl 8 % and 10 %, rotation, angle) | 35 / 35 |
| simulated IDs, both sides | 5 / 5 |
| Gate 1: wrong document, wrong side, swapped sides, six kinds of non-document, a broken file, a blank image | 12 / 12 |
| DocRes round trips (`recheck`): fixed, made worse, two rounds, no third round | 7 / 7 |

Of the 93 photos in the quality groups (everything but Gate 1), the deterministic rules decided 70, a
safety rule 22 and the DocRes recheck 1. Whole submissions route as the worst document: three good
documents → `accept`; a licence photo in the tax-card slot → `reject` with `wrong_document` for that slot.

These are the answers a checkpoint fine-tuned on `datasets/laya` is meant to reproduce. With `--laya`
every photo is still run through the rules as well (`rules_baseline` in the result), so the same run shows
where Laya and the rules differ.

## Limits

* A Laya fine-tuned on these rows imitates the rules it was taught by; it is valuable as a calibrated,
  fast decision layer that can later be retrained on better labels (TeleOCR outcomes: was the photo read
  correctly? did DocRes help?), not as a source of better judgement today. See
  [FAILURE_MODES.md](FAILURE_MODES.md#4-laya-can-only-be-as-good-as-the-rules-it-was-taught-by).
* When Laya's top probability is under 0.55 the rules decide; that share is reported by `evaluate`.
