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

The questions (`gate_block/laya_client.py`; the option order is part of the trained model — never
reorder it):

| Question | Options |
|---|---|
| `decision` | accept · enhance · reject |
| `reject_reason` | blur · glare · dark · overexposed · shadow · cut_off · angle · too_small · obstructed · other |
| `docres_task` | dewarping · deshadowing · appearance · deblurring · binarization |

## What Python does with the answer

See [ARCHITECTURE.md](ARCHITECTURE.md#5-the-decision). In short: the answer is used only if it is
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

## Why the head is fine-tuned

Asked zero-shot, the base multilingual Laya picks sensible reasons and tasks but gets the main decision
wrong in both directions: it chose `reject:other` for clean photos and `accept` for photos with severe
blur or a covering object. Python's validation catches every one of those (the photo is decided by the
rules instead), but then Laya decides nothing. So the gate teaches Laya its decision tree.

## Training data (`python -m gate_block laya-data`)

Each row is a state, the three questions, and the expected answers. The labels are the decisions of the
gate's deterministic rules (`gate_block/rules.py`) — the same rules that are the fallback — on:

* **photos**: the ten real phone photos and the photographed dataset cards, measured by the gate;
* **simulated photos**: every template under random combinations of blur, camera shake, shadow, glare,
  darkness, cut-off, a finger, curl, distance, angle and phone-like tone curves, measured by the gate;
* **synthetic states**: random defect severities and field problems for every document side, labelled
  by the rules (they cover combinations that are rare in photos, and every reason and task);

with each state's DocRes round and already-applied tasks. Identical states are kept a limited number of
times, and a state is never split between training and evaluation, so the evaluation measures states
Laya has not seen.

Only the decision head is trained: the encoder stays bit-identical to the multilingual checkpoint.
Everything around the optimisation is Laya's own `laya.train.finetune` (reading the rows, holding out a
calibration slice, the evaluation before and after, fitting the temperatures and abstention thresholds,
saving a checkpoint `laya.load` reads); `laya-train` replaces only its inner loop. Because a frozen encoder
always returns the same output for the same sequence, that output is computed once, in batches of
similar length; the head layers are then trained on it, and the last layer computes only the positions
that are read (the option markers and token 0) while still attending to every token. The shortcut is
checked against Laya's own forward pass before training starts (difference below 1e-4). Seven epochs are
a CPU job of under two hours (a GPU makes it minutes). The objective is soft cross-entropy (the labels are
exact decisions).
`laya-pack` writes just the head and records a digest of the encoder; `laya-assemble` rebuilds the full
checkpoint from your multilingual folder and refuses a different one.

```powershell
python -m gate_block laya-data  --out laya_data --photos PHOTOS --simulate 400 --synthetic 4000
python -m gate_block laya-train --base MULTILINGUAL_DIR --data laya_data --out gate_laya --epochs 7 --device cpu
python -m gate_block laya-pack  gate_laya --out gate_laya_head.safetensors --base MULTILINGUAL_DIR
python -m gate_block laya-assemble --base MULTILINGUAL_DIR --head gate_laya_head.safetensors --out gate_laya
python -m gate_block evaluate --simulate 300 --seed 7 --laya gate_laya
```

## Results

The shipped head (`models/gate_laya_head.safetensors`) was trained in two steps on a CPU (method
`cached-head`, encoder unchanged):

1. 7 epochs on `laya-data --simulate 400 --synthetic 4000` plus the real and dataset photos: 2 714
   training states (4 744 question items, of which Laya holds out 400 for calibration);
2. 3 more epochs on a top-up of 1 666 states in which the rules enhance and two or more DocRes tasks are
   permitted (now `--synthetic-enhance`), mixed with 1 390 states of step 1 so the rest is not forgotten.

Held-out evaluation, per question, on 531 states (989 items) that never occur in training
(`agent.predict` as the gate calls it):

| Question | Base multilingual Laya | After step 1 | After step 2 (shipped) |
|---|---|---|---|
| `decision` (accept / enhance / reject) | 22.2 % | 99.4 % | **99.4 %** |
| `reject_reason` (10 reasons) | 76.4 % | 99.7 % | **99.7 %** |
| `docres_task` (5 tasks) | 74.8 % | 93.9 % | **100 %** (115 / 115) |

* The base model accepted 302 of the 343 photos that must be retaken and 111 of the 115 that need
  DocRes. The fine-tuned head **accepts none of them**: its three decision errors are two
  `accept → enhance` and one `accept → reject` (a usable photo restored or retaken; never a bad photo
  sent to OCR).
* Calibration (Laya fits the temperatures on 400 calibration items it holds out of the training data;
  measured on the 989 held-out items): expected calibration error 0.9 % after step 1 and 0.7 % after
  step 2, mean confidence 99.4 % at 99.6 % accuracy; no decision fell under the 0.55 confidence floor.
* Step 1's task errors came from too few training states in which more than one task is permitted: the
  head had rarely seen the order the rules use (dewarping, deblurring, deshadowing, appearance) or a task
  that was already applied. Step 2 adds exactly those states. It was designed after looking at these
  errors, so its 100 % on this set is optimistic; the published dataset's validation split and the
  end-to-end run below are the independent checks.

### Training the whole model on a GPU

`training/kaggle_train_laya.ipynb` starts from a fresh `convaiinnovations/laya-multilingual`, pulls the
published data (`RADAXGRS/kyc-gate-laya-decisions`, the same generator with `--synthetic-enhance`), trains
the whole model (encoder included) in two stages — SFT (`soft-ce`) for the decision, then RL (`rlcd`,
proper-scoring-rule rewards) for the confidence — checks it on the held-out split and publishes it only
if every gate passes. See [training/README.md](../training/README.md).

End to end with the gate, see [the end-to-end results](#end-to-end-results).

### End-to-end results

The whole gate with the shipped Laya head (`LayaBackend`, CPU), every photo also run through the rules
alone for comparison:

| Group | Correct |
|---|---|
| real phone photos of the printed licence (table, keyboard, hand-held, blur, clutter, a finger, glare, low light, partial, steep angle) | 10 / 10 |
| print-run renders: 12 identities × (merged ID, tax card, licence) | 36 / 36 |
| simulated photos of the four templates (blur, camera shake, exposure, shadow, glare, cut-off, distance, a finger, curl 8 % and 10 %, rotation, angle) | 35 / 35 |
| simulated IDs, both sides | 5 / 5 |
| Gate 1: wrong document, wrong side, swapped sides, six kinds of non-document, a broken file, a blank image | 12 / 12 |
| DocRes round trips (`recheck`): fixed, made worse, two rounds, no third round | 7 / 7 |

Laya decided 90 of the photos that reached a decision (the other two were decided by the rules because
nothing was left to choose, and one by the recheck), always within the permitted branches and always with
the same answer as the rules (91 of 91 eligible); no answer had a confidence under 0.9. Whole submissions
route as the worst document: three good documents → `accept`; a licence photo in the tax-card slot →
`reject` with `wrong_document` for that slot.

On the published dataset's validation split (608 states the head never saw, made by the same generator
after the head was trained): decision 99.5 %, reject reason 99.6 %, DocRes task **93.2 %**, no false
accept. This is the independent figure for the task choice; the whole-model training of
`training/kaggle_train_laya.ipynb` learns from 1 157 task-choice states and reports its own held-out
numbers.

## Limits

* Laya imitates the rules it was taught by; it is valuable as a calibrated, fast decision layer that can
  later be retrained on better labels (TeleOCR outcomes: was the photo read correctly? did DocRes help?),
  not as a source of better judgement today. See [FAILURE_MODES.md](FAILURE_MODES.md#4-laya-can-only-be-as-good-as-the-rules-it-was-taught-by).
* When Laya's top probability is under 0.55 the rules decide; that share is reported by `evaluate`.
