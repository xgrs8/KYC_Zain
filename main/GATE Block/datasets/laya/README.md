# Laya training data for the GATE Block

The rows Laya needs to learn the gate's decision. The base multilingual Laya, asked the gate's three
questions zero-shot, does not judge severity (on the validation rows it answered `accept` for 591 of 625,
246 of the 277 photos that must be retaken among them), so it has to be fine-tuned on this policy before
`--laya` is useful. The fine-tuning itself is done outside this repository; this folder is its data.

| File | What it is |
|---|---|
| `v1/train.jsonl.gz` | 3 367 rows for training |
| `v1/eval.jsonl.gz` | 625 rows for validation (no state occurs in both) |
| `v1/summary.json` | counts per origin, decision, reason and task, as written by `laya-data` |
| `v1/provenance.json` | the command, the inputs and where the labels come from |
| `publish_dataset.py` | builds a Hugging Face dataset folder with a card written from the data, and uploads it |

Each row is what `laya.train.finetune` reads:

```json
{"state": "{\"doc\":\"tax_card\",\"blur\":\"moderate\",\"allowed\":[\"enhance:deblurring\",\"reject:blur\"]}",
 "questions": {"decision": {"...": "..."}, "reject_reason": {"...": "..."}, "docres_task": {"...": "..."}},
 "expected": {"decision": "enhance", "docres_task": "deblurring"},
 "meta": {"label_source": "rules", "origin": "synthetic"}}
```

## Where the labels come from

From the gate's deterministic rules (`gate_block/rules.py` with `config/rules.yaml` and
`config/decision_tree.yaml`), applied to the measurements of each state. They are not human labels and
not OCR outcomes, so a model trained on them learns to reproduce the rules, and its confidence means
"agrees with the rules".

## Where the rows come from

| origin | train | validation | what it is |
|---|---|---|---|
| `synthetic` | 2 077 | 400 | measurement states drawn at random over the gate's severity scales; no image behind them |
| `synthetic_enhance` | 1 157 | 194 | the same, kept only where the rules enhance and two or more DocRes tasks are permitted |
| `simulated_photo` | 117 | 29 | photos simulated from the four templates (blur, shake, glare, shadow, curl, cut-off, a finger ...) and measured by the gate |
| `photo` | 16 | 2 | real image files measured by the gate: renders of the synthetic print run and phone photos of one printed licence |

Identical states are kept at most three times; no state received two different labels.

## Making it again, publishing it

```powershell
python -m gate_block laya-data --out laya_data --photos PHOTOS --simulate 300 --synthetic 3200 --synthetic-enhance 1700 --seed 0
python datasets\laya\publish_dataset.py --data datasets\laya\v1 --stage hf_dataset                  # look at the card first
python datasets\laya\publish_dataset.py --data datasets\laya\v1 --push RADAXGRS/kyc-gate-laya-decisions
```

Regenerate after any change to the rules, the decision tree or the thresholds: the labels follow them.
Publishing needs a Hugging Face token with write access (`HF_TOKEN`, or `hf auth login`).
