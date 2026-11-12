# Training Laya for the GATE Block

| File | What it is |
|---|---|
| `kaggle_train_laya.ipynb` | the Kaggle notebook: publish the data, smoke test, train, check, publish the model |
| `train_laya.py` | the training script the notebook runs (works on any machine with a GPU, or on a CPU for `--smoke`) |
| `publish_dataset.py` | builds the Hugging Face dataset folder and its card from the data, and uploads it |
| `data/v1/` | the training data (`laya-data` export, gzip) and how it was made (`provenance.json`) |

## The plan, and why it is built this way

The GATE Block's decision is **made by the rules and Laya together**: Python measures the photo and lists
the branches the measurements permit; Laya chooses among them; the router validates the choice and the
safety rules always win. The KYC system's **final confidence is a formula**, not a model: it combines the
OCR's field confidences, the cross-document checks and the scores of the earlier blocks, and it is what
has to keep the promise *"95 % means right 95 % of the time"*.

| stage | objective | teaches | labels | status |
|---|---|---|---|---|
| 1. SFT | `soft-ce` (supervised) | the job: accept / enhance / reject, the reason, the DocRes task | the rules' decisions (this dataset) | ready |
| 2. RL | `rlcd`: GRPO-style policy gradient over noisy samples of Laya's own probabilities, reward = spherical score + ranked probability score, plus the supervised term | an honest probability for every answer | the same rule labels now; real outcomes in phase 2 | ready |
| 3. calibration | one temperature per question, abstention thresholds for a 5 % error target | the probabilities on held-back data | a calibration slice | automatic after each stage |
| phase 2 | stage 2 on outcomes | *"will OCR read this photo correctly?"* | TeleOCR results on labelled photos | needs Block 3 |

**What I would change in the original idea, and why.** SFT for the job and RL for the score is the right
split. Two things matter for the 95 % target, though:

1. *What* the score is calibrated against. On rule labels, a calibrated 95 % means "agrees with the rules
   95 % of the time" — useful for the gate, but it says nothing about whether the document will be read
   correctly. The score the final formula needs has to be trained (stage 2) and checked against **real
   outcomes**: photos that went through TeleOCR and whose fields were compared with the A.L.G labels.
2. *Where* the promise is kept. No training objective guarantees calibration on photos it never saw. The
   formula must be **fitted on one part of the held-out outcomes and verified on another**, with a
   reliability table per confidence bin (the notebook prints one): the bin "0.95-0.99" has to show
   accuracy >= 0.95 with enough documents in it before the system may say 95 %.

The RL objective Laya ships with (`rlcd`) fits stage 2: its rewards are proper scoring rules, which are
maximised only by the true probability. With rule labels it sharpens the confidence of the gate's own
answers; with outcome labels it becomes the readability score for the formula. Laya's authors measured no
gain from its RL term on their own benchmark, so do not take it on faith: each stage's report has the
held-out loss, ECE and Brier score before and after, and the notebook prints the reliability table of the
final model.

## Running it

1. Kaggle: *New Notebook* -> *File* -> *Import notebook* -> `kaggle_train_laya.ipynb`. Settings: GPU (T4 or
   P100), Internet on. Add-ons -> Secrets: `HF_TOKEN` = a Hugging Face token with **write** access.
2. *Run all*. The first cell group publishes `RADAXGRS/kyc-gate-laya-decisions` (public) if it does not
   exist yet; the smoke test takes a few minutes; the real run trains the whole model (SFT 3 epochs, RL 2
   epochs) and publishes `RADAXGRS/laya-kyc-gate` **publicly only if every gate passes**:

| gate | limit |
|---|---|
| decision accuracy (held-out) | >= 0.98 |
| false accepts (a photo the rules reject or enhance, accepted) | 0 |
| decision ECE | <= 0.05 |
| answers with confidence >= 0.95 | right >= 95 % of the time (when there are at least 30) |

3. Use the published model in the GATE Block: `Gate(backend=LayaBackend("RADAXGRS/laya-kyc-gate"))`, or
   download it and point `config/laya.yaml` at the folder.

Locally, a quick check of the whole script on a CPU (the whole model needs a GPU or about 10 GB of free
RAM, so the check trains the head only):

```powershell
python training\train_laya.py --base D:\Desktop\Ai\Models\Laya\Multilingual --data training\data\v1_unpacked --out run_smoke --smoke 24 --sft-epochs 1 --rl-epochs 1 --micro-batch 4 --grad-accum 1 --device cpu --freeze-encoder
```

(`--data` takes a folder with `train.jsonl` and `eval.jsonl`; `python training\publish_dataset.py --data
training\data\v1 --stage training\data\v1_unpacked` unpacks the repository copy.)

## Regenerating the data

```powershell
python -m gate_block laya-data --out laya_data --photos PHOTOS --simulate 300 --synthetic 3200 --synthetic-enhance 1700 --seed 0
```

`PHOTOS` holds the image files whose measured states are added (`provenance.json` lists what v1 used). The
labels come from the rules, so regenerate after any change to `config/rules.yaml`,
`config/decision_tree.yaml` or the thresholds, and train again.
