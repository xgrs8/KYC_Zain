# Failure modes

What can go wrong in Block 1, how it shows, what the gate does about it, and what is left open. Each
entry was observed or reproduced while building and testing the block.

## 1. A photo of a screen, a photocopy or a printed copy is accepted

**What happens.** Gate 1 recognises the *design* of the document. A clear photo of the ID shown on a
phone screen, or a good colour copy, looks like the document to every measurement the gate makes, so it
can be accepted and sent to OCR.

**Why.** Block 1 judges whether a photo can be *read*, not whether the document is *genuine* or
physically present. Presentation-attack detection (moiré, screen glare patterns, paper texture,
liveness) is a different problem with its own models and data.

**What the gate does.** Nothing specific. A screen photo often shows moiré or glare and is then rejected
for that reason, but that is luck, not detection.

**Open.** Presentation-attack detection belongs before or beside Gate 1 and is not in this block. The
reading blocks must not treat "accepted by the gate" as "authentic".

## 2. A very small document is called "no document" instead of "too far"

**What happens.** When the card fills less than about a sixth of the frame width (an ID under ~270 px
wide in a 1600 px photo), Gate 1 no longer finds enough of its features and the photo is rejected as
`no_document` («ما كدرنا نشوف مستند واضح بالصورة …»). Slightly larger documents (text 5–7 px tall) are
recognised and rejected as `too_small` («المستند بعيد وصغير بالصورة …»), which is the better advice.

**Why.** Recognition needs detail that a tiny document does not have; the window in which a document is
still recognisable but already too small to read is narrow.

**What the gate does.** The customer is asked to retake in both cases, so nothing unreadable goes on.
Only the wording is less precise.

**Open.** A dedicated "document-shaped but tiny" check on the outline alone could give the better
message.

## 3. Thresholds are starting values, not measured on customer photos

**What happens.** The limits that turn a measurement into mild / moderate / severe were set on the
templates, on simulated photos, on the project dataset and on ten real phone photos. Real customer photos
(other phones, compression, lighting) may sit differently: some readable photos may be sent to DocRes or
back to the customer, or a borderline one accepted.

**What the gate does.** Every value is in `config/quality_thresholds.yaml` and reported with every
result, and `python -m gate_block calibrate` suggests new limits from labelled photos. Unmeasurable
values are `unknown`, never "fine".

**Open.** Calibrate on real uploads, best against TeleOCR outcomes (did OCR read the fields correctly?)
once Block 3 exists.

## 4. Laya can only be as good as the rules it was taught by

**What happens.** The base multilingual Laya, asked zero-shot, accepted almost every photo, severe blur
and glare included. The gate therefore lists the branches the measurements permit in Laya's state, and
Laya has to be fine-tuned (outside this repository, on the rows in `datasets/laya/`) to choose between
them — but those labels come from the gate's own deterministic rules (applied to simulated photos, the
dataset, the real photos and synthetic measurement states). A Laya trained on them learns to imitate the
rules' choices; it cannot discover a better decision than they encode.

**What the gate does.** Laya's choice is validated against the measurements before anything happens
(`rules_override` when it picks a branch the measurements do not permit), and the safety rules always
win. So a wrong Laya answer degrades to the rules' answer, never to a dangerous one.

**Open.** Retrain on outcome labels — whether TeleOCR read each photo correctly, whether DocRes helped —
so Laya can learn where the rules are too strict or too lenient.

## 5. Camera shake across the text lines is under-measured

**What happens.** Shake *along* the text lines (the common hand movement) is caught reliably: a 15 px
smear is moderate blur on every template. Shake *across* the lines is measured weaker (a 21 px vertical
smear reads as mild on some templates), because the derivative balance of the template's own print, the
reference for the motion ruler, varies more in that direction between one card's text and another's.

**What the gate does.** Mild blur still goes to DocRes (`deblurring`), so such a photo is not accepted
as it is; it is only not rejected outright.

**Open.** A motion ruler built from many cards of each type instead of one template.

## 6. Curled paper: small curls are measured coarsely, strong curls hide covering objects

**What happens.** On the paper licence, a bow of 2–6 % of the height reads between none and mild with
some scatter; from 8 % it is reliably moderate and goes to `dewarping`. A strongly curled sheet also
moves its print far from where a flat sheet would put it and shows the table where the sheet was
expected; before this was handled, an 8–10 % bow on a wooden table was read as a hand or an object
covering the licence (`reject:obstructed`) instead of `enhance:dewarping`.

**Why.** The homography that aligns the photo absorbs part of a gentle bow; what is left is close to the
precision of the block-wise alignment. A strong bow is beyond what the covering-object search can
tolerate.

**What the gate does.** Small curls barely disturb OCR; the ones that bend text lines across field
boxes are caught. When the curl is moderate or worse, covering objects are not looked for
(`obstruction: unknown`, with a warning) and the photo goes to `dewarping`; the recheck of the flattened
image measures coverage again, so a hand on a curled sheet is still caught one round later.

**Open.** The curl reading depends on the alignment, and the alignment on the template's static print:
widening the licence's head-office box to cover the longest founders' lines (three names or «وشركاهم»,
which run past the box's left edge on about a third of the print run) moved a 10 % bend from moderate
to mild in testing, and it was then read as a covering object again. The box was left as it is, so the
left end of those lines is measured at document level only. A curl measure that does not rest on the
blocks whose correlation succeeds (those are the least bent ones) would remove this sensitivity.

## 7. A very deep shadow with strong blur can be reported as a covering object

**What happens.** Objects lying on the document are found by **missing print**: where the template has
structure and the photo does not. This catches phones, pens, keys and papers (which skin colour cannot):
in the project's "clutter" photo a phone lies across the top of the licence and a mouse over its left
edge, and the photo is rejected as `obstructed`. A shadow darkens the print but keeps its structure, so
ordinary shadows are not mistaken for objects. A very deep shadow combined with strong blur, however,
can remove enough structure to be read as an object.

**What the gate does.** Then the customer is asked to retake with the "something covers the document"
message instead of the shadow or blur one: the decision (retake) is right, the advice less precise.

## 8. Unsupported documents and other countries' IDs

**What happens.** Only the four sides in `templates/` are recognised. Another document (a driving
licence, a passport, an ID of another country, an older Iraqi ID design) is rejected as `no_document`
or, if it resembles a template, `wrong_document`.

**What the gate does.** It never sends an unrecognised document to OCR or to DocRes.

**Open.** A new document type is a new template folder (clean image + `layout.json`); see
[MEASUREMENTS.md](MEASUREMENTS.md#templates).
