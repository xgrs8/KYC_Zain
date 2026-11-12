# How the GATE Block works

## 1. Intake (`imageio.py`)

The upload is decoded (JPEG, PNG, WebP, BMP, TIFF; HEIC with `pillow-heif`), turned upright with its
EXIF orientation, and refused early when it cannot be opened (`unreadable_file`), is tiny
(`image_too_small`, shorter side under 240 px), blank or black (`blank_image`), or absurdly large
(decompression-bomb protection). Two working copies are made: one for searching (≤ 900 px) and one for
measuring (≤ 2400 px). The original is never modified.

## 2. Gate 1, the check gate (`recognition.py`, `templates.py`)

Is the photo one of the four supported sides — national ID front, national ID back, tax card, business
licence? Candidates come from document-shaped outlines and from keypoints spread over the whole photo;
each is scored against every template by ORB feature matches on the template's static print (a RANSAC
homography whose inliers must span the card), layout similarity of the rectified candidate, colour and
aspect ratio. The outcome is `recognized` (with the side and the homography), `uncertain` or
`unsupported`.

| Situation | Result | Message |
|---|---|---|
| no supported document | `reject:no_document` | «ما كدرنا نشوف مستند واضح بالصورة …» |
| a supported document, but not the one this slot expects | `reject:wrong_document` | «هذا مو المستند المطلوب …» |
| the ID's back in the front slot (or the front twice) | `reject:wrong_side` (or swapped, when both are swapped) | «هذا مو الوجه المطلوب من الهوية …» |
| some evidence, not enough | measured reason if the photo is bad, else `review:recognition_uncertain` | |

A photo Gate 1 does not recognise is never shown to Laya and never goes to OCR or DocRes.

## 3. The ID merge (`id_merge.py`)

Only after Gate 1 has confirmed front and back, the two uploads are merged the A.L.G way: both sides
scaled (LANCZOS) to a common width — `max(1600, front width, back width)`, so nothing is shrunk — front
on top, a 24 px white band, back below; PNG. The result for the project's sample pair is pixel-identical
to the reference `ID_23.png`. The result records the pixel rows of each side (`regions`), so DocRes can be
asked to work on one side only, and so a merged ID coming back from DocRes can be split again.

## 4. The quality gate (`quality.py`, `measures.py`, `report.py`)

Each side is warped onto its template and measured — the document as a whole (cut, text size, blur and
camera shake, darkness, contrast, shadow, glare, camera angle, curl, noise, a hand or an object on it)
and every field box on its own (cut off, covered, glare, blur, shadow, contrast). See
[MEASUREMENTS.md](MEASUREMENTS.md). The result is the JSON quality report (`result.report`):

```json
{"document_type": "tax_card",
 "quality_metrics": {"blur": {"value": 0.19, "severity": "moderate", "quality_score": 37.5, "reliable": true,
                              "extra": {"sigma_px": 3.6, "motion_px": 14.6, "motion_direction_deg": 0}, "worst_side": "single"},
                     "glare": {"value": 0.0, "severity": "none"}, "...": {}},
 "quality_by_side": {"single": {"...": {}}},
 "field_quality": {"taxpayer_number": {"critical": true, "status": "good", "severity": "none", "cause": null}, "...": {}},
 "document_geometry": {"single": {"frame": "complete", "rotation_deg": -2.1, "text_height_px": 21.6, "...": null}},
 "warnings": []}
```

## 5. The decision (`laya_state.py`, `laya_client.py`, `router.py`, `rules.py`, `decision_tree.py`)

The report is condensed into a short state (severities only, at most 700 characters) that also lists the
**permitted branches** (`allowed`) of the decision tree for this photo, and handed to Laya with three
questions: `decision` (accept / enhance / reject), `reject_reason` and `docres_task`. Laya answers all
three in one forward pass with a probability per option; its job is the judgement between permitted
branches. A photo with no permitted branch at all is decided by the rules without asking Laya.

Python then checks the answer against the same permitted branches:

| Branch | Permitted when |
|---|---|
| accept | nothing worse than a mild defect, no critical field in trouble, sharpness measurable |
| enhance with task T | the defect T is made for was measured (mild or worse), T was not applied yet, rounds remain (2) |
| reject with reason R | R was measured (moderate or worse) |

and around that:

* **Safety rules** (`rules.py`, `config/rules.yaml`) cannot be overridden: the document cut off, a critical
  field lost under glare / a finger / the frame edge, damage DocRes cannot repair (too few pixels, severe
  glare or noise, a large covered area), two or more moderate unrecoverable defects together.
* Laya's answer is not used when it fails, is malformed, truncated, or below the minimum confidence
  (0.55): the rules decide (`rules_fallback`), or a person with `on_uncertain: review`.
* A branch the measurements do not permit is replaced by the rules' choice (`rules_override`).

`result.decision.decided_by` says who owned the outcome (`laya`, `hard_rule`, `rules`, `rules_fallback`,
`rules_override`); `result.decision` also carries Laya's probabilities, its choice, whether it was
eligible, the rules' baseline and the trace of reasons.

The **decision tree** (`config/decision_tree.yaml`):

```
reject ─ cut_off · obstructed · too_small · glare · overexposed · dark · blur · shadow · angle · other
           (priority order; each with its fixed Iraqi-Arabic message in messages_ar.yaml)
enhance ─ dewarping · deblurring · deshadowing · appearance · binarization   (priority order; DocRes, one task per round)
accept  ─ TeleOCR
```

The rules (fallback and baseline) never choose `binarization` on their own; Laya may, when low contrast
was measured.

## 6. Hand-off

| Route | `next_block` | Request |
|---|---|---|
| accept | `teleocr` | `teleocr_request = {"image", "kind": "photo" or "merged_id", "document_type"}` |
| enhance | `docres` | `docres_request = {"task", "image", "kind", "round", "applied_before", "alternatives", "max_side", "regions"}` |
| reject | `retake` | `message_ar` (and `message_ar_with_side` naming the ID side when only one is affected) |
| review | `human_review` | `message_ar` (the review notice) |

After DocRes, `gate.recheck(enhanced_image, previous_result)` runs the gate again on the enhanced image
with the applied task remembered. The enhancement counts as a failure when the defect it targeted got
worse, when it introduced another harmful defect, or when the document is no longer recognised; then the
image from before is used again (to OCR if it was acceptable as it was, otherwise a retake). At most two
DocRes rounds; after the last one the photo is accepted if it is good enough, otherwise the customer is
asked to retake it.

## 7. A whole submission

`gate.process_submission({"id_front": …, "id_back": …, "tax_card": …, "business_license": …})` gates every
slot and returns `{"overall", "next_block", "documents"}`: one result per document, and an overall route
that is the worst of them (reject, then review, then enhance, then accept). An ID with only one side
uploaded is refused and the missing side is named.

## Design rules that keep the gate honest

* No model text reaches the customer: messages are fixed strings chosen by code.
* "Unknown" is never "fine": an unmeasurable value cannot make a photo acceptable.
* Every number lives in `config/`, is validated at start-up, and is reported with the result.
* Every decision carries its trace: who decided, on which measurements, what the alternatives were.
