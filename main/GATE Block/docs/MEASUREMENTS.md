# What the gate measures

Everything is measured on the document **after** Gate 1 has found it and the photo has been warped onto
the template (the "canonical" image, 1000 px on the long side). Wherever a value depends on the design
of the card — a guilloche background is not plain paper, a bold font is not a thin one — it is measured
**relative to the template**, so one set of thresholds works for all four sides.

Each value becomes a severity: `none`, `mild`, `moderate` or `severe`, using the three limits in
`config/quality_thresholds.yaml` (a value exactly on a limit gets the worse severity). A value that
cannot be measured is `unknown` and never counts as "fine".

## Document level

| Defect | Value | How it is measured | Limits (mild / moderate / severe) |
|---|---|---|---|
| `cut` | share of the document outside the photo | the template outline projected into the photo and clipped by the frame | 0.01 / 0.025 / 0.06 |
| `low_resolution` | height of the smallest printed text, in pixels **of the upload** | the template's text height (25th percentile over its fields, measured from the ink rows) × the photo's pixel density at the worst corner | 10 / 7 / 5 px (lower is worse) |
| `blur` | blur as a share of the text height | see below | 0.12 / 0.17 / 0.22 |
| `dark` | brightness of the paper relative to the template | lit-curve of the print (robust to the camera's tone curve) | 0.65 / 0.50 / 0.35 (lower is worse) |
| `low_contrast` | ink/paper contrast relative to the template, corrected for the light | | 0.55 / 0.40 / 0.28 (lower is worse) |
| `shadow` | how much darker the darkest region is than the lit paper | cell grid compared with the template, after removing the tone curve | 0.20 / 0.34 / 0.52 |
| `glare` | share of the printed content washed out by a reflection | clipped, colourless pixels (HSV) where the template has ink | 0.008 / 0.04 / 0.10 |
| `perspective` | mismatch of opposite sides of the outline | camera angle; reported, and a reason for a retake only when it has done damage (far side blurred or too few pixels) | 0.08 / 0.18 / 0.35 |
| `curl` | local misalignment of paper, share of the long side | block-wise phase correlation against the template after the homography (paper documents only; rigid cards: not applicable) | 0.02 / 0.04 / 0.07 |
| `noise` | sensor noise above the template's own texture (grey levels) | | 3 / 5 / 8 |
| `obstruction` | share of the document covered by a hand or an object | a hand by skin colour (after removing the colour of the light; the portrait slot of the ID is ignored), any object by **missing print**: the local structure of the template is absent where something lies on the card. Shadows keep the structure, so they are not mistaken for objects. On paper curled moderately or worse it is `unknown` until the sheet is dewarped | 0.05 / 0.10 / 0.20 |

### Blur, and why it is a share of the text height

Sharpness is the contrast-normalised slope of the strong edges in the text regions. A Gaussian blur of
σ px turns a step into a ramp whose slope is about 0.4/σ of its contrast; dividing by the local contrast
makes the reading independent of exposure and ink colour. Each template carries its own **blur ruler**:
the same reading on the template blurred by σ = 0 … 7 px. A photo's reading is converted to "the σ that
would do this to the template".

Two corrections make the number mean "how hard is this to read":

* **Resolution is not blur.** A sharp photo taken from far away looks soft once enlarged to the canonical
  size, only because it holds few pixels. The softness its pixel count explains (the lens, plus
  interpolation when enlarged) is removed in quadrature. What the pixel count costs is reported as
  `low_resolution` instead — DocRes cannot add pixels that were never taken.
* **The same blur is worse on small print.** σ is divided by the height of the smallest printed text, so
  3 px of blur is a problem on the licence's 15 px print and not on the ID's 26 px print.

**Camera shake** smears the strokes it crosses while the strokes along it stay crisp, and on Arabic
print those (long horizontal strokes) are the majority, so a median edge reading barely moves. Each
template therefore also carries a **motion ruler**: the balance of derivative energy between perpendicular
directions (0°, 45°, 90°, 135°) after smearing the template by 0 … 34 px in each direction. Exposure,
contrast, noise and even blur leave that balance nearly unchanged; a smear moves it a lot. The photo's
balance is read off the ruler as a smear length L, whose Gaussian equivalent L/√12 counts when it is the
larger blur. The report gives `motion_px` and `motion_direction_deg`.

### Text size

`low_resolution` is the height of the smallest printed text in pixels of the upload: what a reader gets
to see. For the merged ID the halves are measured in the pixels of the original uploads, so the merge's
enlargement adds nothing. The limits come from reading real photos: the project's hand-held licence photo
has 8 px text and is readable (mild); under 7 px Arabic dots and digits merge (moderate → retake).

## Field level

Every field box of `layout.json` is checked on its own, because one glare spot on the ID number matters
more than a large one on the background:

| Cause | Value |
|---|---|
| `cut_off` | share of the box outside the photo |
| `obstructed` | share of the box covered by a hand or an object |
| `glare` | share of the box washed out |
| `blur` | the field's own blur ruler, relative to the field's own text height (capped near the document's blur, because one box reads only a few words) |
| `shadow`, `low_contrast` | the light falling on the box, and the contrast relative to the template blurred to the measured blur |

A field is `good`, `degraded` or `lost`. Critical fields (`"critical": true` in `layout.json`: the ID
number, names, dates, the tax number …) decide more: a lost critical field forces a retake whatever
else the photo looks like (`config/rules.yaml`, `field_loss_at`).

## Gate 1: recognition

Candidates come from document-shaped outlines and from ORB keypoints spread over the whole photo. Each
candidate is scored by feature matches against the template's static print (RANSAC homography, matches
must span the card), layout similarity of the rectified candidate, colour and aspect. A photo is
`recognized` when the best score reaches 0.55 and leads the runner-up template by 0.05; `unsupported`
under 0.25 (or when it is a sharp, detailed picture that matches nothing); `uncertain` in between. The
expected slot matters: a tax card in the licence slot is `wrong_document`, an ID back in the front slot
is `wrong_side` (or, with `auto_swap_sides`, swapped when both sides were uploaded the wrong way round).

## Templates

`templates/<doc_type>/` holds one clean image per side (`front.jpg` / `back.jpg`, or `single.jpg`) and
`layout.json`:

```json
{"doc_type": "tax_card", "rigid": true,
 "sides": {"single": {"card": [x0, y0, x1, y1],
                      "fields": {"taxpayer_number": {"box": [x0, y0, x1, y1], "critical": true}, "...": {}},
                      "ignore": [[x0, y0, x1, y1]]}}}
```

Coordinates are fractions of the image. `card` is the document inside the image, `fields` the boxes to be
read, `ignore` regions whose content changes from card to card in a way that is not text (the portrait,
a stamp). `rigid: false` marks paper that can curl. The variable content of the fields is not used for
recognition; only the static print is. `python -m gate_block overlay photo.jpg --out o.png` draws what
the gate found, which is the quickest check after editing a layout.

## Calibrating the thresholds

The limits above are starting values tuned on the templates, simulated photos, the project dataset and
the real phone photos in the project. To fit them to your own capture conditions, put labelled photos in
one folder per condition (`good/`, `blur/`, `glare/` …) and run

```powershell
python -m gate_block calibrate PHOTOS_DIR --out calibration
```

It runs the gate on every photo, writes every measured value to `calibration/measurements.csv`, prints a
summary per folder, and writes suggested limits to `calibration/suggested_thresholds.yaml` that keep the
good photos on the good side. Review the suggestion before copying it into `config/`.
