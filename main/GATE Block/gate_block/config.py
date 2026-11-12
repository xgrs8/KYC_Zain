"""Configuration of the Gate Block.

Every threshold lives here (and in one YAML file per section under ``config/``), never inline in the
algorithms. The defaults are *starting points* tuned on the reference templates and simulated photos:
run ``python -m gate_block calibrate`` on your own photos and put the suggested values in the YAML.

Boundary convention, used everywhere and stated once:
    "higher is worse" metrics:  value >= threshold  -> that severity
    "lower is worse"  metrics:  value <= threshold  -> that severity
so a value exactly on a threshold gets the *worse* severity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

Thr = Tuple[float, float, float]   # (mild, moderate, severe)
_SEV = ("mild", "moderate", "severe")


class ConfigError(ValueError):
    """Unknown key, wrong type or inconsistent thresholds."""


@dataclass
class ImageCfg:
    min_side: int = 240                # shorter side below this -> "image too small"
    max_pixels: int = 100_000_000      # safety cap against decompression bombs
    work_max_side: int = 2400          # working copy used for warping and measuring
    small_max_side: int = 900          # copy used to search for the document
    canonical_long_side: int = 1000    # every document is rectified to this long side


@dataclass
class RecognitionCfg:
    max_candidates: int = 6            # document-shaped quadrilaterals examined per photo
    min_area_frac: float = 0.025       # a candidate must cover at least this share of the frame
    aspect_sigma: float = 0.22         # tolerance (log ratio) when comparing aspect ratios
    orb_features: int = 1500           # keypoints detected in the rectified document
    orb_search_features: int = 6000    # keypoints spread over the whole photo when searching for the document
    orb_template_features: int = 6000  # keypoints detected in a template before keeping its static ones
    orb_static_max: int = 2500         # static template keypoints kept (strongest first)
    ratio: float = 0.78                # Lowe ratio test
    ransac_px: float = 4.0             # RANSAC reprojection threshold, canonical pixels
    flexible_ransac_scale: float = 3.0  # paper documents may curl: looser threshold for non-rigid templates
    min_inliers: int = 12              # fewer inliers than this = no feature evidence
    strong_inliers: int = 35           # this many inliers = full feature evidence
    min_inlier_spread: float = 0.025   # inliers count as evidence when they span this share of the template's keypoint area
    align_inlier_spread: float = 0.20  # ... and may set the alignment only when they span this much
    lf_min: float = 0.30               # layout similarity below this = no evidence
    lf_strong: float = 0.62            # ... at or above this = full evidence
    lf_certain: float = 0.80           # so similar that feature evidence is not needed (blurry photos)
    recognized_score: float = 0.55     # combined score needed for "recognized"
    margin: float = 0.05               # lead over the runner-up template needed for "recognized"
    unsupported_score: float = 0.25    # combined score must be below this for "unsupported"
    unsupported_max_direct_inliers: int = 22  # whole-photo matching at or above this vetoes "unsupported"
    rich_keypoints: int = 250          # a "detailed" picture has at least this many keypoints ...
    rich_sharpness: float = 0.28       # ... is sharp (edge sharpness at least this) ...
    rich_min_luminance: float = 70.0   # ... and is not dark


@dataclass
class QualityCfg:
    # severity thresholds: (mild, moderate, severe)
    blur_ratio: Thr = (0.12, 0.17, 0.22)         # blur (Gaussian sigma) / height of the smallest printed text; higher worse
    dark_ratio: Thr = (0.65, 0.50, 0.35)         # brightness relative to the template; lower worse
    contrast_ratio: Thr = (0.55, 0.40, 0.28)     # contrast relative to the template; lower worse
    shadow_depth: Thr = (0.20, 0.34, 0.52)       # darkest region vs the rest; higher worse
    glare_content: Thr = (0.008, 0.04, 0.10)     # share of the printed content hidden by glare; higher worse
    text_px: Thr = (10.0, 7.0, 5.0)              # height of the smallest printed text in photo pixels; lower worse
    perspective: Thr = (0.08, 0.18, 0.35)        # opposite-side length mismatch; higher worse
    curl: Thr = (0.02, 0.04, 0.07)               # local misalignment, share of the long side; higher worse
    noise_sigma: Thr = (3.0, 5.0, 8.0)           # noise above the template's own texture, grey levels; higher worse
    cut_fraction: Thr = (0.01, 0.025, 0.06)      # share of the document outside the frame; higher worse
    obstruction: Thr = (0.05, 0.10, 0.20)        # share of the document covered by a hand or an object; higher worse
    exposure_ref: float = 1.0                    # brightness ratio of a *good* photo (calibrate)
    glare_v: int = 245                           # glare = pixels at least this bright (HSV V, 0-255) ...
    glare_s: int = 45                            # ... and at most this colourful (HSV S, 0-255)
    # field level
    field_glare: Thr = (0.03, 0.12, 0.30)        # share of the field box saturated; higher worse
    field_covered: Thr = (0.08, 0.20, 0.45)      # share of the field box covered by a hand or an object; higher worse
    field_outside: Thr = (0.005, 0.03, 0.15)     # share of the field box outside the frame; higher worse
    field_min_edge_pixels: int = 60              # fewer strong-edge pixels -> field sharpness "unknown"
    field_pad: float = 0.0                       # padding added to field boxes (share of the document)
    # measuring instruments
    lens_sigma_px: float = 0.6                   # blur of a *sharp* phone photo in its own pixels (not a defect)
    default_text_height: float = 20.0            # text height assumed without a template layout, canonical px
    structure_lost: float = 0.35                 # print-structure agreement below this: something covers the print
    structure_weak: float = 0.55                 # ... and cells below this next to a covered cell belong to it
    structure_min_cells: int = 3                 # a covered patch smaller than this many cells is ignored


@dataclass
class RulesCfg:
    enhance_at: str = "mild"                     # a recoverable defect at/above this goes to DocRes
    reject_recoverable_at: str = "severe"        # blur / dark / low_contrast at/above this -> retake
    # defects DocRes has no task for, and the severity at which each forces a retake
    reject_unrecoverable_at: Dict[str, str] = field(default_factory=lambda: {
        "cut": "mild",
        "obstruction": "severe",
        "low_resolution": "moderate",
        "glare": "severe",
        "noise": "severe",
    })
    max_moderate_unrecoverable: int = 2          # this many moderate unrecoverable defects -> retake
    field_loss_at: str = "moderate"              # a field cut off / covered / in glare at this severity is lost
    reject_on_noncritical_field_loss: bool = True  # a lost non-critical field also forces a retake


@dataclass
class LayaCfg:
    min_decision_confidence: float = 0.55        # below this Laya's answer is not used
    on_uncertain: str = "rules"                  # rules | review : who decides when Laya's answer is unusable
    max_state_chars: int = 700                   # hard cap on the compact JSON handed to Laya


@dataclass
class RoutingCfg:
    max_enhance_rounds: int = 2                  # DocRes passes allowed per document
    on_enhancement_failure: str = "reject"       # reject | review : still not good after the last round
    docres_max_side: int = 1600                  # advice for Block 2 (DocRes input size limit)
    review_enabled: bool = True                  # false: a "review" becomes a retake request
    auto_swap_sides: bool = True                 # ID front and back uploaded the wrong way round are swapped


@dataclass
class IdMergeCfg:
    """Front + back merge of the national ID: the same rule as A.L.G's ``rfalib/merge.py``. With these
    values front.jpg + back.jpg give ID_23.png pixel for pixel."""
    target_width: int = 1600                     # = A.L.G's "merge_width" setting; the larger side is never downscaled
    gap_px: int = 24                             # white band between the two sides
    background: int = 255                        # grey level of the band
    resample: str = "lanczos"                    # lanczos | bicubic | bilinear


@dataclass
class PathsCfg:
    templates_dir: str = "templates"


@dataclass
class GateConfig:
    image: ImageCfg = field(default_factory=ImageCfg)
    recognition: RecognitionCfg = field(default_factory=RecognitionCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    rules: RulesCfg = field(default_factory=RulesCfg)
    laya: LayaCfg = field(default_factory=LayaCfg)
    routing: RoutingCfg = field(default_factory=RoutingCfg)
    id_merge: IdMergeCfg = field(default_factory=IdMergeCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    # one YAML file per section: file name -> section name
    SECTION_FILES = {
        "image.yaml": "image",
        "recognition.yaml": "recognition",
        "quality_thresholds.yaml": "quality",
        "rules.yaml": "rules",
        "laya.yaml": "laya",
        "routing.yaml": "routing",
        "id_merge.yaml": "id_merge",
        "paths.yaml": "paths",
    }

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "GateConfig":
        cfg = cls()
        if data:
            if not isinstance(data, dict):
                raise ConfigError("config root must be a mapping")
            _merge(cfg, data, "")
        cfg.validate()
        return cfg

    @classmethod
    def load_dir(cls, directory: Union[str, Path]) -> "GateConfig":
        """Load a config directory (one YAML per section, see SECTION_FILES).
        A missing file keeps the defaults of its section; an unknown key raises ConfigError."""
        d = Path(directory)
        if not d.is_dir():
            raise ConfigError(f"config directory not found: {d}")
        data: Dict[str, Any] = {}
        for fname, section in cls.SECTION_FILES.items():
            fp = d / fname
            if fp.is_file():
                with open(fp, "r", encoding="utf-8") as fh:
                    content = yaml.safe_load(fh) or {}
                if not isinstance(content, dict):
                    raise ConfigError(f"{fp} must contain a mapping")
                data[section] = content
        try:
            return cls.from_dict(data)
        except ConfigError as exc:
            raise ConfigError(f"{d}: {exc}") from exc

    @staticmethod
    def default_dir() -> Path:
        """``<repo>/config`` next to the ``gate_block`` package (may not exist)."""
        return Path(__file__).resolve().parent.parent / "config"

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        q = self.quality
        rising = ("blur_ratio", "shadow_depth", "glare_content", "perspective", "curl", "noise_sigma",
                  "cut_fraction", "obstruction", "field_glare", "field_covered", "field_outside")
        falling = ("dark_ratio", "contrast_ratio", "text_px")
        for name in rising:
            a, b, c = getattr(q, name)
            if not (a < b < c):
                raise ConfigError(f"quality.{name} must rise: mild < moderate < severe, got {[a, b, c]}")
        for name in falling:
            a, b, c = getattr(q, name)
            if not (a > b > c):
                raise ConfigError(f"quality.{name} must fall: mild > moderate > severe, got {[a, b, c]}")
        if q.exposure_ref <= 0:
            raise ConfigError("quality.exposure_ref must be > 0")
        if not (0 <= q.glare_v <= 255 and 0 <= q.glare_s <= 255):
            raise ConfigError("quality.glare_v and quality.glare_s must be within 0..255")
        if not (0.0 <= q.field_pad <= 0.05):
            raise ConfigError("quality.field_pad must be within 0..0.05")
        if not (0.0 <= q.lens_sigma_px <= 3.0) or q.default_text_height <= 0:
            raise ConfigError("quality.lens_sigma_px must be within 0..3 and quality.default_text_height > 0")
        if not (-1.0 < q.structure_lost < q.structure_weak < 1.0) or q.structure_min_cells < 1:
            raise ConfigError("quality.structure_lost < quality.structure_weak (both within -1..1), structure_min_cells >= 1")

        r = self.rules
        for label, val in (("rules.enhance_at", r.enhance_at), ("rules.reject_recoverable_at", r.reject_recoverable_at),
                           ("rules.field_loss_at", r.field_loss_at)):
            if val not in _SEV:
                raise ConfigError(f"{label} must be one of {list(_SEV)}, got {val!r}")
        from .types import DEFECTS
        for k, v in r.reject_unrecoverable_at.items():
            if k not in DEFECTS:
                raise ConfigError(f"rules.reject_unrecoverable_at: unknown defect {k!r}")
            if v not in _SEV:
                raise ConfigError(f"rules.reject_unrecoverable_at[{k}] must be one of {list(_SEV)}, got {v!r}")
        if r.max_moderate_unrecoverable < 1:
            raise ConfigError("rules.max_moderate_unrecoverable must be >= 1")

        if not (0.0 <= self.laya.min_decision_confidence <= 1.0):
            raise ConfigError("laya.min_decision_confidence must be within 0..1")
        if self.laya.on_uncertain not in ("rules", "review"):
            raise ConfigError("laya.on_uncertain must be 'rules' or 'review'")
        if self.laya.max_state_chars < 200:
            raise ConfigError("laya.max_state_chars must be >= 200")

        if self.routing.on_enhancement_failure not in ("reject", "review"):
            raise ConfigError("routing.on_enhancement_failure must be 'reject' or 'review'")
        if self.routing.max_enhance_rounds < 0:
            raise ConfigError("routing.max_enhance_rounds must be >= 0")

        m = self.id_merge
        if m.target_width < 64 or m.gap_px < 0 or not (0 <= m.background <= 255):
            raise ConfigError("id_merge: need target_width >= 64, gap_px >= 0 and background within 0..255")
        if m.resample not in ("lanczos", "bicubic", "bilinear"):
            raise ConfigError("id_merge.resample must be lanczos, bicubic or bilinear")

        rc = self.recognition
        if not (0 <= rc.unsupported_score < rc.recognized_score <= 1.0):
            raise ConfigError("recognition: need 0 <= unsupported_score < recognized_score <= 1")
        if not (0 < rc.min_inliers < rc.strong_inliers):
            raise ConfigError("recognition: need 0 < min_inliers < strong_inliers")
        if not (0 <= rc.lf_min < rc.lf_strong <= rc.lf_certain <= 1.0):
            raise ConfigError("recognition: need 0 <= lf_min < lf_strong <= lf_certain <= 1")
        if rc.max_candidates < 1 or rc.flexible_ransac_scale < 1.0:
            raise ConfigError("recognition: need max_candidates >= 1 and flexible_ransac_scale >= 1")
        if not (0 < rc.min_inlier_spread <= rc.align_inlier_spread <= 1.0):
            raise ConfigError("recognition: need 0 < min_inlier_spread <= align_inlier_spread <= 1")

        im = self.image
        if not (64 <= im.small_max_side <= im.work_max_side) or im.canonical_long_side < 400 or im.min_side < 32:
            raise ConfigError("image: need 64 <= small_max_side <= work_max_side, canonical_long_side >= 400, min_side >= 32")


# ---------------------------------------------------------------------- helpers
def _merge(obj: Any, data: Dict[str, Any], prefix: str) -> None:
    """Merge a plain dict into a dataclass. Unknown keys raise: a typo must never pass silently."""
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise ConfigError(f"unknown config key {prefix}{key} (valid here: {sorted(valid)})")
        current = getattr(obj, key)
        where = f"{prefix}{key}"
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{where} must be a mapping")
            _merge(current, value, where + ".")
        elif isinstance(current, tuple):
            if not isinstance(value, (list, tuple)) or len(value) != len(current) or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
                raise ConfigError(f"{where} must be a list of {len(current)} numbers")
            setattr(obj, key, tuple(float(x) for x in value))
        elif isinstance(current, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{where} must be a mapping")
            setattr(obj, key, {str(k): str(v) for k, v in value.items()})
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where} must be true or false")
            setattr(obj, key, value)
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ConfigError(f"{where} must be a whole number")
            setattr(obj, key, int(value))
        elif isinstance(current, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{where} must be a number")
            setattr(obj, key, float(value))
        else:
            if not isinstance(value, str):
                raise ConfigError(f"{where} must be text")
            setattr(obj, key, value)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [float(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    return obj
