"""Find the photos in a folder and work out what each one is (shared by the calibration and export tools).

Recognised names:

    ID_23.png                     a merged national ID, as A.L.G writes it (front on top, back below)
    TAX_07.jpg                    a tax card
    LIC_21.png                    a business licence
    anything_front.jpg +          the two uploads of one national ID
    anything_back.jpg
    anything else                 a single photo; Gate 1 says what it is

The first folder below the root is the sample's *label* (``good/``, ``blur/`` ...); files directly in the
root have the label ``unlabeled``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from ..templates import IMAGE_EXT

_SIDE = re.compile(r"^(?P<stem>.+)[_\- ](?P<side>front|back)$", re.IGNORECASE)


@dataclass
class Sample:
    name: str                    # unique within the folder, without extension
    label: str                   # first folder below the root, or "unlabeled"
    kind: str                    # merged_id | id_pair | photo
    path: Path                   # the image (the front upload for an id_pair)
    back: Optional[Path] = None  # the back upload, for an id_pair
    expected: Optional[str] = None   # national_id | tax_card | business_license, when the name says so


def iter_samples(root) -> Iterator[Sample]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT)
    by_key = {(p.parent, p.stem.lower()): p for p in files}
    for p in files:
        rel = p.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else "unlabeled"
        name = str(rel.with_suffix("")).replace("\\", "/")
        m = _SIDE.match(p.stem)
        if m:
            if m.group("side").lower() == "back":
                partner = next((by_key[k] for k in ((p.parent, f"{m.group('stem')}{sep}front".lower()) for sep in "_- ") if k in by_key), None)
                if partner is not None:
                    continue                                   # emitted together with its front
            else:
                back = next((by_key[k] for k in ((p.parent, f"{m.group('stem')}{sep}back".lower()) for sep in "_- ") if k in by_key), None)
                if back is not None:
                    yield Sample(name=str(rel.parent / m.group("stem")).replace("\\", "/").lstrip("./"), label=label,
                                 kind="id_pair", path=p, back=back, expected="national_id")
                    continue
        upper = p.stem.upper()
        if upper.startswith("ID_"):
            yield Sample(name, label, "merged_id", p, expected="national_id")
        elif upper.startswith("TAX_"):
            yield Sample(name, label, "photo", p, expected="tax_card")
        elif upper.startswith("LIC_"):
            yield Sample(name, label, "photo", p, expected="business_license")
        else:
            yield Sample(name, label, "photo", p)


def run_sample(gate, sample: Sample, workdir=None):
    """Run the gate on one sample; returns a GateResult."""
    if sample.kind == "merged_id":
        return gate.process_merged_id(sample.path, workdir=workdir)
    if sample.kind == "id_pair":
        return gate.process_id(sample.path, sample.back, workdir=workdir, name=Path(sample.name).name)
    return gate.process_image(sample.path, expected=sample.expected, workdir=workdir)
