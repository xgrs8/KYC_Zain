"""A.L.G label building (plan 8.1-8.5, D10, D12).

Human-driven labeling: the labeler types every field or marks it
[UNREADABLE]; nothing is auto-filled.  When the uploaded image carries a
truth sidecar, each field offers a "Use truth"
suggestion that must be accepted field-by-field.  Every acceptance, edit
and disagreement is logged in the label sidecar.

Generate stays disabled until every field is resolved (non-empty xor
[UNREADABLE]); on Generate the ID front+back are merged with
rfalib.merge and the label JSON is written in the user's exact format
(6.4).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from rfalib import merge as M
from rfalib import naming
from rfalib import sidecar as SC
from rfalib.constants import CONDITION_TAGS, UNREADABLE
from rfalib.normalize import same
from rfalib.schema import build_label, field_warnings, label_keys, validate_label
from rfalib.sidecar import find_truth_sidecar


class LabelSession:
    """One sample under labeling: field values, statuses, truth policy."""

    def __init__(self, doc_type: str, name: str, front: str,
                 back: str | None, workspace: Path, labeler: str = ""):
        self.doc_type = doc_type
        self.name = name
        self.front = str(front)
        self.back = str(back) if back else None
        self.workspace = Path(workspace)
        self.labeler = labeler
        self.fields: dict[str, str] = {k: "" for k in label_keys(doc_type)}
        self.status: dict[str, str] = {k: "typed" for k in self.fields}
        self.edited_from_truth: set[str] = set()
        self.conditions: list[str] = []
        self.warnings_overridden: list[str] = []
        truth_path = find_truth_sidecar(Path(self.front)) or (
            find_truth_sidecar(Path(self.back)) if self.back else None)
        self.truth: dict | None = None
        self.truth_ref: str | None = None
        if truth_path:
            try:
                self.truth = SC.read_json(truth_path)
                self.truth_ref = str(truth_path)
            except Exception:  # noqa: BLE001 - unreadable sidecar -> ignore
                self.truth = None

    # ------------------------------------------------------------ truth policy
    def truth_values(self) -> dict[str, str]:
        if not self.truth:
            return {}
        return {k: str(v) for k, v in (self.truth.get("printed") or {}).items()
                if k in self.fields}

    def use_truth(self, key: str) -> bool:
        """Fill one field with the sidecar's printed value (D10)."""
        values = self.truth_values()
        if key not in values:
            return False
        self.fields[key] = values[key]
        self.status[key] = "accepted-truth"
        self.edited_from_truth.discard(key)
        return True

    # ------------------------------------------------------------ editing
    def set_value(self, key: str, value: str) -> None:
        if key not in self.fields:
            raise KeyError(key)
        if self.status[key] == "accepted-truth" and not same(value, self.fields[key]):
            self.edited_from_truth.add(key)
            self.status[key] = "edited"
        elif self.status[key] == "unreadable" and value.strip():
            self.status[key] = "typed"
        self.fields[key] = value

    def mark_unreadable(self, key: str) -> None:
        if key not in self.fields:
            raise KeyError(key)
        self.fields[key] = UNREADABLE
        if self.status[key] == "accepted-truth":
            self.status[key] = "edited"   # disagreement: truth existed, unreadable now
            self.edited_from_truth.add(key)
        else:
            self.status[key] = "unreadable"

    def toggle_condition(self, tag: str) -> None:
        if tag not in CONDITION_TAGS:
            raise ValueError(tag)
        if tag in self.conditions:
            self.conditions.remove(tag)
        else:
            self.conditions.append(tag)

    # ------------------------------------------------------------ gates (8.4)
    def field_state(self, key: str) -> str:
        value = self.fields.get(key, "")
        if value == UNREADABLE:
            return "unreadable"
        if not value.strip():
            return "empty"
        if UNREADABLE in value:
            return "junk"      # mixed junk like "...x [UNREADABLE] y"
        return "filled"

    def resolution(self) -> tuple[int, int]:
        """(resolved, total) - resolved = filled or unreadable."""
        total = len(self.fields)
        resolved = sum(1 for k in self.fields
                       if self.field_state(k) in ("filled", "unreadable"))
        return resolved, total

    def gate_problems(self) -> dict[str, str]:
        """Blocking problems: empty fields and mixed junk (contract absolute)."""
        problems: dict[str, str] = {}
        for key in self.fields:
            state = self.field_state(key)
            if state == "empty":
                problems[key] = "Field is empty - type a value or mark [Unreadable]."
            elif state == "junk":
                problems[key] = (f"Mixed junk: a field is either a transcription "
                                 f"or exactly {UNREADABLE}.")
        return problems

    def warnings(self) -> dict[str, str]:
        """Warn-level regex mismatches (overridable, logged)."""
        return field_warnings(self.doc_type, {
            k: v for k, v in self.fields.items()
            if self.field_state(k) == "filled"})

    # ------------------------------------------------------------ outputs (8.5)
    def _output_base(self) -> str:
        base = naming.derive_output_name(self.name, self.doc_type, self.workspace)
        if not self._output_name_exists(base):
            return base

        # A.L.G keeps all results in one folder, so a deliberate counter reset
        # must not overwrite files from an earlier labeling run. Skip occupied
        # names until an unused sequence number is found.
        while True:
            number = naming.take_next(self.workspace, self.doc_type)
            base = naming.doc_name(self.doc_type, number)
            if not self._output_name_exists(base):
                return base

    def _output_name_exists(self, base: str) -> bool:
        output_dir = self.workspace / "labeled"
        return any((output_dir / f"{base}{suffix}").exists()
                   for suffix in (".png", ".jpg", ".label.json",
                                  ".label.sidecar.json"))

    def generate(self, merge_width: int = 1140, merge_gap: int = 24,
                 jpeg: bool = False, labeler: str | None = None) -> dict:
        """Merge (ID only), save the image(s) + label JSON + sidecar.

        Returns {"base": name, "image": path, "label": path, "sidecar": path}.
        Raises ValueError when the resolution gate is not satisfied.
        """
        problems = self.gate_problems()
        if problems:
            raise ValueError("Generate blocked - resolve every field: "
                             + "; ".join(f"{k}: {v}" for k, v in problems.items()))
        base = self._output_base()
        out_dir = self.workspace / "labeled"
        out_dir.mkdir(parents=True, exist_ok=True)
        photo_name = f"{base}.png"
        image_path = out_dir / photo_name

        merge_geom: dict = {}
        if self.doc_type == "ID":
            if not (self.front and self.back):
                raise ValueError("ID labeling needs both a front and a back image")
            merge_geom = M.merge_files(self.front, self.back, image_path,
                                       width=merge_width, gap=merge_gap,
                                       jpeg=out_dir / f"{base}.jpg" if jpeg else None)
        else:
            with Image.open(self.front) as im:
                im.convert("RGB").save(image_path, format="PNG")
            if jpeg:
                with Image.open(self.front) as im:
                    im.convert("RGB").save(out_dir / f"{base}.jpg",
                                           format="JPEG", quality=95)

        label = build_label(self.doc_type, photo_name, self.fields)
        problems2 = validate_label(label, self.doc_type)
        if problems2:
            raise ValueError("label invalid: " + "; ".join(problems2))
        label_path = out_dir / f"{base}.label.json"
        SC.write_json(label_path, label)

        # disagreements: accepted/edited truth that no longer matches (8.3)
        disagreements = []
        truth_values = self.truth_values()
        for key in self.fields:
            if key in truth_values and self.status[key] in ("accepted-truth", "edited"):
                if not same(self.fields[key], truth_values[key]):
                    disagreements.append({"field": key, "truth": truth_values[key],
                                          "label": self.fields[key]})

        split = None
        synthetic = False
        if self.truth:
            split = self.truth.get("split")
            synthetic = bool(self.truth.get("synthetic"))

        sidecar = SC.label_sidecar(
            doc_id=base, photo=photo_name, doc_type=self.doc_type,
            labeler=labeler or self.labeler,
            field_status=dict(self.status),
            truth_disagreements=disagreements or None,
            conditions=list(self.conditions) or None,
            truth_ref=self.truth_ref, split=split, synthetic=synthetic,
            warnings_overridden=self.warnings_overridden or None)
        sidecar_path = out_dir / f"{base}.label.sidecar.json"
        SC.write_json(sidecar_path, sidecar)
        return {"base": base, "image": str(image_path), "label": str(label_path),
                "sidecar": str(sidecar_path), "merge": merge_geom}

    def skip(self, reason: str, labeler: str | None = None) -> dict:
        """Skip requires a reason; writes the sidecar only (8.2 step 6)."""
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("Skip requires a reason")
        out_dir = self.workspace / "labeled"
        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = SC.label_sidecar(
            doc_id=self.name, photo=None, doc_type=self.doc_type,
            labeler=labeler or self.labeler,
            field_status={k: "typed" for k in self.fields},
            truth_disagreements=None,
            conditions=list(self.conditions) or None,
            truth_ref=self.truth_ref, split=None,
            synthetic=bool(self.truth and self.truth.get("synthetic")),
            skipped=reason)
        path = out_dir / f"{self.name}.label.sidecar.json"
        SC.write_json(path, sidecar)
        return {"sidecar": str(path)}
