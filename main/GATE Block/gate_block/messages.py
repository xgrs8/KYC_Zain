"""Iraqi-Arabic messages shown to the user: the leaves of the response tree.

Python picks the message for the reason that was decided; no model ever writes text to the user. The
defaults live here, so the gate never fails for lack of a file, and ``config/messages_ar.yaml`` overrides
them (edit the YAML to change the wording). The blur message is the project owner's own wording.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import yaml

from .config import ConfigError
from .types import GATE1_REASONS, REJECT_REASONS

DEFAULT_MESSAGES_AR: Dict[str, str] = {
    # Gate 1: is it the right document?
    "no_document": "ما كدرنا نشوف مستند واضح بالصورة. من فضلك ارجع التقط الصورة وخلي المستند كله ظاهر وواضح.",
    "wrong_document": "هذا مو المستند المطلوب. من فضلك ارفع المستند الصحيح.",
    "wrong_side": "هذا مو الوجه المطلوب من الهوية. من فضلك ارفع الوجه الأمامي بمكانه والوجه الخلفي بمكانه.",
    "unreadable_file": "ما كدرنا نفتح الصورة. من فضلك ارفعها مرة ثانية.",
    "image_too_small": "الصورة صغيرة كلش. من فضلك ارفع صورة أوضح وأكبر.",
    "blank_image": "الصورة فارغة أو سودة. من فضلك ارجع التقط الصورة.",
    # quality reasons: the reject branch of the decision tree
    "blur": "الصورة مغوشة من فضلك ارجع التقط الصورة بدون تغويش",
    "glare": "اكو لمعة أو انعكاس ضوء على المستند. من فضلك ارجع التقط الصورة بدون لمعة.",
    "dark": "الصورة مظلمة وما واضحة. من فضلك ارجع التقط الصورة بمكان بيه ضوه زين.",
    "overexposed": "الضوه قوي كلش على المستند والكتابة ما واضحة. من فضلك ارجع التقط الصورة بضوه أخف.",
    "shadow": "اكو ظل على المستند. من فضلك ارجع التقط الصورة بمكان ما بيه ظل.",
    "cut_off": "المستند مقطوع من الصورة. من فضلك ارجع التقط الصورة وخلي المستند كله داخل الإطار.",
    "angle": "زاوية التصوير مايلة كلش. من فضلك صوّر المستند من فوق بشكل مستقيم.",
    "too_small": "المستند بعيد وصغير بالصورة. من فضلك قرّب الكاميرا شوية وارجع التقط الصورة.",
    "obstructed": "اكو شي، مثل الإصبع، مغطي جزء من المستند. من فضلك ارجع التقط الصورة وخلي المستند مكشوف.",
    "other": "الصورة مو واضحة بما يكفي. من فضلك ارجع التقط الصورة.",
    # human review
    "review": "الصورة راح يراجعها موظف للتأكد منها.",
}
SIDE_LABELS_AR: Dict[str, str] = {"front": "الوجه الأمامي", "back": "الوجه الخلفي"}

REQUIRED_CODES = tuple(REJECT_REASONS) + tuple(GATE1_REASONS) + ("review",)
assert set(REQUIRED_CODES) == set(DEFAULT_MESSAGES_AR), "every reason needs exactly one default message"
_ARABIC = re.compile(r"[؀-ۿ]")


class MessageError(ConfigError):
    pass


def validate_messages(messages: Dict[str, str]) -> None:
    for code in REQUIRED_CODES:
        text = messages.get(code)
        if not isinstance(text, str) or not text.strip():
            raise MessageError(f"the message for {code!r} is missing or empty")
        if not _ARABIC.search(text):
            raise MessageError(f"the message for {code!r} contains no Arabic text")


def load_messages(config_dir: Union[str, Path, None] = None) -> Dict[str, str]:
    """The default messages, overlaid with ``<config_dir>/messages_ar.yaml`` when that file exists."""
    messages = dict(DEFAULT_MESSAGES_AR)
    if config_dir is not None:
        fp = Path(config_dir) / "messages_ar.yaml"
        if fp.is_file():
            with open(fp, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise MessageError(f"{fp} must contain a mapping of code -> message")
            for code, text in data.items():
                if code not in DEFAULT_MESSAGES_AR:
                    raise MessageError(f"unknown message code {code!r} in {fp} (valid: {sorted(DEFAULT_MESSAGES_AR)})")
                messages[code] = text
    validate_messages(messages)
    return messages


def message_for(code: str, messages: Dict[str, str], sides: Optional[Sequence[str]] = None) -> str:
    """The message for ``code``. When exactly one side of the ID is affected, its name is added in brackets."""
    base = messages[code]
    unique = set(sides or ())
    if len(unique) == 1:
        side = next(iter(unique))
        if side in SIDE_LABELS_AR:
            return f"{base} ({SIDE_LABELS_AR[side]})"
    return base
