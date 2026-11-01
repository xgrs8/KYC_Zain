"""T-NORM vectors (6.7) - ALG side (identical vendored rfalib)."""
from __future__ import annotations

from rfalib.normalize import normalize, same


def test_basic_and_whitespace():
    assert normalize("بغداد") == "بغداد"
    assert normalize(" بغداد \n") == "بغداد"
    assert normalize("أ   ب") == "أ ب"


def test_marks_tatweel_zero_widths():
    assert normalize("زَيْنَب") == "زينب"
    assert normalize("الـسـيـد") == "السيد"
    for ch in ("\u200b", "\u200c", "\u200e", "\u200f", "\ufeff"):
        assert normalize(f"نور{ch}") == "نور"


def test_never_normalize_distinguishing_letters():
    assert normalize("علي") != normalize("على")
    assert normalize("امر") != normalize("آمر")
    assert normalize("امه") != normalize("أمه")


def test_same_semantics():
    assert same("مُحمد", "محمد")
    assert same("عليـي", "عليي")
    assert not same("علي", "على")
