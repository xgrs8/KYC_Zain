"""Configuration: one YAML per section, defaults in code, typos never pass silently."""
import pytest
import yaml

from gate_block.config import ConfigError, GateConfig
from gate_block.decision_tree import DEFAULT_TREE, load_tree
from gate_block.messages import DEFAULT_MESSAGES_AR, load_messages, message_for, validate_messages
from gate_block.types import DOCRES_TASKS, GATE1_REASONS, REJECT_REASONS


def test_shipped_yaml_equals_code_defaults():
    cdir = GateConfig.default_dir()
    assert cdir.is_dir()
    assert GateConfig.load_dir(cdir).to_dict() == GateConfig.from_dict(None).to_dict()


def test_every_section_file_exists_and_lists_every_key():
    cdir = GateConfig.default_dir()
    defaults = GateConfig.from_dict(None).to_dict()
    for fname, section in GateConfig.SECTION_FILES.items():
        content = yaml.safe_load((cdir / fname).read_text(encoding="utf-8"))
        assert set(content) == set(defaults[section]), fname


def test_missing_file_keeps_defaults(tmp_path):
    (tmp_path / "rules.yaml").write_text("enhance_at: moderate\n", encoding="utf-8")
    cfg = GateConfig.load_dir(tmp_path)
    assert cfg.rules.enhance_at == "moderate"
    assert cfg.quality.blur_ratio == GateConfig.from_dict(None).quality.blur_ratio


def test_unknown_key_is_an_error(tmp_path):
    (tmp_path / "rules.yaml").write_text("enhance_att: mild\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="enhance_att"):
        GateConfig.load_dir(tmp_path)


@pytest.mark.parametrize("data, needle", [
    ({"quality": {"blur_ratio": [0.2, 0.1, 0.3]}}, "blur_ratio"),            # must rise
    ({"quality": {"dark_ratio": [0.3, 0.5, 0.6]}}, "dark_ratio"),            # must fall
    ({"quality": {"blur_ratio": [0.1, 0.2]}}, "blur_ratio"),                 # three numbers
    ({"quality": {"blur_ratio": "high"}}, "blur_ratio"),
    ({"quality": {"text_px": [5.0, 7.0, 10.0]}}, "text_px"),                  # must fall
    ({"quality": {"structure_lost": 0.7}}, "structure_lost"),
    ({"rules": {"enhance_at": "sometimes"}}, "enhance_at"),
    ({"rules": {"reject_unrecoverable_at": {"wrinkles": "mild"}}}, "wrinkles"),
    ({"laya": {"on_uncertain": "accept"}}, "on_uncertain"),
    ({"laya": {"min_decision_confidence": 1.5}}, "min_decision_confidence"),
    ({"routing": {"on_enhancement_failure": "accept"}}, "on_enhancement_failure"),
    ({"id_merge": {"resample": "nearest"}}, "resample"),
    ({"recognition": {"recognized_score": 0.1}}, "recognized_score"),
    ({"quality": 3}, "quality"),
])
def test_invalid_values_are_refused(data, needle):
    with pytest.raises(ConfigError, match=needle):
        GateConfig.from_dict(data)


def test_missing_config_directory(tmp_path):
    with pytest.raises(ConfigError):
        GateConfig.load_dir(tmp_path / "nope")


# ---------------------------------------------------------------------- messages
def test_a_message_exists_for_every_code():
    for code in REJECT_REASONS + GATE1_REASONS + ("review",):
        assert DEFAULT_MESSAGES_AR[code].strip(), code
    validate_messages(DEFAULT_MESSAGES_AR)


def test_the_blur_message_is_the_agreed_wording():
    assert DEFAULT_MESSAGES_AR["blur"] == "الصورة مغوشة من فضلك ارجع التقط الصورة بدون تغويش"


def test_messages_are_arabic():
    for code, text in DEFAULT_MESSAGES_AR.items():
        arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
        assert arabic > 0.6 * sum(1 for ch in text if ch.isalpha()), code


def test_shipped_messages_equal_defaults():
    assert load_messages(GateConfig.default_dir()) == DEFAULT_MESSAGES_AR


def test_message_overlay_and_validation(tmp_path):
    (tmp_path / "messages_ar.yaml").write_text('blur: "نص جديد"\n', encoding="utf-8")
    msgs = load_messages(tmp_path)
    assert msgs["blur"] == "نص جديد" and msgs["glare"] == DEFAULT_MESSAGES_AR["glare"]
    (tmp_path / "messages_ar.yaml").write_text('blurr: "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_messages(tmp_path)
    (tmp_path / "messages_ar.yaml").write_text('blur: ""\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_messages(tmp_path)


def test_message_names_the_side_only_when_one_side_is_affected():
    base = DEFAULT_MESSAGES_AR["blur"]
    assert message_for("blur", DEFAULT_MESSAGES_AR, []) == base
    assert message_for("blur", DEFAULT_MESSAGES_AR, ["front", "back"]) == base
    one = message_for("blur", DEFAULT_MESSAGES_AR, ["back"])
    assert one != base and base in one


# ---------------------------------------------------------------------- decision tree
def test_shipped_tree_equals_defaults():
    tree = load_tree(GateConfig.default_dir())
    assert {k: list(getattr(tree, k)) for k in DEFAULT_TREE} == {k: list(v) for k, v in DEFAULT_TREE.items()}
    assert sorted(tree.operation_order) == sorted(DOCRES_TASKS)
    assert sorted(tree.reason_priority) == sorted(REJECT_REASONS)


@pytest.mark.parametrize("text", [
    "operation_order: [dewarping, deblurring]\n",                                   # a task is missing
    "operation_order: [dewarping, deblurring, deshadowing, appearance, sharpen]\n",  # unknown task
    "reason_priority: [blur, blur]\n",
    "rules_never_choose: [crop]\n",
    "something_else: 1\n",
])
def test_tree_overlay_is_validated(tmp_path, text):
    (tmp_path / "decision_tree.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_tree(tmp_path)
