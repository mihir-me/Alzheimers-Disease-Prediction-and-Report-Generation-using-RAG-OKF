"""Guard against fused words in the OKF bundle.

A fused word is two words glued together because a line break was dropped
instead of being replaced by a space (``"...language and\\nrecognition..."`` ->
``andrecognition``). The bundle is embedded and quoted verbatim, so one is a
defect. These tests pin the detector used by ``scripts/validate_okf.py``.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "knowledge" / "okf_bundle"
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_okf.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_okf", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _bundle_files():
    return sorted(p for p in BUNDLE_DIR.rglob("*.md") if p.is_file())


# ---- the real bundle is clean ---------------------------------------------
def test_bundle_has_no_fused_words():
    offenders = {}
    for path in _bundle_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = validator.find_fused_words(text)
        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = hits
    assert not offenders, (
        "fused words found in the OKF bundle "
        "(two words joined by a removed line break): "
        f"{offenders}"
    )


# ---- the detector catches what it should ----------------------------------
@pytest.mark.parametrize(
    "word",
    ["andrecognition", "ofdementia", "andlanguage", "imagingmodalities"],
)
def test_reported_fused_words_are_detected(word):
    assert validator.find_fused_words(f"prefix {word} suffix") == [(1, word)]


def test_every_known_fused_word_is_detected():
    missing = [
        word
        for word in sorted(validator.KNOWN_FUSED_WORDS)
        if not any(hit[1] == word for hit in validator.find_fused_words(f"x {word} y"))
    ]
    assert not missing, f"known fused words not detected by the checker: {missing}"


def test_fused_word_is_detected_in_frontmatter_and_body():
    text = (
        "---\n"
        "type: andrecognition\n"
        "---\n"
        "difficulty with language andrecognition here\n"
    )
    assert validator.find_fused_words(text) == [
        (2, "andrecognition"),
        (4, "andrecognition"),
    ]


def test_shape_only_real_words_are_not_flagged():
    text = (
        "withdrawal theory therefore orientation interpreted information "
        "individuals includes increasing indicators"
    )
    assert validator.find_fused_words(text) == []


def test_allowlist_and_known_lists_stay_disjoint():
    overlap = validator.KNOWN_FUSED_WORDS & validator.FUSED_WORD_ALLOWLIST
    assert not overlap, f"a token cannot be both known-fused and allowlisted: {overlap}"


# ---- the validator itself passes ------------------------------------------
def test_validate_okf_script_passes():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAILED" not in result.stdout
