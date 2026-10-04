"""Guard against U+FFFD replacement characters in the OKF bundle.

A U+FFFD means a character was lost on the way into the file: the bytes were
not valid UTF-8 and a loader decoded them with ``errors='replace'``, or a
transcoding step already baked the mark into the text. The bundle is embedded
and quoted verbatim, so the mark reaches a generated report as a black diamond
where the original ``knowledge/*.md`` had a real character (here, U+2014 EM
DASH). These tests pin the detector used by ``scripts/validate_okf.py``.
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

REPLACEMENT_CHAR = "\ufffd"
EM_DASH = "\u2014"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_okf", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _bundle_files():
    return sorted(p for p in BUNDLE_DIR.rglob("*.md") if p.is_file())


# ---- the real bundle is clean ---------------------------------------------
def test_bundle_has_no_replacement_chars():
    offenders = {}
    for path in _bundle_files():
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            offenders[str(path.relative_to(PROJECT_ROOT))] = [f"not valid UTF-8: {exc}"]
            continue
        hits = validator.find_replacement_chars(text)
        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = hits
    assert not offenders, (
        "U+FFFD replacement characters found in the OKF bundle "
        "(a character was lost from the source text; restore it from "
        f"knowledge/*.md): {offenders}"
    )


def test_bundle_files_are_valid_utf8():
    bad = []
    for path in _bundle_files():
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            bad.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
    assert not bad, f"bundle files that are not valid UTF-8: {bad}"


# ---- the detector catches what it should ----------------------------------
def test_replacement_char_in_body_is_detected():
    text = f"1. **Clinical Summary** {REPLACEMENT_CHAR} a 2-3 sentence summary\n"
    hits = validator.find_replacement_chars(text)
    assert len(hits) == 1
    lineno, col, snippet = hits[0]
    assert lineno == 1
    # snippet is a +/-20 character window centred on the mark
    assert col == text.index(REPLACEMENT_CHAR) + 1
    assert REPLACEMENT_CHAR in snippet
    assert "Clinical Summary" in snippet


def test_replacement_char_in_frontmatter_is_detected():
    text = (
        "---\n"
        f"type: Concept {REPLACEMENT_CHAR}dash\n"
        "---\n"
        "body is clean\n"
    )
    hits = validator.find_replacement_chars(text)
    assert [h[0] for h in hits] == [2]


def test_every_replacement_char_on_a_line_is_reported():
    text = f"a{REPLACEMENT_CHAR}b{REPLACEMENT_CHAR}c\nd{REPLACEMENT_CHAR}e\n"
    assert len(validator.find_replacement_chars(text)) == 3


def test_clean_text_has_no_hits():
    text = f"1. **Clinical Summary** {EM_DASH} a 2-3 sentence summary\n"
    assert validator.find_replacement_chars(text) == []


# ---- the em dashes match the originals -------------------------------------
@pytest.mark.parametrize(
    "bundle_rel, original_rel, needles",
    [
        (
            "knowledge/okf_bundle/reporting/report_template_guidelines.md",
            "knowledge/report_template.md",
            [
                "1. **Clinical Summary**",
                "2. **Imaging (MRI) Analysis**",
                "3. **Clinical Scores Interpretation**",
                "4. **Risk Interpretation**",
                "5. **Recommended Next Steps**",
                "6. **References / Knowledge Sources**",
                "7. **Disclaimer**",
            ],
        ),
        (
            "knowledge/okf_bundle/scores/cdr_scale.md",
            "knowledge/clinical_scores.md",
            [
                "- **CDR 0**",
                "- **CDR 0.5**",
                "- **CDR 1**",
                "- **CDR 2**",
                "- **CDR 3**",
            ],
        ),
    ],
)
def test_dash_lines_match_the_original_documents(bundle_rel, original_rel, needles):
    bundle_lines = (PROJECT_ROOT / bundle_rel).read_text(encoding="utf-8").splitlines()
    original_lines = (PROJECT_ROOT / original_rel).read_text(encoding="utf-8").splitlines()

    def first_starting_with(lines, prefix):
        for line in lines:
            if line.startswith(prefix):
                return line
        raise AssertionError(f"no line starting with {prefix!r} in {original_rel}")

    for needle in needles:
        got = first_starting_with(bundle_lines, needle)
        want = first_starting_with(original_lines, needle)
        assert REPLACEMENT_CHAR not in got, f"{bundle_rel}: {needle!r} still has U+FFFD"
        assert got == want, f"{bundle_rel}: {needle!r}\n  bundle:   {got!r}\n  original: {want!r}"


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