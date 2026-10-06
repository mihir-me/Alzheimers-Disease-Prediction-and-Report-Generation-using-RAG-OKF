"""Generate the markdown clinical report (LLM with a template fallback).

Citation integrity rules enforced here:

* Every clinical statement (interpretation, recommendation, next step) is taken
  from the **body of the concept it cites**. The template generator selects a
  sentence out of the retrieved concept text; it never emits a hardcoded
  clinical claim. When no retrieved concept contains a statement, the statement
  is omitted. Purely structural text (headings, "Interpret in the full clinical
  context", the severity-band note, the disclaimer, the model probabilities)
  stays hardcoded and uncited.
* A **bare checklist is never quoted one item at a time**. When the selected
  statement is an item of a bullet list that has no lead-in of its own (the
  shape of ``diagnosis/red_flags_referral``: a list of referral criteria), the
  statement becomes one lead-in built from the concept's own title followed by
  every item of that list, e.g. "Red flags requiring specialist referral
  include: Rapidly progressive symptoms (weeks to months); Focal neurological
  signs; ...". Lifting a single criterion out of such a list would read as a
  statement about the patient ("Rapidly progressive symptoms"), which the source
  never claims. Lists that do carry a lead-in, and lists whose items are
  labelled definitions ("CDR 2 - Moderate dementia"), are unaffected.
* Any sentence tagged ``[concept: <id>]`` is only kept when at least
  :data:`CITATION_SUPPORT_MIN_COVERAGE` of its content words occur in that
  concept's body. Otherwise the citation is dropped and, in the template path,
  the whole sentence is dropped as well. The same guard runs over the LLM output
  and every drop is logged.
* The ``Sources`` section lists exactly the concepts cited inline. Retrieved
  concepts that are never cited are not listed there; the app shows them in its
  "Retrieved knowledge sources" expander, marked "retrieved, not cited".
* ``reporting/*`` concepts (the report structure guidelines) are formatting
  guidance for the generator, never clinical evidence: they are passed to the
  LLM as guidance only, are never cited inline and are never listed in
  ``Sources``.

Medication dosages are forbidden and the not-a-diagnosis disclaimer is always
the last thing in the report.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .config import RagConfig
from .vector_store import Hit

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[\s*concept:\s*([^\]\s]+)\s*\]", re.IGNORECASE)

# Minimum share of a sentence's content words that must occur in the body of the
# concept it cites. Below this the citation is considered unsupported.
CITATION_SUPPORT_MIN_COVERAGE = 0.4

# Dosage patterns that must never reach the report (the prompt forbids them,
# this is the safety net for models that ignore the instruction).
_DOSAGE_RE = re.compile(
    r"\b\d+(?:\s*[-–]\s*\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?)\b"
    r"(?:\s*/\s*(?:kg|day|dose))?"
    r"|\b(?:once|twice|three times|four times)\s+(?:a\s+)?(?:daily|day|week)\b",
    re.IGNORECASE,
)
_SOURCES_HEADING = "## Sources"
_DISCLAIMER = (
    "This report is generated automatically as decision-support output. It is "
    "not a clinical diagnosis. A qualified clinician must confirm any findings."
)
_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for Alzheimer's disease. "
    "You generate a structured markdown report from an automated model "
    "prediction, clinical scores, and a set of retrieved OKF knowledge concepts.\n"
    "Hard rules:\n"
    "1. Write every clinical statement by reusing the wording of the retrieved "
    "concept bodies. At least 40% of the content words of a clinical sentence "
    "must come from the body of the concept you cite; sentences that cannot be "
    "backed that way must be omitted, not reworded.\n"
    "2. Cite every clinical statement inline with the id of the concept that "
    "supports it, in the form [concept: <concept_id>] immediately after the "
    "sentence, e.g. CDR 0 means no dementia [concept: scores/cdr_scale].\n"
    "3. Only use concepts from the evidence list. Never cite a concept id that "
    "is not listed, and never make a statement the listed concepts do not "
    "support. Omit anything that is not supported.\n"
    "4. The report formatting guidelines are style guidance only: never cite "
    "them inline and never list them in the Sources section.\n"
    "5. Never give medication dosages or any drug regimen. You may mention that "
    "a treatment class exists and that a clinician decides on prescribing.\n"
    "6. Do not invent numbers, scores or probabilities; only repeat values from "
    "the prediction payload.\n"
    "7. Finish the report with a '## Sources' section listing the concept ids "
    "you actually cited together with their titles, then a '## Disclaimer' "
    "section stating that this is decision support and not a clinical diagnosis."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ---- tokenisation ---------------------------------------------------------
# Ranking tokens (intent matching) keep every word: "Very Mild Impairment" must
# beat "Mild Impairment" when the model predicted the former.
_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")
# Coverage tokens (citation guard) drop function words: they carry no clinical
# meaning, so they must not be able to satisfy the guard on their own.
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can cannot could did do does
    doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more
    most my myself no nor not of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would
    you your yours yourself yourselves
    """.split()
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _content_words(text: str) -> list[str]:
    return [w for w in _tokens(text) if len(w) > 2 and w not in _STOPWORDS]


def _coverage(text: str, body: str) -> float:
    """Share of the content words of ``text`` that occur in ``body``.

    A sentence without content words carries no clinical claim and is treated as
    trivially supported.
    """
    words = _content_words(text)
    if not words:
        return 1.0
    body_words = set(_content_words(_clean_markup(body)))
    return sum(1 for w in words if w in body_words) / len(words)


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---- concept bodies -------------------------------------------------------
# Where the report generator reads the body a citation has to be backed by. The
# OKF bundle is the source of truth; the retrieved chunk text is the fallback for
# concepts that are not in the bundle.
_MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\([^)\s]*\)")
_MD_MARK_RE = re.compile(r"(\*\*|__|~~|\*|`)")
_MOJIBAKE_RE = re.compile("\ufffd+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_NAV_LINE_RE = re.compile(
    r"^\s*(?:related|see also|see related|also see|further reading)\b", re.IGNORECASE
)


def _clean_markup(text: str) -> str:
    """Markdown links / emphasis / mojibake dashes -> plain readable text."""
    cleaned = _MD_LINK_RE.sub(r"\1", str(text))
    cleaned = _MOJIBAKE_RE.sub("\u2014", cleaned)
    cleaned = _MD_MARK_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_leading_title(text: str, title: str) -> str:
    """Drop the chunk title line that ``chunk_concept`` prepends to the text."""
    lines = str(text or "").splitlines()
    if title and lines and lines[0].strip() == title.strip():
        return "\n".join(lines[1:]).strip()
    return "\n".join(lines).strip()


def concept_bodies(
    cfg: RagConfig | None, hits: Iterable[Hit]
) -> dict[str, str]:
    """``concept_id -> body text`` for every retrieved concept.

    The OKF bundle is used when a config is available so the guard sees the whole
    concept; otherwise (and for concepts outside the bundle) the retrieved chunk
    text is used, minus the title line ``chunk_concept`` prepends.
    """
    concepts: Mapping[str, Any] = {}
    if cfg is not None:
        try:
            from .okf_loader import load_bundle

            concepts = load_bundle(cfg.knowledge_bundle_dir).concepts
        except Exception:  # pragma: no cover - unreadable bundle
            logger.warning(
                "report: knowledge bundle unavailable, falling back to chunk text",
                exc_info=True,
            )
            concepts = {}
    bodies: dict[str, str] = {}
    for hit in hits:
        concept_id = hit.concept_id or hit.source
        if not concept_id or concept_id in bodies:
            continue
        concept = concepts.get(concept_id)
        body = concept.body.strip() if concept is not None else ""
        bodies[concept_id] = body or _strip_leading_title(hit.text, hit.title)
    return bodies


def _body_segments(body: str) -> list[str]:
    """Prose / bullet blocks of a concept body, navigation lines removed."""
    segments: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            text = _clean_markup(" ".join(current))
            if text:
                segments.append(text)
            current.clear()

    for raw in str(body or "").splitlines():
        line = raw.strip()
        if not line or _HEADING_RE.match(line) or _NAV_LINE_RE.match(line):
            flush()
            continue
        if _LIST_MARKER_RE.match(raw):
            flush()
        current.append(_LIST_MARKER_RE.sub("", line, count=1))
    flush()
    return segments


# ---- bare checklists ------------------------------------------------------
# Some concepts are nothing but a checklist of criteria or measures
# (``diagnosis/red_flags_referral``: five referral criteria; the lifestyle and
# non-pharmacological management concepts: the measures they list). Quoting one
# item of such a list on its own turns a criterion into a claim about the patient
# ("Rapidly progressive symptoms (weeks to months)." reads as a finding), which
# the source never says. Those bodies are therefore quoted whole, behind a
# lead-in built from the concept's own title.
#
# The body has to be a *pure* checklist for this to apply: a single prose
# paragraph or a lead-in line ending in a colon means the list is introduced and
# the individual items are already statements in their own right. Items that
# open with a labelled definition ("CDR 2 - Moderate dementia") are definitions
# rather than criteria and stay quotable one by one.
CRITERIA_LIST_MIN_ITEMS = 3
_LABELLED_ITEM_RE = re.compile(r"^\**[^*\n]{1,60}?\**\s*[:\u2013\u2014-]\s*\S")


def _as_clause(text: str) -> str:
    """One checklist item as a clause: markup stripped, final period removed."""
    return _clean_markup(text).strip().rstrip(".").strip()


def _checklist_items(body: str) -> list[str]:
    """Items of a pure, unlabelled checklist body, or ``[]`` for any other body.

    Items are returned as clause fragments with the source's final period
    stripped, so a caller can join them with ``"; "`` and terminate the list
    once instead of emitting ``"weeks to months).; Focal neurological signs."``.
    """
    lines = [
        raw
        for raw in str(body or "").splitlines()
        if raw.strip() and not _HEADING_RE.match(raw) and not _NAV_LINE_RE.match(raw)
    ]
    if not lines or any(not _LIST_MARKER_RE.match(raw) for raw in lines):
        return []
    items: list[str] = []
    current: list[str] = []
    for raw in lines:
        if _LIST_MARKER_RE.match(raw):
            if current:
                text = _as_clause(" ".join(current))
                if text:
                    items.append(text)
                current = []
            current.append(_LIST_MARKER_RE.sub("", raw.strip(), count=1))
        else:  # a wrapped continuation line belongs to the item above it
            current.append(raw.strip())
    if current:
        text = _as_clause(" ".join(current))
        if text:
            items.append(text)
    if len(items) < CRITERIA_LIST_MIN_ITEMS:
        return []
    if any(_LABELLED_ITEM_RE.match(item) for item in items):
        return []
    return items


def _lead_in_from_title(title: str) -> str:
    """``"Red Flags Requiring Specialist Referral"`` -> ``"Red flags ... include:"``."""
    text = _clean_markup(title).strip().rstrip(":").rstrip(".")
    if not text:
        return ""
    return f"{text[:1].upper()}{text[1:].lower()} include:"


def _checklist_statement(body: str, title: str) -> str:
    """The whole checklist of ``body`` behind a lead-in, or '' if not one.

    A body that is a pure checklist carries no lead-in of its own, so there is
    nothing to quote but the whole list. Returns '' when the body is not such a
    list, when its title yields no lead-in, or when the assembled statement is
    not covered by the body it claims to come from.
    """
    checklist = _checklist_items(body)
    if not checklist:
        return ""
    lead_in = _lead_in_from_title(title)
    if not lead_in:
        logger.warning(
            "report: %r is a bare checklist with no usable title, dropping it", title
        )
        return ""
    text = f"{lead_in} {'; '.join(checklist)}."
    if _coverage(text, body) < CITATION_SUPPORT_MIN_COVERAGE:
        logger.warning(
            "report: checklist of %r is not covered by its own body, dropping it",
            title,
        )
        return ""
    return text


def _is_bare_checklist_item(sentence: str, body: str, title: str) -> bool:
    """Does ``sentence`` quote one item of a bare checklist without its lead-in?

    True for "Rapidly progressive symptoms (weeks to months)." cited to
    ``diagnosis/red_flags_referral``: the 40% coverage guard is happy (every
    content word is in the body) but the sentence reads as a finding about the
    patient, which the source never claims. False once the lead-in is present.
    """
    if not body or not _checklist_items(body):
        return False
    # the citation is not evidence that the lead-in is present: "red" and
    # "referral" come out of the concept id `diagnosis/red_flags_referral`
    # itself, which would otherwise make every criterion look already-led
    words = set(_content_words(CITATION_RE.sub(" ", sentence)))
    lead_in = _lead_in_from_title(title)
    lead_words = _content_words(lead_in)
    if lead_words and all(word in words for word in lead_words[:1]):
        return False
    if not words:
        return False
    for item in _checklist_items(body):
        item_words = set(_content_words(item))
        if item_words and item_words <= words:
            return True
    return False


# ---- statements taken from a concept body --------------------------------
_RANGE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|to)\s*(\d+(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)
_RANGE_FRAGMENT_RE = re.compile(r"^\d+(?:\.\d+)?\s*(?:-|\u2013|\u2014|to)\s*\d")


def _value_match(segment: str, value: float, label: str) -> bool:
    """Does this segment speak about ``label = value`` (directly or by range)?"""
    low = segment.lower()
    if label:
        literal = f"{label.lower()} {value:g}".lower()
        if re.search(rf"(?<![a-z0-9]){re.escape(literal)}(?![0-9])", low):
            return True
    for low_bound, high_bound in _RANGE_RE.findall(segment):
        try:
            if float(low_bound) <= value <= float(high_bound):
                return True
        except ValueError:  # pragma: no cover - regex guarantees numbers
            continue
    return False


def _trim_to_sentences(text: str, max_sentences: int) -> str:
    if max_sentences < 1:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return " ".join(parts[:max_sentences]).strip()


def _select_statement(
    body: str,
    *,
    intent: Iterable[str] = (),
    label_match: str = "",
    value: float | None = None,
    label: str = "",
    max_sentences: int = 1,
    require_label: bool = False,
    title: str = "",
) -> str:
    """Pick the best-supported sentence(s) of ``body``, or '' if none matches.

    ``intent`` phrases score the segments (all words of a phrase must occur in
    the segment); ``label_match`` prefers a segment that starts with that label
    (so "Very Mild Impairment" wins over "Mild Impairment"); ``value``/``label``
    prefer a segment that names the actual score value or a range containing it.
    With ``require_label`` only segments carrying the label are eligible, so the
    caller never falls back to a segment about a different stage.
    The winner is truncated to ``max_sentences`` sentences and must still pass
    the citation coverage guard.

    ``title`` is the concept's own title. It is used only when the winner is an
    item of a bare checklist, which is quoted in full behind a lead-in derived
    from that title (see :func:`_checklist_items`).
    """
    checklist = _checklist_items(body)
    if checklist:
        return _checklist_statement(body, title)

    segments = _body_segments(body)
    if not segments:
        return ""
    ranked: list[tuple[float, int, str]] = []
    for index, segment in enumerate(segments):
        score = 0.0
        tokens = set(_tokens(segment))
        matched_label = bool(
            label_match and segment.lower().startswith(label_match.lower())
        )
        if require_label and label_match and not matched_label:
            continue
        if matched_label:
            score += 1000.0
        if value is not None and _value_match(segment, value, label):
            score += 500.0
        for phrase in intent:
            phrase_tokens = [t for t in _tokens(phrase) if len(t) > 2]
            if phrase_tokens and all(t in tokens for t in phrase_tokens):
                score += float(len(phrase_tokens))
        ranked.append((-score, index, segment))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (item[0], item[1]))
    best_score = -ranked[0][0]
    if best_score <= 0.0:
        return ""
    index = ranked[0][1]
    segment = ranked[0][2]

    # a matched range reads better with the list lead-in it belongs to
    if _RANGE_FRAGMENT_RE.match(segment):
        for back in range(index - 1, max(-1, index - 4), -1):
            if segments[back].endswith(":"):
                segment = f"{segments[back]} {segment}"
                break

    text = _trim_to_sentences(_clean_markup(segment), max_sentences)
    if not text:
        return ""
    if _coverage(text, body) < CITATION_SUPPORT_MIN_COVERAGE:  # pragma: no cover
        logger.warning(
            "report: selected statement is not covered by its own body, dropping it"
        )
        return ""
    return text


# ---- generation guidance vs clinical evidence ----------------------------
def _is_generation_guidance(concept_id: str, concept_type: str = "") -> bool:
    """Formatting guides: usable as generation guidance, never as evidence."""
    normalized = str(concept_id or "").strip().lower()
    if normalized == "reporting" or normalized.startswith("reporting/"):
        return True
    kind = str(concept_type or "").strip().lower()
    return "guideline" in kind or "style guide" in kind


def _is_guidance_hit(hit: Hit) -> bool:
    return _is_generation_guidance(hit.concept_id or hit.source, hit.concept_type)


def _citable(hits: Iterable[Hit]) -> list[Hit]:
    """The hits that may be used as clinical evidence (guidance removed)."""
    return [h for h in hits if not _is_guidance_hit(h)]


# ---- prompt blocks --------------------------------------------------------
def _prediction_block(pred: dict[str, Any]) -> str:
    return json.dumps(pred, indent=2, ensure_ascii=False)


def _retrieval_block(hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        concept_id = h.concept_id or h.source
        tag = "" if h.direct else f" | via link from {h.link_source}"
        lines.append(
            f"[{i}] concept_id: {concept_id} | Title: {h.title} | "
            f"score={h.score:.3f}{tag}\n{h.text}"
        )
    return "\n\n".join(lines) if lines else "(no knowledge concepts retrieved)"


def _evidence_block(hits: list[Hit]) -> str:
    return _retrieval_block(_citable(hits))


def _guidance_block(hits: list[Hit]) -> str:
    """Formatting guidance, clearly separated from the clinical evidence."""
    guidance = [h for h in hits if _is_guidance_hit(h)]
    if not guidance:
        return ""
    ids = ", ".join(sorted({h.concept_id or h.source for h in guidance}))
    return (
        f"Report formatting guidance (structure and style only, never cite "
        f"{ids}, never list it in Sources):\n{_retrieval_block(guidance)}"
    )


def _probability_lines(probs: dict[str, Any]) -> str:
    lines = []
    for name, value in probs.items():
        number = _as_float(value)
        if number is None:
            continue
        if 1.0 < number <= 100.0:
            number /= 100.0
        if 0.0 <= number <= 1.0:
            lines.append(f"- **{name}:** {number * 100:.1f}%")
    return "\n".join(lines) or "- *no valid probability breakdown available*"


# ---- which concept supports a statement -----------------------------------
def _pick_concept(
    hits: list[Hit], preferred: str, keywords: Iterable[str] = ()
) -> str:
    """Choose the retrieved concept that best supports a statement.

    Uses the preferred concept when it was retrieved, otherwise the retrieved
    concept with the strongest keyword overlap. Returns '' when no retrieved
    concept matches, so a statement is never cited to unrelated knowledge.
    Generation-guidance concepts are never eligible.
    """
    citable_hits = [h for h in hits if not _is_guidance_hit(h)]
    for hit in citable_hits:
        concept_id = hit.concept_id or hit.source
        if preferred and concept_id == preferred:
            return concept_id
    best_id = ""
    best_score = 0.0
    keywords = [k.lower() for k in keywords]
    for hit in citable_hits:
        concept_id = hit.concept_id or hit.source
        blob = " ".join(
            [concept_id.replace("/", " "), hit.title.lower(), hit.concept_type.lower()]
        )
        score = 0.0
        for keyword in keywords:
            if keyword in blob:
                score += 1.0
        if score > best_score:
            best_score = score
            best_id = concept_id
    return best_id


def _concept_titles(hits: Iterable[Hit]) -> dict[str, str]:
    """``concept_id -> own title`` for the retrieved concepts."""
    titles: dict[str, str] = {}
    for hit in hits:
        concept_id = hit.concept_id or hit.source
        if concept_id and concept_id not in titles and hit.title:
            titles[concept_id] = hit.title
    return titles


def _citation(concept_id: str, statement: str, bodies: Mapping[str, str]) -> str:
    """Inline citation for ``statement``, or '' when the body does not support it."""
    if not concept_id or _is_generation_guidance(concept_id):
        return ""
    if _coverage(statement, bodies.get(concept_id, "")) < CITATION_SUPPORT_MIN_COVERAGE:
        logger.warning(
            "report: dropping statement and its citation to %r - the concept body "
            "does not support it: %r",
            concept_id,
            statement[:80],
        )
        return ""
    return f" [concept: {concept_id}]"


def _cited_line(
    prefix: str, statement: str, concept_id: str, bodies: Mapping[str, str]
) -> str | None:
    """A bullet carrying a verified citation, or None when it must be dropped."""
    if not statement:
        return None
    citation = _citation(concept_id, statement, bodies)
    if not citation:
        return None
    return f"- {prefix}{statement}{citation}"


# ---- template report ------------------------------------------------------
_SUMMARY_INTENT = (
    "cognitive decline",
    "questionable",
    "dementia",
    "impairment",
    "staging",
)
_RISK_INTENT = (
    "decision-support",
    "definitive diagnosis",
    "clinical context",
    "qualified clinician",
    "image quality",
    "risk factor",
    "delay onset",
)

_SUMMARY_SPEC = (
    "overview/clinical_stages_cdr",
    ("clinical stage", "severity", "staging", "cdr", "impairment"),
)
_RISK_SPEC = (
    "diagnosis/ai_assisted_prediction_caveats",
    ("caveat", "screening", "decision support", "not a diagnosis"),
)

# Clinical score scales the knowledge base describes, with the phrases that mark
# the relevant segment of each body.
_SCORE_SPECS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "cdr": (
        "scores/cdr_scale",
        ("cdr", "clinical dementia rating", "scale"),
        ("dementia", "staging", "scale", "domains"),
    ),
    "mmse": (
        "scores/mmse",
        ("mmse", "mini-mental", "cognitive test"),
        ("interpreted", "impairment", "normal", "screening", "maximum"),
    ),
    "nwbv": (
        "scores/nwbv",
        ("nwbv", "whole brain", "volume", "atrophy"),
        ("atrophy", "normalized", "intracranial", "fraction", "lower"),
    ),
}

# (preferred concept, intent phrases) per predicted stage. The intent phrases
# locate the supporting sentence inside the concept body; the wording itself
# always comes from that body.
_STEP_SPECS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "no impairment": (
        (
            "risk/modifiable_risk_reduction",
            ("blood pressure", "diabetes", "physical activity", "delay onset"),
        ),
        (
            "treatment/lifestyle_risk_modification",
            ("blood pressure", "diabetes", "diet", "smoking", "sleep"),
        ),
        (
            "risk/risk_factors",
            ("risk factor", "age", "cardiovascular", "modifiable"),
        ),
        (
            "overview/disease_progression",
            ("continuum", "progression", "preclinical"),
        ),
    ),
    "moderate": (
        (
            "diagnosis/diagnostic_workup",
            ("cognitive screening", "functional assessment", "neuroimaging"),
        ),
        (
            "treatment/non_pharmacological",
            (
                "cognitive stimulation",
                "physical exercise",
                "social engagement",
                "caregiver",
                "non-drug",
            ),
        ),
        (
            "treatment/nmda_antagonist",
            ("memantine", "cholinesterase", "moderate", "functional decline"),
        ),
        (
            "diagnosis/red_flags_referral",
            ("rapidly progressive", "focal neurological", "seizures", "referral"),
        ),
    ),
    "mild": (
        (
            "diagnosis/diagnostic_workup",
            ("cognitive screening", "functional assessment", "neuroimaging"),
        ),
        (
            "treatment/early_stage_guidance",
            ("next steps", "assessment", "monitoring", "intervention"),
        ),
        (
            "risk/modifiable_risk_reduction",
            ("blood pressure", "diabetes", "physical activity", "delay onset"),
        ),
    ),
}
_DEFAULT_STEP_SPECS = _STEP_SPECS["mild"]


def _step_specs(stage: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(preferred concept, intent phrases) triples for the recommendations."""
    normalized = str(stage).lower()
    if "no impairment" in normalized:
        return _STEP_SPECS["no impairment"]
    if "moderate" in normalized:
        return _STEP_SPECS["moderate"]
    if "mild" in normalized:
        return _STEP_SPECS["mild"]
    return _DEFAULT_STEP_SPECS


def _score_statement(
    bodies: Mapping[str, str], hits: list[Hit], key: str, value: Any
) -> tuple[str, str] | None:
    """(statement, concept id) for one clinical score, or None."""
    spec = _SCORE_SPECS.get(str(key).strip().lower())
    if spec is None:
        return None
    preferred, keywords, intent = spec
    concept_id = _pick_concept(hits, preferred, keywords)
    if not concept_id:
        logger.info("report: no retrieved concept describes the %s scale", key)
        return None
    statement = _select_statement(
        bodies.get(concept_id, ""),
        intent=intent,
        value=_as_float(value),
        label=str(key).strip().upper(),
        max_sentences=2,
        title=_concept_titles(hits).get(concept_id, ""),
    )
    if not statement:
        logger.info(
            "report: %s body has no sentence covering %s = %s", concept_id, key, value
        )
        return None
    return statement, concept_id


def _summary_statement(
    bodies: Mapping[str, str], hits: list[Hit], stage: str
) -> tuple[str, str] | None:
    preferred, keywords = _SUMMARY_SPEC
    concept_id = _pick_concept(hits, preferred, keywords)
    if not concept_id:
        return None
    # the summary has to describe the stage that was predicted: a segment about
    # a different stage is dropped rather than paraphrased into the report
    statement = _select_statement(
        bodies.get(concept_id, ""),
        intent=_SUMMARY_INTENT,
        label_match=stage,
        require_label=True,
        title=_concept_titles(hits).get(concept_id, ""),
    )
    return (statement, concept_id) if statement else None


def _risk_statement(
    bodies: Mapping[str, str], hits: list[Hit]
) -> tuple[str, str] | None:
    preferred, keywords = _RISK_SPEC
    concept_id = _pick_concept(hits, preferred, keywords)
    if not concept_id:
        return None
    statement = _select_statement(
        bodies.get(concept_id, ""),
        intent=_RISK_INTENT,
        max_sentences=2,
        title=_concept_titles(hits).get(concept_id, ""),
    )
    return (statement, concept_id) if statement else None


def _step_statement(
    bodies: Mapping[str, str], hits: list[Hit], stage: str
) -> list[str]:
    """Recommendation bullets, each one a sentence of the concept it cites."""
    lines: list[str] = []
    seen: set[str] = set()
    titles = _concept_titles(hits)
    for preferred, intent in _step_specs(stage):
        # no keyword fallback here: a next step must come from the concept that
        # actually prescribes it, not from whatever happened to be retrieved
        concept_id = _pick_concept(hits, preferred, ())
        if not concept_id:
            continue
        statement = _select_statement(
            bodies.get(concept_id, ""),
            intent=intent,
            max_sentences=2,
            title=titles.get(concept_id, ""),
        )
        if not statement or statement in seen:
            continue
        line = _cited_line("", statement, concept_id, bodies)
        if not line:
            continue
        seen.add(statement)
        lines.append(line)
    return lines


# ---- severity bands -------------------------------------------------------
# The scores and the imaging stage are three different instruments, so they are
# mapped onto one ordinal ladder before being compared. The bands are the ones
# the knowledge base itself states: ``scores/mmse`` (26-30 normal, 21-25 mild
# cognitive impairment, 10-20 moderate impairment, 0-9 severe impairment) and
# ``scores/cdr_scale`` (0 none, 0.5 very mild, 1 mild, 2 moderate, 3 severe),
# with "moderate impairment" on the MMSE taken to mean the same rung as
# "Moderate dementia" on the CDR.
SEVERITY_LADDER = ("none", "very mild / MCI", "mild dementia", "moderate dementia", "severe dementia")
# Both spellings are accepted for each rung: the CDR label the knowledge base
# uses ("moderate dementia") and the MRI class name the models actually predict
# (splits.CLASSES: "Moderate Impairment"). Only the latter reaches here from the
# app, and a rung missing from this table silently disables the A3 disagreement
# note for that stage.
_STAGE_TIERS = {
    "no impairment": 0,
    "non-dementia": 0,
    "non dementia": 0,
    "normal": 0,
    "very mild impairment": 1,
    "very mild dementia": 1,
    "mild impairment": 2,
    "mild dementia": 2,
    "mci": 2,
    "moderate impairment": 3,
    "moderate dementia": 3,
    "severe impairment": 4,
    "severe dementia": 4,
}
# Two rungs apart is wider than the spread a single instrument can explain on its
# own, so a one-rung difference is not reported as a disagreement.
SEVERITY_DISAGREEMENT_TIERS = 2
# Structural guidance, not a clinical claim about this patient: hardcoded and
# deliberately uncited.
_SEVERITY_NOTE = (
    "Individual scores and the imaging stage can point to different severity "
    "bands; interpret them together with clinical judgement."
)


def _mmse_tier(value: float | None) -> int | None:
    if value is None or value < 0:
        return None
    if value >= 26:
        return 0
    if value >= 21:
        return 1
    if value >= 10:
        return 3
    return 4


def _cdr_tier(value: float | None) -> int | None:
    if value is None or value < 0:
        return None
    if value <= 0:
        return 0
    if value < 0.75:
        return 1
    if value < 1.5:
        return 2
    if value < 2.5:
        return 3
    return 4


def _stage_tier(stage: str) -> int | None:
    return _STAGE_TIERS.get(str(stage).strip().lower())


def _severity_note(clinical_inputs: Any, stage: str) -> str:
    """Neutral note when the scores and the imaging stage disagree, else ''."""
    tiers: list[int] = []
    # the caller supplies the score names as they appear in the UI ("CDR",
    # "MMSE"), so they are matched case-insensitively: looking them up verbatim
    # finds nothing and the note silently never fires.
    supplied = (
        {str(key).strip().lower(): value for key, value in clinical_inputs.items()}
        if isinstance(clinical_inputs, Mapping)
        else {}
    )
    for key, band in (("mmse", _mmse_tier), ("cdr", _cdr_tier)):
        tier = band(_as_float(supplied.get(key)))
        if tier is not None:
            tiers.append(tier)
    stage_tier = _stage_tier(stage)
    if stage_tier is not None:
        tiers.append(stage_tier)
    if len(tiers) < 2:
        return ""
    if max(tiers) - min(tiers) >= SEVERITY_DISAGREEMENT_TIERS:
        return _SEVERITY_NOTE
    return ""


def _template_report(
    pred: dict[str, Any],
    query: str,
    hits: list[Hit],
    bodies: Mapping[str, str] | None = None,
) -> str:
    """Deterministic markdown report used when no LLM is available.

    Every clinical statement is selected from the body of the concept it cites;
    sections without a supported statement are omitted entirely.
    """
    if bodies is None:
        bodies = concept_bodies(None, hits)
    stage = str(pred.get("predicted_stage", pred.get("stage", "Unknown")))
    probs = pred.get("probabilities") or pred.get("probs") or {}
    if not isinstance(probs, dict):
        probs = {}

    lines: list[str] = ["# Automated Clinical Decision-Support Report", ""]
    lines.append(f"**Predicted cognitive stage: {stage}**")
    lines.append(f"**Query:** {query}")

    summary_line = None
    summary = _summary_statement(bodies, hits, stage)
    if summary:
        summary_line = _cited_line("", summary[0], summary[1], bodies)
    else:
        logger.info("report: no retrieved concept supports a clinical summary")
    lines += ["", "## Clinical Summary", ""]
    if summary_line:
        lines.append(summary_line)
    lines.append("Interpret the predicted stage above in the full clinical context.")

    # model output: structural, never cited
    lines += ["", "## Imaging (MRI) Analysis", "", _probability_lines(probs)]

    clinical_inputs = pred.get("clinical_inputs")
    lines += ["", "## Clinical Scores Interpretation", ""]
    score_lines: list[str] = []
    if isinstance(clinical_inputs, Mapping) and clinical_inputs:
        lines += ["| Score | Value |", "| --- | --- |"]
        for key, value in clinical_inputs.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        for key, value in clinical_inputs.items():
            found = _score_statement(bodies, hits, key, value)
            if not found:
                continue
            statement, concept_id = found
            line = _cited_line(f"**{key} = {value}** \u2014 ", statement, concept_id, bodies)
            if line:
                score_lines.append(line)
    if not clinical_inputs:
        lines.append("*(no clinical scores were provided)*")
    elif not score_lines:
        logger.info("report: no retrieved concept interprets the clinical scores")
    lines += score_lines
    severity_note = _severity_note(clinical_inputs, stage)
    if severity_note:
        lines += ["", severity_note]

    risk = _risk_statement(bodies, hits)
    risk_line = _cited_line("", risk[0], risk[1], bodies) if risk else None
    if risk_line:
        lines += ["", "## Risk Interpretation", "", risk_line]
    else:
        logger.info("report: no retrieved concept supports a risk interpretation")

    steps = _step_statement(bodies, hits, stage)
    if steps:
        lines += ["", "## Recommended Next Steps", ""] + steps
    else:
        logger.info("report: no retrieved concept supports a next step")

    lines += _sources_section(_cited_concepts("\n".join(lines), list(hits)))
    lines += _disclaimer_section()
    return "\n".join(lines)


def generate_template_report(
    pred: dict[str, Any],
    query: str,
    hits: list[Hit],
    bodies: Mapping[str, str] | None = None,
) -> str:
    """Public wrapper around the deterministic (offline) report generator.

    ``bodies`` should come from :func:`concept_bodies` so the guard sees the full
    concept text; without it the retrieved chunk text is used.
    """
    return _template_report(pred, query, hits, bodies)


# ---- LLM post-processing --------------------------------------------------
def _drop_dosage_sentences(text: str) -> str:
    """Remove statements that prescribe a medication dosage."""

    out: list[str] = []
    for line in text.splitlines():
        if not _DOSAGE_RE.search(line):
            out.append(line)
            continue
        logger.warning("report: removing a statement containing a dosage")
        kept = [
            part
            for part in _SENTENCE_SPLIT_RE.split(line)
            if part.strip() and not _DOSAGE_RE.search(part)
        ]
        if not kept:
            continue
        remainder = " ".join(part.strip() for part in kept)
        indent = line[: len(line) - len(line.lstrip())]
        if line.lstrip().startswith(("#", "-", "*", ">", "|")):
            out.append(indent + remainder)
        else:
            out.append(remainder)
    return "\n".join(out)


def _strip_sources_and_disclaimer(text: str) -> str:
    """Drop any LLM-authored Sources / Disclaimer tail (rebuilt afterwards)."""
    kept: list[str] = []
    for line in text.splitlines():
        heading = line.strip().lstrip("#").strip().rstrip(":").strip().lower()
        if heading in {"sources", "disclaimer", "references"}:
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def _drop_unknown_citations(text: str, hits: list[Hit]) -> str:
    """Remove citations pointing at concepts that were not retrieved, and any
    citation to a generation-guidance concept."""

    def replace(match: re.Match[str]) -> str:
        concept_id = match.group(1).strip()
        known = any((h.concept_id or h.source) == concept_id for h in hits)
        if known and not _is_generation_guidance(concept_id):
            return f"[concept: {concept_id}]"
        logger.warning(
            "report: dropping citation to unretrieved or non-evidence concept %r",
            concept_id,
        )
        return ""

    return CITATION_RE.sub(replace, text)


def _tidy(sentence: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", sentence.strip())
    return re.sub(r"\s+([,;:.!?])", r"\1", text).strip()


def _verify_sentence(sentence: str, bodies: Mapping[str, str]) -> tuple[str, bool]:
    """Drop citations the cited concept body does not support.

    Returns ``(sentence, supported)``: ``supported`` is False when at least one
    citation had to be removed because the concept body does not contain enough
    of the sentence's content words.
    """
    words = _content_words(CITATION_RE.sub(" ", sentence))
    body_words = {cid: set(_content_words(_clean_markup(body))) for cid, body in bodies.items()}
    supported = True

    def replace(match: re.Match[str]) -> str:
        nonlocal supported
        concept_id = match.group(1).strip()
        concept_words = body_words.get(concept_id)
        if concept_words is not None:
            share = (
                sum(1 for w in words if w in concept_words) / len(words)
                if words
                else 1.0
            )
            if share >= CITATION_SUPPORT_MIN_COVERAGE:
                return f"[concept: {concept_id}]"
        supported = False
        logger.warning(
            "report: dropping citation to %r - only %s of the sentence's content "
            "words occur in that concept body",
            concept_id,
            "none" if concept_words is None else f"{share:.0%}",
        )
        return ""

    return _tidy(CITATION_RE.sub(replace, sentence)), supported


def _expand_bare_checklist_items(
    sentence: str,
    bodies: Mapping[str, str],
    titles: Mapping[str, str],
    *,
    citations: Iterable[str] | None = None,
) -> str:
    """Rewrite a quoted checklist criterion as the full list behind its lead-in.

    The LLM is free to lift one criterion out of a bare checklist
    (``diagnosis/red_flags_referral``) and, because every content word of that
    criterion is in the body, the coverage guard happily keeps it. Left alone it
    reads as a finding about the patient. The sentence is therefore replaced by
    the concept's own lead-in followed by every item, which is what the concept
    actually says. A criterion that already sits behind its lead-in, or one whose
    concept is not a bare checklist, is left untouched.

    ``citations`` overrides the citations looked for in ``sentence``. The citation
    often ends up in a later sentence fragment ("... (weeks to months). is
    present in this patient [concept: ...]"), so the caller passes the citations
    of the whole line it is splitting.
    """
    for concept_id in citations if citations is not None else CITATION_RE.findall(sentence):
        concept_id = concept_id.strip()
        replacement = _checklist_statement(
            bodies.get(concept_id, ""), titles.get(concept_id, "")
        )
        if replacement and _is_bare_checklist_item(
            sentence, bodies.get(concept_id, ""), titles.get(concept_id, "")
        ):
            logger.warning(
                "report: expanding a bare checklist item of %r into its full list",
                concept_id,
            )
            return f"{replacement} [concept: {concept_id}]"
    return sentence


def _enforce_citation_support(
    text: str,
    bodies: Mapping[str, str],
    *,
    drop_unsupported_sentences: bool,
    titles: Mapping[str, str] | None = None,
) -> str:
    """Apply the 40% content-word guard sentence by sentence.

    ``drop_unsupported_sentences`` is True for the template path, where an
    unsupported sentence must disappear; the LLM path only loses the citation so
    the surrounding paragraph structure is preserved. Both paths additionally run
    the bare-checklist rule, so a quoted criterion always reaches the reader
    behind the lead-in built from its concept's title.
    """
    titles = titles or {}
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # structural lines cannot be split into sentences without breaking them
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("```")
            or stripped.startswith(">")
        ):
            out.append(line)
            continue
        bullet = ""
        marker = re.match(r"^(\s*)(?:[-*+]|\d+[.)])\s+", line)
        if marker:
            bullet = marker.group(0)
            line = line[len(bullet) :]
        kept: list[str] = []
        line_citations = CITATION_RE.findall(line)
        # the checklist rule runs on the whole line, before the coverage guard and
        # before the line is split into sentences: the citation usually ends up in
        # a different sentence fragment than the criterion it supports, and the
        # claim the model built on that criterion ("... is present in this
        # patient") is not supported either once the criterion is replaced
        expanded_line = _expand_bare_checklist_items(
            line, bodies, titles, citations=line_citations
        )
        if expanded_line != line:
            replaced, _ = _verify_sentence(expanded_line, bodies)
            out.append(bullet + (replaced or expanded_line))
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            if not sentence.strip():
                continue
            cleaned, supported = _verify_sentence(sentence, bodies)
            if not supported and drop_unsupported_sentences:
                continue
            if cleaned:
                kept.append(cleaned)
        if kept:
            out.append(bullet + " ".join(kept))
        elif not drop_unsupported_sentences:
            out.append(bullet.rstrip())
    return "\n".join(out)


def _cited_concepts(text: str, hits: list[Hit]) -> list[Hit]:
    """Retrieved concepts that survive as inline citations, in citation order.

    Never falls back to "all retrieved hits": Sources lists exactly the concepts
    the report cites. Concepts that were retrieved but not cited are surfaced by
    the app instead.
    """
    by_id: dict[str, Hit] = {}
    for hit in hits:
        by_id.setdefault(hit.concept_id or hit.source, hit)
    order: list[str] = []
    seen: set[str] = set()
    for raw in CITATION_RE.findall(text):
        concept_id = raw.strip()
        if not concept_id or concept_id in seen or concept_id not in by_id:
            continue
        if _is_generation_guidance(concept_id, by_id[concept_id].concept_type):
            continue
        seen.add(concept_id)
        order.append(concept_id)
    return [by_id[cid] for cid in order]


def cited_concept_ids(report: str) -> list[str]:
    """Concept ids cited inline in ``report``, in citation order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in CITATION_RE.findall(report):
        concept_id = raw.strip()
        if not concept_id or concept_id in seen or _is_generation_guidance(concept_id):
            continue
        seen.add(concept_id)
        out.append(concept_id)
    return out


def _concept_lines(hits: Iterable[Hit]) -> list[str]:
    """One line per cited concept, in citation order."""
    lines: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        concept_id = hit.concept_id or hit.source
        if not concept_id or concept_id in seen:
            continue
        seen.add(concept_id)
        if _is_generation_guidance(concept_id, hit.concept_type):
            continue
        kind = "direct" if hit.direct else f"linked from {hit.link_source}"
        # never reference the guidance concept in the sources line
        if hit.link_source and _is_generation_guidance(hit.link_source):
            kind = "linked"
        lines.append(
            f"- `{concept_id}` \u2014 {hit.title} "
            f"({hit.concept_type or 'Concept'}, {kind})"
        )
    return lines


def _sources_section(hits: list[Hit]) -> list[str]:
    lines = ["", _SOURCES_HEADING, ""]
    concept_lines = _concept_lines(hits)
    if concept_lines:
        lines.extend(concept_lines)
    else:
        lines.append("- *(no knowledge concepts cited)*")
    return lines


def _disclaimer_section() -> list[str]:
    return ["", "## Disclaimer", "", _DISCLAIMER]


def generate_report(
    cfg: RagConfig,
    prediction: dict[str, Any],
    query: str,
    hits: list[Hit],
    *,
    allow_llm: bool = True,
) -> tuple[str, bool]:
    """Return (report_markdown, used_llm).

    Uses the OpenAI-compatible chat endpoint when a key is configured and the
    call succeeds; otherwise returns a template-based report. ``allow_llm=False``
    forces the template path without attempting a call, which is what a
    per-session LLM budget needs. Either way the result is post-processed:
    unknown and generation-guidance citations are dropped, every remaining
    citation is checked against the concept body, the Sources section is rebuilt
    from the cited concepts only and the disclaimer is enforced.
    """
    bodies = concept_bodies(cfg, hits)
    if cfg.has_api_key and allow_llm:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            guidance = _guidance_block(list(hits))
            user_msg = (
                "Patient query:\n"
                f"{query}\n\n"
                "Automated prediction:\n"
                f"{_prediction_block(prediction)}\n\n"
                "Retrieved clinical evidence (these are the ONLY concepts you may "
                "cite or rely on):\n"
                f"{_evidence_block(list(hits))}"
            )
            if guidance:
                user_msg += "\n\n" + guidance
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                body = _strip_sources_and_disclaimer(content.strip())
                body = _drop_unknown_citations(body, hits)
                body = _drop_dosage_sentences(body)
                body = _enforce_citation_support(
                    body,
                    bodies,
                    drop_unsupported_sentences=False,
                    titles=_concept_titles(hits),
                )
                lines = [body, ""]
                lines += _sources_section(_cited_concepts(body, hits))
                lines += _disclaimer_section()
                return "\n".join(lines).strip(), True
        except Exception:
            logger.warning("report: LLM call failed, using the template report", exc_info=True)
    return _template_report(prediction, query, hits, bodies), False