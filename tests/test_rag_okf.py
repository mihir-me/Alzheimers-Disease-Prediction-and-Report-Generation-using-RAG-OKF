"""Tests for the OKF-aware RAG pipeline (loader, link expansion, modes, reports)."""
from __future__ import annotations

import dataclasses
import re
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.config import RAG_MODES, load_config  # noqa: E402
from rag_pipeline.okf_loader import load_bundle  # noqa: E402
from rag_pipeline import report_generator as rg  # noqa: E402
from rag_pipeline.report_generator import (  # noqa: E402
    CITATION_RE,
    generate_template_report,
)
from rag_pipeline.service import RagReportService  # noqa: E402
from rag_pipeline.vector_store import (  # noqa: E402
    ORIGIN_DIRECT,
    ORIGIN_STAGE_REQUIRED,
    Hit,
    VectorStore,
    origin_of,
)

BUNDLE_DIR = PROJECT_ROOT / "knowledge" / "okf_bundle"
EXPECTED_CONCEPT_COUNT = 27

SAMPLE_PAYLOAD = {
    "predicted_stage": "Very Mild Impairment",
    "probabilities": {
        "No Impairment": 0.08,
        "Very Mild Impairment": 0.61,
        "Mild Impairment": 0.22,
        "Moderate Impairment": 0.09,
    },
    "ensemble_confidence": 0.61,
    "clinical_inputs": {"MMSE": 24, "CDR": 0.5, "nWBV": 0.72},
    "clinical_diagnosis": "MCI",
    "clinical_demented_probability": 0.55,
    "fusion_diagnosis": "DEMENTED",
    "fusion_confidence": 0.6,
    "needs_clinical_review": True,
}
SAMPLE_QUERY = (
    "What do these results suggest about this patient's cognitive status "
    "and what should we do next?"
)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE_DIR)


@pytest.fixture
def cfg(tmp_path):
    """Offline config: no API key (hashing embeddings + template report)."""
    base = load_config()
    return dataclasses.replace(
        base,
        api_key="",
        knowledge_bundle_dir=BUNDLE_DIR,
        faiss_index_path=tmp_path / "faiss_index",
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        min_score=0.0,
        rag_mode="okf_rag",
        max_concepts=8,
    )


def service_for(cfg, mode: str) -> RagReportService:
    return RagReportService(dataclasses.replace(cfg, rag_mode=mode))


# --------------------------------------------------------------------------
# 1. loader
# --------------------------------------------------------------------------
def test_loader_finds_all_concepts(bundle):
    assert len(bundle) == EXPECTED_CONCEPT_COUNT
    ids = bundle.by_id()
    assert len(ids) == EXPECTED_CONCEPT_COUNT
    assert "scores/cdr_scale" in ids
    assert "scores/mmse" in ids
    assert "overview/clinical_stages_cdr" in ids


def test_loader_skips_index_files(bundle):
    assert not any(cid == "index" or cid.endswith("/index") for cid in bundle.by_id())


def test_concept_fields_are_populated(bundle):
    cdr = bundle.get("scores/cdr_scale")
    assert cdr is not None
    assert cdr.title == "CDR (Clinical Dementia Rating)"
    assert cdr.type == "Clinical Score"
    assert cdr.source == "scores/cdr_scale.md"
    assert "cdr" in cdr.tags
    assert cdr.description.startswith("The CDR is a 5-point scale")
    assert "5-point scale" in cdr.body
    assert cdr.links == [
        "overview/clinical_stages_cdr",
        "scores/mmse",
        "imaging/mri_biomarkers",
    ]
    assert isinstance(cdr.sources, list)


def test_loader_resolves_every_link(bundle):
    assert bundle.unresolved_links() == []
    known = set(bundle.by_id())
    linked = 0
    for concept in bundle:
        for link in concept.links:
            assert link in known, f"{concept.concept_id} -> {link}"
            assert link != concept.concept_id
            linked += 1
    assert linked > 0


# --------------------------------------------------------------------------
# 2. link expansion
# --------------------------------------------------------------------------
def test_link_expansion_adds_linked_concepts(cfg, bundle):
    service = service_for(cfg, "okf_rag")
    direct = service._vector_concepts(SAMPLE_QUERY, top_k=2)
    assert direct, "vector search returned nothing"
    direct_ids = [h.concept_id for h in direct]
    assert all(h.direct for h in direct)

    expanded = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD, top_k=2)
    expanded_ids = [h.concept_id for h in expanded]

    # direct hits come first, in rank order
    assert expanded_ids[: len(direct_ids)] == direct_ids
    linked = [h for h in expanded if not h.direct]
    assert linked, "link expansion produced no linked concepts"

    # every linked concept is a real outgoing link of a direct hit
    allowed = {
        link for cid in direct_ids for link in bundle.outgoing(cid)
    }
    for hit in linked:
        assert hit.concept_id in allowed
        assert hit.link_source in direct_ids
        assert hit.concept_id not in direct_ids
        assert 0.0 <= hit.score <= 1.0


def test_link_expansion_respects_cap(cfg):
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    assert 0 < len(hits) <= cfg.max_concepts
    assert len({h.concept_id for h in hits}) == len(hits)

    # even with a generous direct budget the total is capped
    service_big = service_for(dataclasses.replace(cfg, top_k=50), "okf_rag")
    capped = service_big.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD, top_k=50)
    assert len(capped) <= cfg.max_concepts


def test_rag_only_skips_link_expansion(cfg):
    service = service_for(cfg, "rag_only")
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    assert hits
    assert all(h.direct for h in hits)
    assert not any(h.link_source for h in hits)


# --------------------------------------------------------------------------
# 2b. chunking per concept
# --------------------------------------------------------------------------
def test_chunking_is_per_concept_with_metadata(cfg, bundle):
    from rag_pipeline.chunking import build_concept_chunks

    chunks = build_concept_chunks(bundle, cfg)
    assert chunks
    covered = {c.concept_id for c in chunks}
    assert covered == set(bundle.by_id())

    for chunk in chunks:
        assert chunk.concept_id
        assert chunk.title
        assert chunk.concept_type
        assert chunk.source == f"{chunk.concept_id}.md"
        # the concept title is prepended to the embedded text
        assert chunk.text.startswith(chunk.title)

    per_concept: dict[str, int] = {}
    for chunk in chunks:
        per_concept[chunk.concept_id] = per_concept.get(chunk.concept_id, 0) + 1
    short = [cid for cid in per_concept if per_concept[cid] == 1]
    long_concepts = [cid for cid in per_concept if per_concept[cid] > 1]
    assert short and long_concepts
    # a concept that fits in one chunk is never split or merged with another one
    for chunk in chunks:
        if per_concept[chunk.concept_id] == 1:
            assert len(chunk.text) <= cfg.chunk_size


# --------------------------------------------------------------------------
# 3. the three RAG modes
# --------------------------------------------------------------------------
def test_three_modes_return_different_concept_sets(cfg):
    # the modes are only distinguishable with room to spare: the cap is a budget
    # for what the *query* contributes, and when it is fully consumed by the
    # required concepts the link-expanding modes collapse onto `rag_only`.
    roomy = dataclasses.replace(cfg, max_concepts=12)
    modes = {mode: service_for(roomy, mode) for mode in RAG_MODES}
    results = {
        mode: service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
        for mode, service in modes.items()
    }
    ids = {mode: [h.concept_id for h in hits] for mode, hits in results.items()}
    for mode, mode_ids in ids.items():
        assert mode_ids, f"{mode} retrieved nothing"

    assert set(ids["rag_only"]) <= set(ids["okf_rag"]), ids
    assert len(ids["okf_rag"]) > len(ids["rag_only"])
    assert set(ids["okf_rag"]) != set(ids["rag_only"])
    assert set(ids["okf_only"]) != set(ids["rag_only"])
    assert set(ids["okf_only"]) != set(ids["okf_rag"])


def test_okf_only_matches_payload_without_vectors(cfg, monkeypatch):
    service = service_for(cfg, "okf_only")
    monkeypatch.setattr(
        service, "_vector_concepts", lambda *a, **k: pytest.fail("vectors used")
    )
    assert service.ensure_index() >= 0
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    ids = [h.concept_id for h in hits]
    # CDR / MMSE / nWBV / stage knowledge for a CDR 0.5 + MMSE 24 patient
    assert "scores/cdr_scale" in ids or "scores/mmse" in ids
    assert any(cid.startswith("scores/") for cid in ids)


def test_okf_only_ignores_empty_payload(cfg):
    service = service_for(cfg, "okf_only")
    payload_hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    other = service.retrieve(SAMPLE_QUERY, prediction=None)
    assert [h.concept_id for h in payload_hits] != [h.concept_id for h in other]


# --------------------------------------------------------------------------
# 3c. stage-driven required concepts
# --------------------------------------------------------------------------
CAVEATS = "diagnosis/ai_assisted_prediction_caveats"
EARLY_STAGE = "treatment/early_stage_guidance"
LIFESTYLE = "treatment/lifestyle_risk_modification"
MODIFIABLE_RISK = "risk/modifiable_risk_reduction"
NON_PHARMA = "treatment/non_pharmacological"
RED_FLAGS = "diagnosis/red_flags_referral"
STAGE_CONCEPT = "overview/clinical_stages_cdr"
SCORE_CONCEPTS = {
    "cdr": "scores/cdr_scale",
    "mmse": "scores/mmse",
    "nwbv": "scores/nwbv",
}

# predicted stage -> (CDR, MMSE) of the sample patient
STAGE_CASES = {
    "No Impairment": (0.0, 29),
    "Very Mild Impairment": (0.5, 24),
    "Mild Impairment": (1.0, 20),
    "Moderate Impairment": (2.0, 15),
}


def payload_for(stage: str, cdr: float, mmse: int) -> dict:
    """Minimal prediction payload for a stage / clinical-score combination."""
    return {
        "predicted_stage": stage,
        "probabilities": {stage: 0.7},
        "clinical_inputs": {"MMSE": mmse, "CDR": cdr, "nWBV": 0.72},
    }


def section_of(report: str, heading: str) -> str:
    """Body of one ``## heading`` section, up to the next heading."""
    body = report.split(f"## {heading}", 1)[1]
    return body.split("\n## ", 1)[0]


def sources_ids(report: str) -> set[str]:
    """Concept ids listed in the Sources section."""
    block = report.split("## Sources", 1)[1].split("## Disclaimer", 1)[0]
    return set(re.findall(r"^- `([^`]+)`", block, re.MULTILINE))


def test_required_concepts_follow_the_stage_and_the_scores(cfg):
    service = service_for(cfg, "okf_rag")
    required = {
        stage: set(service._required_concept_ids(payload_for(stage, cdr, mmse)))
        for stage, (cdr, mmse) in STAGE_CASES.items()
    }
    # the prediction caveats belong in every report
    for stage, ids in required.items():
        assert CAVEATS in ids, stage

    assert EARLY_STAGE in required["Very Mild Impairment"]
    assert EARLY_STAGE in required["Mild Impairment"]
    assert {LIFESTYLE, MODIFIABLE_RISK} <= required["No Impairment"]
    assert {NON_PHARMA, RED_FLAGS} <= required["Moderate Impairment"]

    # the CDR thresholds apply on their own, whatever stage was predicted
    assert EARLY_STAGE in set(
        service._required_concept_ids(payload_for("Unknown Stage", 0.5, 24))
    )
    assert {NON_PHARMA, RED_FLAGS} <= set(
        service._required_concept_ids(payload_for("Unknown Stage", 2.0, 15))
    )


def test_required_concepts_always_cover_the_stage_and_the_supplied_scores(cfg):
    """The stage concept is always required; a scale concept only when supplied."""
    service = service_for(cfg, "okf_rag")

    for stage, (cdr, mmse) in STAGE_CASES.items():
        ids = service._required_concept_ids(payload_for(stage, cdr, mmse))
        assert STAGE_CONCEPT in ids, stage
        # every score in the payload brings its own scale concept
        assert SCORE_CONCEPTS["cdr"] in ids, stage
        assert SCORE_CONCEPTS["mmse"] in ids, stage
        assert SCORE_CONCEPTS["nwbv"] in ids, stage
        # and nothing is required twice
        assert len(ids) == len(set(ids)), stage

    # no score supplied -> no scale concept for it
    cdr_only = service._required_concept_ids(
        {"predicted_stage": "Mild Impairment", "clinical_inputs": {"CDR": 1.0}}
    )
    assert SCORE_CONCEPTS["cdr"] in cdr_only
    assert SCORE_CONCEPTS["mmse"] not in cdr_only
    assert SCORE_CONCEPTS["nwbv"] not in cdr_only
    assert STAGE_CONCEPT in cdr_only

    # an empty payload still gets the stage, so the stage sentence can render
    bare = service._required_concept_ids({"predicted_stage": "No Impairment"})
    assert STAGE_CONCEPT in bare
    assert not any(cid in bare for cid in SCORE_CONCEPTS.values())

    # every required concept really exists in the bundle
    bundle_ids = set(service.bundle.by_id())
    for stage, (cdr, mmse) in STAGE_CASES.items():
        for concept_id in service._required_concept_ids(
            payload_for(stage, cdr, mmse)
        ):
            assert concept_id in bundle_ids, (stage, concept_id)


def test_required_concepts_are_retrieved_in_every_mode(cfg):
    for stage, (cdr, mmse) in STAGE_CASES.items():
        payload = payload_for(stage, cdr, mmse)
        for mode in RAG_MODES:
            service = service_for(cfg, mode)
            hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
            ids = [hit.concept_id for hit in hits]
            assert CAVEATS in ids, (mode, stage)
            assert len(ids) == len(set(ids)), (mode, stage)
            assert len(hits) <= cfg.max_concepts, (mode, stage)


def test_added_required_hits_are_marked_stage_required_after_the_direct_ones(cfg):
    service = service_for(cfg, "okf_rag")
    payload = payload_for("Moderate Impairment", 2.0, 15)
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    labels = [origin_of(hit) for hit in hits]

    required = service._required_concept_ids(payload)
    origins = {hit.concept_id: origin_of(hit) for hit in hits}
    for concept_id in required:
        if concept_id in origins:
            assert origins[concept_id] in {ORIGIN_STAGE_REQUIRED, ORIGIN_DIRECT}

    assert ORIGIN_STAGE_REQUIRED in labels
    # direct hits are kept, and stay ahead of the concepts added for the stage
    direct_at = [i for i, label in enumerate(labels) if label == ORIGIN_DIRECT]
    required_at = [i for i, label in enumerate(labels) if label == ORIGIN_STAGE_REQUIRED]
    assert direct_at and required_at
    assert max(direct_at) < min(required_at)


def test_required_concepts_reached_by_link_expansion_are_marked_stage_required(cfg):
    """A required concept keeps the `stage-required` origin however it was found."""
    service = service_for(cfg, "okf_rag")
    payload = payload_for("Moderate Impairment", 2.0, 15)
    required = set(service._required_concept_ids(payload))
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)

    for hit in hits:
        if hit.concept_id in required:
            assert origin_of(hit) in {ORIGIN_STAGE_REQUIRED, ORIGIN_DIRECT}, (
                hit.concept_id,
                origin_of(hit),
            )

    # the promotion is what makes this observable: a required concept can only be
    # reached by a link from a direct hit, never by the vector search itself
    promoted = [
        h.concept_id
        for h in hits
        if h.concept_id in required and not h.direct
    ]
    for concept_id in promoted:
        hit = next(h for h in hits if h.concept_id == concept_id)
        assert hit.link_source, concept_id
        assert origin_of(hit) == ORIGIN_STAGE_REQUIRED


def test_required_concepts_are_never_evicted_by_the_cap(cfg):
    """A cap below the required set may not cost the report a required concept."""
    tight = dataclasses.replace(cfg, max_concepts=3)
    service = service_for(tight, "okf_rag")
    payload = payload_for("Moderate Impairment", 2.0, 15)
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    ids = [hit.concept_id for hit in hits]

    required = service._required_concept_ids(payload)
    assert len(required) > tight.max_concepts, "the cap is not below the required set"
    # every required concept survives even though together they exceed the cap
    assert set(required) <= set(ids), (set(required) - set(ids), ids)
    # the required concepts are all that survives: no query hit gets a slot
    assert set(ids) == set(required)
    assert len(ids) == len(required) > tight.max_concepts
    assert STAGE_CONCEPT in ids
    assert CAVEATS in ids
    assert {SCORE_CONCEPTS["cdr"], SCORE_CONCEPTS["mmse"]} <= set(ids)


def test_linked_concepts_give_way_before_direct_ones(cfg):
    """Overflow is paid for by the lowest-scoring non-required linked hits."""
    payload = payload_for("Moderate Impairment", 2.0, 15)
    reference = service_for(cfg, "okf_rag")
    required = set(reference._required_concept_ids(payload))
    assert len(required) >= 6

    # one slot of headroom above the required set: there is room for a query hit,
    # and a linked concept is the one that has to give way for it
    roomy = service_for(dataclasses.replace(cfg, max_concepts=len(required) + 3), "okf_rag")
    roomy_hits = roomy.retrieve(SAMPLE_QUERY, prediction=payload)
    optional = [h for h in roomy_hits if h.concept_id not in required]
    assert any(not h.direct for h in optional), "no linked concept to sacrifice"
    assert any(h.direct for h in optional), "no direct concept competing with it"

    tight = service_for(dataclasses.replace(cfg, max_concepts=len(required) + 1), "okf_rag")
    tight_hits = tight.retrieve(SAMPLE_QUERY, prediction=payload)
    assert len(tight_hits) <= len(required) + 1
    kept = [h for h in tight_hits if h.concept_id not in required]
    # the non-required linked concept was evicted, the direct hit was kept
    assert [h for h in kept if not h.direct] == []
    assert [h for h in kept if h.direct]


# --------------------------------------------------------------------------
# 4. report citations / sources / disclaimer
# --------------------------------------------------------------------------
def test_very_mild_report_recommends_early_stage_guidance(cfg):
    service = service_for(cfg, "okf_rag")
    payload = payload_for("Very Mild Impairment", 0.5, 24)
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

    assert "## Recommended Next Steps" in report
    steps = section_of(report, "Recommended Next Steps")
    assert f"[concept: {EARLY_STAGE}]" in steps
    assert EARLY_STAGE in {m.strip() for m in CITATION_RE.findall(report)}


def test_no_impairment_report_recommends_lifestyle_or_risk_concepts(cfg):
    service = service_for(cfg, "okf_rag")
    payload = payload_for("No Impairment", 0.0, 29)
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

    assert "## Recommended Next Steps" in report
    steps = section_of(report, "Recommended Next Steps")
    cited = {m.strip() for m in CITATION_RE.findall(steps)}
    assert cited & {LIFESTYLE, MODIFIABLE_RISK}, steps


@pytest.mark.parametrize("stage", sorted(STAGE_CASES))
def test_caveat_concept_supports_the_risk_section_for_every_stage(cfg, stage):
    cdr, mmse = STAGE_CASES[stage]
    payload = payload_for(stage, cdr, mmse)
    for mode in RAG_MODES:
        service = service_for(cfg, mode)
        hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
        assert CAVEATS in {hit.concept_id for hit in hits}, (mode, stage)

        report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )
        assert "## Risk Interpretation" in report, (mode, stage)
        risk = section_of(report, "Risk Interpretation")
        assert f"[concept: {CAVEATS}]" in risk, (mode, stage)


# --------------------------------------------------------------------------
# 3d. report integrity: bare checklists, stage sentence, score lines, A3 note
# --------------------------------------------------------------------------
CHECKLIST_ITEMS = (
    "Rapidly progressive symptoms",
    "New focal neurological findings",
    "Seizures",
    "Unexplained weight loss",
)


def test_a_bare_checklist_criterion_is_never_presented_as_a_finding(cfg, bundle):
    """`red_flags_referral` is a pure checklist: it is quoted whole, behind a lead-in."""
    concept = bundle.get(RED_FLAGS)
    assert concept, "the referral checklist concept is missing from the bundle"
    items = rg._checklist_items(concept.body)
    assert len(items) >= 3, "the fixture is no longer a bare checklist"

    payload = payload_for("Moderate Impairment", 2.0, 15)
    for mode in RAG_MODES:
        service = service_for(cfg, mode)
        hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
        assert RED_FLAGS in {hit.concept_id for hit in hits}, mode
        report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

        lead_in = rg._lead_in_from_title(concept.title)
        assert lead_in, concept.title
        # the whole list arrives behind the lead-in, in one statement
        assert lead_in in report, (mode, report)
        for item in items:
            assert item in report, (mode, item)
        # and no line offers a single criterion on its own as if it were a finding
        for line in report.splitlines():
            for item in items:
                if item in line:
                    assert lead_in in line, (mode, line)


def test_llm_output_cannot_leak_a_bare_checklist_criterion(cfg, bundle):
    """The citation guard alone keeps the criterion: the lead-in must be restored."""
    concept = bundle.get(RED_FLAGS)
    items = rg._checklist_items(concept.body)
    bodies = {RED_FLAGS: concept.body}
    titles = {RED_FLAGS: concept.title}

    # a single lifted criterion, cited to the checklist concept, in LLM wording
    lifted = (
        f"{items[0]} is present in this patient "
        f"[concept: {RED_FLAGS}]."
    )
    out = rg._enforce_citation_support(
        lifted, bodies, drop_unsupported_sentences=False, titles=titles
    )

    lead_in = rg._lead_in_from_title(concept.title)
    assert lead_in in out, out
    for item in items:
        assert item in out, (item, out)
    # the claim that it "is present in this patient" is not supported and must go
    assert "is present in this patient" not in out, out


def test_checklist_lead_in_is_not_re_added_to_an_already_led_sentence(cfg, bundle):
    """A criterion that already sits behind its lead-in is left as it is."""
    concept = bundle.get(RED_FLAGS)
    items = rg._checklist_items(concept.body)
    bodies = {RED_FLAGS: concept.body}
    titles = {RED_FLAGS: concept.title}
    lead_in = rg._lead_in_from_title(concept.title)

    already = f"{lead_in} {'; '.join(items)} [concept: {RED_FLAGS}]"
    out = rg._enforce_citation_support(
        already, bodies, drop_unsupported_sentences=False, titles=titles
    )
    assert out.count(lead_in) == 1, out

    # a different, non-checklist concept is untouched by the rule
    mmse = bundle.get("scores/mmse")
    other = f"MMSE is scored out of thirty [concept: scores/mmse]."
    out = rg._enforce_citation_support(
        other,
        {"scores/mmse": mmse.body},
        drop_unsupported_sentences=False,
        titles={"scores/mmse": mmse.title},
    )
    assert "scored out of thirty" in out, out


def test_severity_note_fires_when_the_stage_is_the_outlier(cfg):
    """A3: the app's class names have to be on the severity ladder for this to work.

    `splits.CLASSES` predicts "Moderate Impairment", not "moderate dementia"; when
    the stage name is not recognised the note silently disappears, which is worst
    exactly when the imaging stage is the band that disagrees with the scores.
    """
    assert rg._stage_tier("Moderate Impairment") is not None
    assert rg._stage_tier("Mild Impairment") is not None
    assert rg._stage_tier("Very Mild Impairment") is not None
    assert rg._stage_tier("No Impairment") is not None
    # the knowledge base's own wording still maps to the same rungs
    assert rg._stage_tier("Moderate Impairment") == rg._stage_tier("moderate dementia")
    assert rg._stage_tier("Mild Impairment") == rg._stage_tier("mild dementia")

    payload = {
        "predicted_stage": "Moderate Impairment",
        "probabilities": {"Moderate Impairment": 0.8},
        "clinical_inputs": {"CDR": 0.0, "MMSE": 29, "nWBV": 0.75},
    }
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

    note = rg._SEVERITY_NOTE
    assert report.count(note) == 1, report
    # the note is structural guidance, so it carries no citation
    line = next(line for line in report.splitlines() if note in line)
    assert "[concept:" not in line, line


def test_no_severity_note_when_the_bands_agree(cfg):
    """The note is only for disagreement: agreeing bands must not carry it."""
    payload = {
        "predicted_stage": "Moderate Impairment",
        "probabilities": {"Moderate Impairment": 0.8},
        "clinical_inputs": {"CDR": 2.0, "MMSE": 15, "nWBV": 0.62},
    }
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )
    assert rg._SEVERITY_NOTE not in report, report


@pytest.mark.parametrize("stage", sorted(STAGE_CASES))
def test_stage_sentence_and_one_cited_line_per_score(cfg, stage):
    """A2 in the report: the stage is described and every supplied score has a line."""
    cdr, mmse = STAGE_CASES[stage]
    payload = payload_for(stage, cdr, mmse)
    for mode in RAG_MODES:
        service = service_for(cfg, mode)
        hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
        report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

        cited = {m.strip() for m in CITATION_RE.findall(report)}
        assert STAGE_CONCEPT in cited, (mode, stage)
        assert SCORE_CONCEPTS["cdr"] in cited, (mode, stage)
        assert SCORE_CONCEPTS["mmse"] in cited, (mode, stage)
        assert SCORE_CONCEPTS["nwbv"] in cited, (mode, stage)

        # one cited statement per supplied score, each naming the score
        for key in ("CDR", "MMSE", "nWBV"):
            lines = [
                line
                for line in section_of(report, "Clinical Scores Interpretation").splitlines()
                if f"[concept: {SCORE_CONCEPTS[key.lower()]}]" in line
            ]
            assert lines, (mode, stage, key)
            assert any(key in line for line in lines), (mode, stage, key)


def test_moderate_cdr_2_mmse_21_has_a_stage_sentence_and_a_cdr_line(cfg):
    """A2 on the reported case: Moderate / CDR 2 / MMSE 21.

    MMSE 21 sits in the mild band while CDR 2 and the imaging stage are
    moderate, so this payload also exercises A3: the bands disagree and the one
    neutral, uncited note has to be there exactly once.
    """
    payload = payload_for("Moderate Impairment", 2.0, 21)
    for mode in RAG_MODES:
        service = service_for(cfg, mode)
        hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
        report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

        # the Clinical Summary always carries the predicted stage, cited
        summary = section_of(report, "Clinical Summary")
        assert "Moderate Impairment" in summary, (mode, summary)
        assert f"[concept: {STAGE_CONCEPT}]" in summary, (mode, summary)

        # and the Clinical Scores section has a cited line for CDR 2
        scores = section_of(report, "Clinical Scores Interpretation")
        cdr_lines = [
            line
            for line in scores.splitlines()
            if f"[concept: {SCORE_CONCEPTS['cdr']}]" in line
        ]
        assert cdr_lines, (mode, scores)
        assert any("CDR" in line and "2" in line for line in cdr_lines), (mode, cdr_lines)

        # the three bands disagree by two rungs, so the neutral note is present
        # exactly once and, being structural, carries no citation
        note_lines = [line for line in report.splitlines() if rg._SEVERITY_NOTE in line]
        assert len(note_lines) == 1, (mode, report)
        assert "[concept:" not in note_lines[0], note_lines[0]


# lines of the template report that are structure rather than clinical text:
# headings, the stage/query banners, the pointer to clinical context, the score
# table and the model's own probability breakdown
_STRUCTURAL_LINE_RE = re.compile(
    r"^(?:"
    r"\#{1,6}\s\S"                            # a heading
    r"|\*\*Predicted cognitive stage: "       # the stage banner
    r"|\*\*Query:\*\* "                       # the query banner
    r"|Interpret the predicted stage above"   # structural pointer
    r"|\|[^|]*\|"                             # a markdown table row
    r"|-\s+\*\*[^*]+:\*\*\s+\d"              # a model probability line
    r")"
)


def test_the_only_uncited_clinical_text_is_the_severity_note(cfg):
    """A3: outside structural text, the neutral note is the only uncited line."""
    payload = payload_for("Moderate Impairment", 2.0, 21)
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
    report = generate_template_report(
        payload,
        SAMPLE_QUERY,
        hits,
        bodies=rg.concept_bodies(service.cfg, hits),
    )

    body = report.split("## Sources", 1)[0]
    severity_notes = 0
    unexpected: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or "[concept:" in stripped:
            continue
        if stripped == rg._SEVERITY_NOTE:
            severity_notes += 1
            continue
        if _STRUCTURAL_LINE_RE.match(stripped):
            continue
        unexpected.append(stripped)

    assert unexpected == [], unexpected
    # one note at most, and only when the bands actually disagree
    assert severity_notes <= 1, report


@pytest.mark.parametrize("stage", sorted(STAGE_CASES))
def test_sources_section_equals_the_inline_citations(cfg, stage):
    cdr, mmse = STAGE_CASES[stage]
    payload = payload_for(stage, cdr, mmse)
    for mode in RAG_MODES:
        service = service_for(cfg, mode)
        hits = service.retrieve(SAMPLE_QUERY, prediction=payload)
        report = generate_template_report(
            payload,
            SAMPLE_QUERY,
            hits,
            bodies=rg.concept_bodies(service.cfg, hits),
        )

        cited = {m.strip() for m in CITATION_RE.findall(report)}
        assert cited, (mode, stage)
        assert sources_ids(report) == cited, (mode, stage)
        assert sources_ids(report) <= {hit.concept_id for hit in hits}, (mode, stage)


def test_report_sections_are_omitted_rather_than_invented(cfg):
    """With no support for a section, it disappears instead of being faked."""
    from rag_pipeline import report_generator as rg

    unsupported = [
        Hit(
            text="MMSE is a cognitive screening instrument.",
            source="scores/mmse.md",
            title="MMSE",
            score=0.5,
            concept_id="scores/mmse",
            concept_type="Clinical Score",
        )
    ]
    report = rg.generate_template_report(
        {"predicted_stage": "Moderate Impairment", "probabilities": {}},
        "what next?",
        unsupported,
    )
    assert "## Recommended Next Steps" not in report
    assert "## Risk Interpretation" not in report
    assert not CITATION_RE.findall(report)
    assert "## Sources" in report


def test_template_report_cites_concepts_and_lists_sources(cfg):
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    assert hits
    report = generate_template_report(SAMPLE_PAYLOAD, SAMPLE_QUERY, hits)

    assert CITATION_RE.search(report), "template report has no [concept: ...] citation"
    cited = {m.strip() for m in CITATION_RE.findall(report)}
    retrieved = {h.concept_id for h in hits}
    assert cited and cited <= retrieved

    assert "## Sources" in report
    sources_block = report.split("## Sources", 1)[1]
    cited = {m.strip() for m in CITATION_RE.findall(report)}
    for hit in hits:
        cid = hit.concept_id or hit.source
        if cid == 'reporting/report_template_guidelines' or cid.startswith('reporting/') or hit.concept_type == 'Report Guideline':
            assert cid not in cited
            continue
        # under new rules, only cited concepts appear in Sources
    assert cited.issubset({h.concept_id or h.source for h in hits})

    assert "not a clinical diagnosis" in report
    assert report.rstrip().endswith(
        "A qualified clinician must confirm any findings."
    )
    assert report.index("## Sources") < report.index("## Disclaimer")


def test_template_report_does_not_give_dosages(cfg):
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    report = generate_template_report(SAMPLE_PAYLOAD, SAMPLE_QUERY, hits).lower()
    for banned in (" mg", " mg.", "mg/dose", "twice daily", "milligrams"):
        assert banned not in report


def test_report_generator_drops_unknown_citations(cfg, monkeypatch):
    from rag_pipeline import report_generator as rg

    hit = Hit(
        text="CDR 0 means no dementia.",
        source="scores/cdr_scale.md",
        title="CDR (Clinical Dementia Rating)",
        score=0.5,
        concept_id="scores/cdr_scale",
        concept_type="Clinical Score",
    )
    client = _FakeClient(
        "No dementia [concept: scores/cdr_scale]. "
        "MRI is normal [concept: imaging/does_not_exist]. "
        "Give donepezil 10 mg [concept: scores/cdr_scale].\n\n"
        "## Sources\n- fabricated [concept: imaging/does_not_exist]\n\n"
        "## Disclaimer\nMade up."
    )
    monkeypatch.setattr("openai.OpenAI", lambda *a, **k: client)
    llm_cfg = dataclasses.replace(cfg, api_key="test-key-not-used", llm_model="fake")
    report, used_llm = rg.generate_report(llm_cfg, SAMPLE_PAYLOAD, SAMPLE_QUERY, [hit])

    assert used_llm
    assert "[concept: scores/cdr_scale]" in report
    assert "does_not_exist" not in report
    assert "10 mg" not in report
    assert "## Sources" in report
    assert "CDR (Clinical Dementia Rating)" in report
    assert report.rstrip().endswith("A qualified clinician must confirm any findings.")


def test_report_generator_falls_back_to_template_on_llm_error(cfg, monkeypatch):
    from rag_pipeline import report_generator as rg

    def _boom(*_args, **_kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr("openai.OpenAI", _boom)
    llm_cfg = dataclasses.replace(cfg, api_key="test-key-not-used", llm_model="fake")
    service = service_for(cfg, "okf_rag")
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    report, used_llm = rg.generate_report(llm_cfg, SAMPLE_PAYLOAD, SAMPLE_QUERY, hits)
    assert used_llm is False
    assert CITATION_RE.search(report)
    assert "## Sources" in report


# --------------------------------------------------------------------------
# 5. index rebuild on bundle / embedding-mode change
# --------------------------------------------------------------------------
def test_index_is_reused_when_nothing_changed(cfg):
    service = service_for(cfg, "okf_rag")
    first = service.ensure_index()
    assert first > 0
    service.index_rebuilt = False
    second = service.ensure_index()
    assert second == first
    assert service.index_rebuilt is False


def test_index_rebuilds_when_bundle_hash_changes(cfg, tmp_path):
    bundle_copy = tmp_path / "bundle"
    shutil.copytree(BUNDLE_DIR, bundle_copy)
    local_cfg = dataclasses.replace(cfg, knowledge_bundle_dir=bundle_copy)

    service = RagReportService(local_cfg)
    n_before = service.ensure_index()
    assert n_before > 0
    signature_before = service.store.signature
    assert service.index_rebuilt is True

    # a second service over the unchanged bundle reuses the persisted index
    reused = RagReportService(local_cfg)
    reused.index_rebuilt = False
    assert reused.ensure_index() == n_before
    assert reused.index_rebuilt is False
    assert reused.store.signature == signature_before

    # mutate the bundle: the index must be rebuilt automatically
    new_concept = bundle_copy / "scores" / "mmse_followup.md"
    new_concept.write_text(
        "---\ntype: Clinical Score\n---\n\n# MMSE Follow-up Interval\n\n"
        "Repeat the MMSE at six to twelve month intervals.\n\n"
        "Related: [MMSE](/scores/mmse.md).\n",
        encoding="utf-8",
    )
    changed = RagReportService(local_cfg)
    changed.index_rebuilt = False
    n_after = changed.ensure_index()
    assert changed.index_rebuilt is True
    assert changed.store.signature != signature_before
    assert n_after > n_before
    assert "scores/mmse_followup" in {
        m["concept_id"] for m in changed.store.metadata
    }


def test_index_rebuilds_when_embedding_mode_changes(cfg):
    service = service_for(cfg, "okf_rag")
    service.ensure_index()
    local_signature = service.store.signature

    # switching from the deterministic hashing embedder to the API embedder
    # changes the signature, so the persisted index is not reused
    remote = RagReportService(dataclasses.replace(cfg, api_key="test-key-not-used"))
    assert remote.cfg.index_signature(remote.bundle_hash) != local_signature

    # changing the embedding model alone also invalidates the index
    other_model = dataclasses.replace(cfg, embedding_model="other-embedding-model")
    other_service = RagReportService(other_model)
    assert (
        other_model.index_signature(other_service.bundle_hash) != local_signature
    )


def test_vector_store_persists_signature(cfg):
    service = service_for(cfg, "okf_rag")
    service.ensure_index()
    path = service.store.save()
    assert path.with_suffix(".index").is_file()
    assert path.with_suffix(".pkl").is_file()

    reloaded = VectorStore.load(cfg, service.embedder)
    assert reloaded.signature == service.store.signature
    assert reloaded.n_total == service.store.n_total


def test_invalid_rag_mode_rejected(cfg):
    with pytest.raises(ValueError):
        RagReportService(dataclasses.replace(cfg, rag_mode="nonsense"))


# --------------------------------------------------------------------------
# helpers: fake OpenAI-compatible client
# --------------------------------------------------------------------------
class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_sentence_dropped_when_concept_lacks_content(cfg):
    from rag_pipeline import report_generator as rg
    import dataclasses

    hit = rg.Hit(
        text='MMSE is a cognitive screening instrument.',
        source='scores/mmse.md',
        title='MMSE',
        score=0.5,
        concept_id='scores/mmse',
        concept_type='Clinical Score',
    )
    # Create a fake hit with different concept id but same? No, we need to test
    # a sentence citing a concept whose body lacks its content is dropped
    # Simpler: mock bodies
    bodies = {'scores/mmse': 'The CDR is a 5-point scale.'}  # body doesn't mention MMSE content
    report = rg._enforce_citation_support(
        'MMSE scores are commonly interpreted [concept: scores/mmse].',
        bodies,
        drop_unsupported_sentences=True,
    )
    assert 'MMSE scores' not in report
    assert 'scores/mmse' not in report


def test_sources_equals_inline_citations_only(cfg):
    from rag_pipeline import report_generator as rg
    import dataclasses

    hits = [
        rg.Hit(text='A', source='s/a.md', title='A', score=0.9, concept_id='s/a', concept_type='C'),
        rg.Hit(text='B', source='s/b.md', title='B', score=0.8, concept_id='s/b', concept_type='C'),
    ]
    # Simulate report with only a cited
    report = 'A statement [concept: s/a].\n\n## Sources\n\n'
    cited = rg._cited_concepts('A statement [concept: s/a].', hits)
    # build sources
    lines = [''] + ['## Sources', ''] + rg._concept_lines(cited)
    assert 's/b' not in '\n'.join(lines)
    assert 's/a' in '\n'.join(lines)


def test_report_template_guidelines_never_cited_or_listed(cfg):
    from rag_pipeline import report_generator as rg
    service = service_for(cfg, 'okf_rag')
    from tests.test_rag_okf import SAMPLE_QUERY, SAMPLE_PAYLOAD
    hits = service.retrieve(SAMPLE_QUERY, prediction=SAMPLE_PAYLOAD)
    report = rg.generate_template_report(SAMPLE_PAYLOAD, SAMPLE_QUERY, hits)
    for m in rg.CITATION_RE.findall(report):
        assert not m.startswith('reporting/')
    if '## Sources' in report:
        assert 'report_template_guidelines' not in report.split('## Sources', 1)[1]
