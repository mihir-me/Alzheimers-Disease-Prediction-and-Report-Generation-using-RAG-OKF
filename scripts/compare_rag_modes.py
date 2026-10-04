#!/usr/bin/env python3
"""Compare the three RAG retrieval modes on three sample prediction payloads.

Runs ``okf_rag``, ``rag_only`` and ``okf_only`` over a No Impairment (CDR 0),
a Very Mild / MCI (CDR 0.5) and a Moderate (CDR 2) case, renders the report
with the deterministic template generator (no LLM call, no network) and writes
the retrieved concept ids plus the generated reports to
``docs/rag_mode_comparison.md``.

Usage::

    python scripts/compare_rag_modes.py
    python scripts/compare_rag_modes.py --output docs/rag_mode_comparison.md
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.config import RAG_MODES, RagConfig, load_config  # noqa: E402
from rag_pipeline.report_generator import (  # noqa: E402
    CITATION_RE,
    concept_bodies,
    generate_template_report,
)
from rag_pipeline.service import RagReportService  # noqa: E402
from rag_pipeline.vector_store import (  # noqa: E402
    ORIGIN_DIRECT,
    ORIGIN_LINKED,
    ORIGIN_STAGE_REQUIRED,
    Hit,
    origin_of,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "rag_mode_comparison.md"
SHARED_QUERY = (
    "What do these results suggest about this patient's cognitive status "
    "and what should we do next?"
)

CASES: tuple[dict, ...] = (
    {
        "name": "Case A - No Impairment (CDR 0)",
        "query": "MMSE 29 and CDR 0 - is this normal, and what follow-up is needed?",
        "payload": {
            "predicted_stage": "No Impairment",
            "probabilities": {
                "No Impairment": 0.78,
                "Very Mild Impairment": 0.14,
                "Mild Impairment": 0.06,
                "Moderate Impairment": 0.02,
            },
            "ensemble_confidence": 0.78,
            "clinical_inputs": {"MMSE": 29, "CDR": 0.0, "nWBV": 0.79},
            "clinical_diagnosis": "NON-DEMENTED",
            "clinical_demented_probability": 0.06,
            "clinical_non_demented_probability": 0.94,
            "fusion_diagnosis": "NON-DEMENTED",
            "fusion_confidence": 0.93,
            "fusion_demented_probability": 0.07,
            "fusion_non_demented_probability": 0.93,
            "mri_stage_implies_dementia": False,
            "mri_clinical_conflict": False,
            "needs_clinical_review": False,
        },
    },
    {
        "name": "Case B - Very Mild / MCI (CDR 0.5)",
        "query": "CDR 0.5 with an MMSE of 24 - does this count as MCI, and what next?",
        "payload": {
            "predicted_stage": "Very Mild Impairment",
            "probabilities": {
                "No Impairment": 0.11,
                "Very Mild Impairment": 0.57,
                "Mild Impairment": 0.24,
                "Moderate Impairment": 0.08,
            },
            "ensemble_confidence": 0.57,
            "clinical_inputs": {"MMSE": 24, "CDR": 0.5, "nWBV": 0.74},
            "clinical_diagnosis": "MCI",
            "clinical_demented_probability": 0.44,
            "clinical_non_demented_probability": 0.56,
            "fusion_diagnosis": "DEMENTED",
            "fusion_confidence": 0.52,
            "fusion_demented_probability": 0.52,
            "fusion_non_demented_probability": 0.48,
            "mri_stage_implies_dementia": False,
            "mri_clinical_conflict": True,
            "needs_clinical_review": True,
        },
    },
    {
        "name": "Case C - Moderate impairment (CDR 2)",
        "query": "CDR 2 and MMSE 15 - what treatment and care options apply at this stage?",
        "payload": {
            "predicted_stage": "Moderate Impairment",
            "probabilities": {
                "No Impairment": 0.01,
                "Very Mild Impairment": 0.04,
                "Mild Impairment": 0.17,
                "Moderate Impairment": 0.78,
            },
            "ensemble_confidence": 0.78,
            "clinical_inputs": {"MMSE": 15, "CDR": 2.0, "nWBV": 0.68},
            "clinical_diagnosis": "DEMENTED",
            "clinical_demented_probability": 0.88,
            "clinical_non_demented_probability": 0.12,
            "fusion_diagnosis": "DEMENTED",
            "fusion_confidence": 0.9,
            "fusion_demented_probability": 0.9,
            "fusion_non_demented_probability": 0.1,
            "mri_stage_implies_dementia": True,
            "mri_clinical_conflict": False,
            "needs_clinical_review": True,
        },
    },
)

MODE_NOTES = {
    "okf_rag": "FAISS vector search over OKF concept chunks + one-hop link expansion.",
    "rag_only": "FAISS vector search only, no link expansion.",
    "okf_only": (
        "No vectors: concepts matched by tags / types / keywords from the "
        "prediction payload (CDR, MMSE, nWBV, stage), then one-hop link expansion."
    ),
}


def offline_config(index_path: Path | None) -> RagConfig:
    """Deterministic, offline configuration (hashing embeddings, no LLM)."""
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        api_key="",  # keep the hashing embedder and the template report
        faiss_index_path=index_path if index_path else cfg.faiss_index_path,
    )


def repo_relative(path: Path) -> str:
    """Path relative to the repository root, so the generated doc is portable.

    The markdown is committed, so an absolute path would bake the generating
    machine's home directory into it.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _concept_lines(hits: list[Hit]) -> list[str]:
    lines = ["| # | concept_id | title | type | score | origin |", "| --- | --- | --- | --- | --- | --- |"]
    for i, hit in enumerate(hits, 1):
        origin = origin_of(hit)
        if origin == ORIGIN_STAGE_REQUIRED:
            # the score of a stage-required hit is a marker, not a relevance
            # score: the stage demands the concept, so it is never ranked out
            score = "_(required)_"
            label = "**stage-required**"
        else:
            score = f"{hit.score:.3f}"
            label = (
                "direct"
                if origin == ORIGIN_DIRECT
                else f"linked from `{hit.link_source}`"
            )
        lines.append(
            f"| {i} | `{hit.concept_id}` | {hit.title} | {hit.concept_type} | "
            f"{score} | {label} |"
        )
    if not hits:
        lines.append("| - | *(nothing retrieved)* | - | - | - | - |")
    return lines


def build_markdown(services: dict[str, RagReportService], shared_query: str | None) -> str:
    results: dict[str, dict[str, dict]] = {}
    for case in CASES:
        query = case.get("query") or shared_query or SHARED_QUERY
        results[case["name"]] = {}
        for mode in RAG_MODES:
            service = services[mode]
            hits = service.retrieve(query, prediction=case["payload"])
            # full concept bodies, exactly as the report service passes them, so
            # the rendered report is not limited to a single indexed chunk
            report = generate_template_report(
                case["payload"],
                query,
                hits,
                bodies=concept_bodies(service.cfg, hits),
            )
            results[case["name"]][mode] = {
                "query": query,
                "hits": hits,
                "ids": [h.concept_id for h in hits],
                "report": report,
                "citations": sorted({m.strip() for m in CITATION_RE.findall(report)}),
            }

    bundle = services["okf_rag"].bundle
    lines: list[str] = [
        "# RAG Mode Comparison (OKF + RAG retrieval)",
        "",
        "Generated by `python scripts/compare_rag_modes.py`.",
        "",
        "For each sample prediction payload the three retrieval strategies "
        "configured through `RAG_MODE` were run and the deterministic template "
        "report generator was used (no LLM call, no network, deterministic "
        "hashing embeddings). Every clinical statement in a report carries an "
        "inline `[concept: <id>]` citation and each report ends with a `Sources` "
        "section and the not-a-diagnosis disclaimer. The `Risk Interpretation` "
        "and `Recommended Next Steps` sections appear only when a retrieved "
        "concept supports them: their sentences are taken from the body of the "
        "concept they cite, never hardcoded.",
        "",
        "## Setup",
        "",
        f"- Knowledge bundle: `{repo_relative(bundle.root)}` ({len(bundle)} concepts, "
        f"content hash `{bundle.content_hash()[:16]}...`)",
        "- Each case carries its own question (the same question a clinician "
        "would type for that patient); pass `--shared-query` to send one "
        "identical question to all cases instead.",
        f"- Concept budget per report: `RAG_MAX_CONCEPTS="
        f"{services['okf_rag'].cfg.max_concepts}` (direct hits first)",
        f"- Indexed chunks: {services['okf_rag'].store.n_total}",
        "",
        "### Modes",
        "",
    ]
    for mode in RAG_MODES:
        lines.append(f"- **`{mode}`** — {MODE_NOTES[mode]}")
    lines += ["", "## Retrieved concepts per mode", ""]

    for case in CASES:
        payload = case["payload"]
        clinical = payload["clinical_inputs"]
        lines += [
            f"### {case['name']}",
            "",
            f"Question: _{results[case['name']][RAG_MODES[0]]['query']}_",
            "",
            f"Predicted stage: **{payload['predicted_stage']}** · "
            f"MMSE {clinical['MMSE']} · CDR {clinical['CDR']} · nWBV {clinical['nWBV']} · "
            f"clinical branch: {payload['clinical_diagnosis']} · "
            f"fusion: {payload['fusion_diagnosis']}",
            "",
            "| mode | retrieved concept ids |",
            "| --- | --- |",
        ]
        for mode in RAG_MODES:
            ids = results[case["name"]][mode]["ids"]
            rendered = ", ".join(f"`{cid}`" for cid in ids) if ids else "*(none)*"
            lines.append(f"| `{mode}` | {rendered} |")
        lines.append("")

        for mode in RAG_MODES:
            entry = results[case["name"]][mode]
            lines += [
                f"<details><summary><b>{mode}</b> - hit table and report</summary>",
                "",
                *_concept_lines(entry["hits"]),
                "",
                "Cited concepts: "
                + (", ".join(f"`{c}`" for c in entry["citations"]) or "*(none)*"),
                "",
                "```markdown",
                entry["report"],
                "```",
                "",
                "</details>",
                "",
            ]

    # ---- cross-mode summary -------------------------------------------
    lines += [
        "## Summary",
        "",
        "| case | mode | concepts | direct | linked | stage-required |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in CASES:
        for mode in RAG_MODES:
            hits = results[case["name"]][mode]["hits"]
            direct = sum(1 for h in hits if origin_of(h) == ORIGIN_DIRECT)
            linked = sum(1 for h in hits if origin_of(h) == ORIGIN_LINKED)
            required = sum(1 for h in hits if origin_of(h) == ORIGIN_STAGE_REQUIRED)
            lines.append(
                f"| {case['name'].split(' - ')[0]} | `{mode}` | {len(hits)} | "
                f"{direct} | {linked} | {required} |"
            )
    lines += [
        "",
        "Observations:",
        "",
        "- Every mode finishes by admitting the concepts the predicted stage and "
        "the clinical scores require, marked `stage-required`: the AI-prediction "
        "caveats always, `treatment/early_stage_guidance` for a Very Mild / Mild "
        "stage or CDR >= 0.5, `treatment/lifestyle_risk_modification` + "
        "`risk/modifiable_risk_reduction` for No Impairment, and "
        "`treatment/non_pharmacological` + `diagnosis/red_flags_referral` for "
        "Moderate or CDR >= 2. Without them the `Risk Interpretation` and "
        "`Recommended Next Steps` sections would drop out whenever the wording of "
        "the question happened to rank those concepts below the cut.",
        "- A stage-required concept is never evicted to satisfy "
        "`RAG_MAX_CONCEPTS`: the direct-hit budget reserves room for it, and if "
        "the cap is still exceeded the lowest-scoring non-required linked "
        "concepts go first. Its score column reads _(required)_ because the "
        "number is a marker rather than a relevance score.",
        "- `rag_only` returns the vector hits for the question alone, so it "
        "changes with the wording of the question rather than with the patient.",
        "- `okf_rag` keeps those direct hits and appends concepts one OKF link "
        "away (marked `linked from ...`), which pulls in the neighbouring scale "
        "/ management concepts without displacing the direct hits; the report "
        "budget stays at `RAG_MAX_CONCEPTS`.",
        "- `okf_only` ignores embeddings entirely and follows the prediction "
        "payload (CDR band, MMSE band, nWBV, predicted stage), so its concept "
        "set tracks the clinical severity of the case and stays stable across "
        "rewordings of the question. It is also the only mode that needs no "
        "vector index at all.",
        "- All three modes produce the same report skeleton: inline "
        "`[concept: <id>]` citations, a `Sources` section listing exactly the "
        "concepts cited inline, and the not-a-diagnosis disclaimer last.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shared-query",
        action="store_true",
        help="use one identical question for every case instead of per-case questions",
    )
    args = parser.parse_args()

    cfg = offline_config(None)
    services = {mode: RagReportService(dataclasses.replace(cfg, rag_mode=mode)) for mode in RAG_MODES}
    markdown = build_markdown(services, SHARED_QUERY if args.shared_query else None)

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output} ({len(markdown.splitlines())} lines)")
    for mode, service in services.items():
        print(f"  {mode}: index chunks={service.store.n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
