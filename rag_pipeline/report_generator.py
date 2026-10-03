"""Generate the markdown clinical report (LLM with a template fallback)."""
from __future__ import annotations

import json
import math
from typing import Any

from .config import RagConfig
from .vector_store import Hit

_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for Alzheimer's disease. "
    "You generate a structured markdown report from an automated model prediction, "
    "clinical scores, and retrieved medical knowledge. Be conservative, accurate, "
    "and clear. Do not invent numbers. Follow the report structure described in the "
    "knowledge base document 'report_template.md'. End with a disclaimer that this "
    "is decision support, not a clinical diagnosis."
)


def _prediction_block(pred: dict[str, Any]) -> str:
    return json.dumps(pred, indent=2, ensure_ascii=False)


def _retrieval_block(hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] Source: {h.source} | Title: {h.title} | score={h.score:.3f}\n{h.text}")
    return "\n\n".join(lines)


def _probability_lines(probs: dict[str, Any]) -> str:
    lines = []
    for name, value in probs.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        if 1.0 < number <= 100.0:
            number /= 100.0
        if 0.0 <= number <= 1.0:
            lines.append(f"- **{name}:** {number * 100:.1f}%")
    return "\n".join(lines) or "- *no valid probability breakdown available*"


def _next_steps(stage: str) -> list[str]:
    normalized = str(stage).lower()
    if "no impairment" in normalized:
        return [
            "Repeat cognitive screening at the interval recommended by a clinician.",
            "Review vascular risk factors and preventive care.",
            "Monitor for changes in memory, language, or daily functioning.",
        ]
    if "moderate" in normalized:
        return [
            "Arrange prompt specialist assessment for cognitive and functional decline.",
            "Review medication options and care goals with the treating clinician.",
            "Assess safety, caregiver support, and need for assistance with daily activities.",
        ]
    if "mild" in normalized or "very mild" in normalized:
        return [
            "Arrange specialist cognitive and functional assessment.",
            "Review vascular risk factors, sleep, mood, and medication contributors.",
            "Establish a monitoring plan and discuss caregiver support where appropriate.",
        ]
    return [
        "Formal cognitive assessment with a specialist.",
        "Vascular risk-factor optimization.",
        "Monitoring of symptom progression and caregiver support.",
    ]


def _template_report(pred: dict[str, Any], query: str, hits: list[Hit]) -> str:
    """Deterministic markdown report used when no LLM is available."""
    stage = pred.get("predicted_stage", pred.get("stage", "Unknown"))
    probs = pred.get("probabilities") or pred.get("probs") or {}
    if not isinstance(probs, dict):
        probs = {}
    probs_lines = _probability_lines(probs)

    head = ["# Automated Clinical Decision-Support Report", ""]
    head.append(f"**Predicted cognitive stage: {stage}**")
    head.append(f"**Query:** {query}")
    head += ["", "## Clinical Summary", ""]
    head.append(
        f"Based on the automated analysis, the predicted cognitive stage is "
        f"**{stage}**. Interpret this in the full clinical context."
    )
    head += ["", "## Imaging (MRI) Analysis", "", probs_lines, ""]
    head.append("## Clinical Scores Interpretation")
    if pred.get("clinical_inputs"):
        head.append("")
        head.append("| Score | Value |")
        head.append("| --- | --- |")
        for k, v in pred["clinical_inputs"].items():
            head.append(f"| {k} | {v} |")
    head += ["", "## Risk Interpretation", ""]
    head.append(
        "The model output is a screening signal rather than a diagnosis. Risk cannot "
        "be established from this output alone; interpret it with clinical history, "
        "functional examination, validated cognitive testing, and clinician review."
    )
    head += ["", "## Recommended Next Steps", ""]
    head.extend(f"- {step}" for step in _next_steps(stage))
    head += ["", "## References / Knowledge Sources", ""]
    if hits:
        head += [f"- `{h.source}` — {h.title}" for h in hits]
    else:
        head.append("- *(no knowledge chunks retrieved)*")
    head += ["", "## Disclaimer", ""]
    head.append(
        "This report is generated automatically as decision-support output. It is "
        "not a clinical diagnosis. A qualified clinician must confirm any findings."
    )
    return "\n".join(head)


def generate_report(
    cfg: RagConfig,
    prediction: dict[str, Any],
    query: str,
    hits: list[Hit],
) -> tuple[str, bool]:
    """Return (report_markdown, used_llm).

    Uses the OpenAI-compatible chat endpoint when a key is configured and the
    call succeeds; otherwise returns a template-based report.
    """
    if cfg.has_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            user_msg = (
                "Patient query:\n"
                f"{query}\n\n"
                "Automated prediction:\n"
                f"{_prediction_block(prediction)}\n\n"
                "Retrieved knowledge:\n"
                f"{_retrieval_block(hits)}"
            )
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
                return content.strip(), True
        except Exception:
            pass
    return _template_report(prediction, query, hits), False
