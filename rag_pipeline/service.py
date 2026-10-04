"""High-level RAG report service used by the Streamlit app.

Three retrieval strategies are selectable through ``RAG_MODE``:

``okf_rag`` (default)
    FAISS vector search over the OKF concept chunks, then a one-hop expansion
    along the outgoing OKF links of the retrieved concepts.
``rag_only``
    FAISS vector search only, no link expansion.
``okf_only``
    No vectors at all: concepts are selected by matching tags / types / keywords
    derived from the prediction payload (CDR, MMSE, nWBV, stage ...), then the
    one-hop link expansion is applied.
"""
from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Any, Iterable

from .chunking import chunk_concept, load_bundle_chunks
from .config import (
    RAG_MODE_OKF_ONLY,
    RAG_MODE_RAG_ONLY,
    RAG_MODES,
    RagConfig,
    load_config,
)
from .embeddings import Embedder
from .okf_loader import Concept, OkfBundle, load_bundle
from .report_generator import generate_report
from .vector_store import (
    ORIGIN_DIRECT,
    ORIGIN_LINKED,
    ORIGIN_STAGE_REQUIRED,
    Hit,
    VectorStore,
    origin_of,
)

logger = logging.getLogger(__name__)

# Score multiplier applied to concepts reached through an OKF link, so direct
# vector hits always outrank linked concepts.
LINK_SCORE_DECAY = 0.5

# How many extra FAISS candidates to pull so concept-level dedup still has
# enough material to reach the requested number of concepts.
CANDIDATE_FACTOR = 4

# ---- stage-driven required concepts ----------------------------------------
# A report is only useful if it can say what the predicted stage implies, and
# the concepts carrying that guidance are not guaranteed to survive a query
# ranked by the wording of the question. These are admitted after retrieval in
# every RAG mode, so "Risk Interpretation" and "Recommended Next Steps" have
# evidence to quote instead of silently disappearing.
CAVEATS_CONCEPT = "diagnosis/ai_assisted_prediction_caveats"
EARLY_STAGE_CONCEPT = "treatment/early_stage_guidance"
PREVENTION_CONCEPTS = (
    "treatment/lifestyle_risk_modification",
    "risk/modifiable_risk_reduction",
)
ADVANCED_STAGE_CONCEPTS = (
    "treatment/non_pharmacological",
    "diagnosis/red_flags_referral",
)

# One slot of RAG_MAX_CONCEPTS is held back so the link-expanding modes can
# always place a graph neighbour. It is held back in every mode so that all
# modes draw the same direct-hit budget and the `rag_only` concept set stays a
# subset of the `okf_rag` one.
LINK_SLOT_RESERVE = 1

# Sentinel score for a stage-required hit: these concepts are admitted because
# the stage requires them, so the number is a marker rather than a relevance
# score (they are never dropped by the cap).
STAGE_REQUIRED_SCORE = 1.0

_QUERY_STOPWORDS = {
    "what", "does", "this", "that", "these", "those", "with", "from", "into",
    "should", "would", "could", "patient", "results", "result", "suggest",
    "suggested", "suggestion", "next", "steps", "step", "tell", "about",
    "their", "there", "here", "have", "has", "been", "were", "are", "the",
    "and", "for", "you", "your", "our", "please", "explain", "mean", "means",
}

_STAGE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("no impairment", ("cdr 0", "cognitively normal", "normal", "no dementia", "monitoring")),
    ("very mild", ("mci", "early-stage", "mild cognitive impairment", "cdr 0.5", "questionable")),
    ("mild", ("mild dementia", "cdr 1", "functional impairment")),
    ("moderate", ("moderate dementia", "cdr 2", "nmda antagonist")),
    ("severe", ("severe dementia", "cdr 3", "cholinesterase inhibitors")),
)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-\.]*", str(text).lower()) if len(w) > 2]


@lru_cache(maxsize=2048)
def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary pattern for a (possibly multi-word) term."""
    return re.compile(
        rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])".replace(r"\ ", r"\s+")
    )


def _contains(term: str, blob: str) -> bool:
    return bool(_term_pattern(term).search(blob))


def _count(term: str, blob: str) -> int:
    return len(_term_pattern(term).findall(blob))


def _mmse_band(value: float) -> str:
    if value >= 27:
        return "normal"
    if value >= 21:
        return "mild cognitive impairment"
    if value >= 11:
        return "moderate"
    return "severe"


def _cdr_band(value: float) -> str:
    mapping = {
        0.0: "cdr 0",
        0.5: "cdr 0.5",
        1.0: "cdr 1",
        2.0: "cdr 2",
        3.0: "cdr 3",
    }
    return mapping.get(float(value), f"cdr {value}")


def _clinical_score(prediction: dict[str, Any] | None, prefix: str) -> float | None:
    """Numeric clinical input whose name starts with ``prefix`` (e.g. ``CDR``)."""
    if not isinstance(prediction, dict):
        return None
    clinical = prediction.get("clinical_inputs")
    if not isinstance(clinical, dict):
        return None
    for name, raw in clinical.items():
        if not str(name).strip().lower().startswith(prefix):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


class RagReportService:
    """Builds the FAISS index over the OKF bundle and generates reports."""

    def __init__(self, cfg: RagConfig | None = None):
        self.cfg = cfg or load_config()
        if self.cfg.rag_mode not in RAG_MODES:
            raise ValueError(f"unknown RAG_MODE {self.cfg.rag_mode!r}")
        self.embedder = Embedder(self.cfg)
        self.store = VectorStore(self.cfg, self.embedder)
        self._bundle: OkfBundle | None = None
        self.index_rebuilt = False

    # ---- OKF bundle -----------------------------------------------------
    @property
    def bundle(self) -> OkfBundle:
        if self._bundle is None:
            self._bundle = load_bundle(self.cfg.knowledge_bundle_dir)
        return self._bundle

    @property
    def bundle_hash(self) -> str:
        return self.bundle.content_hash()

    def reload_bundle(self) -> OkfBundle:
        self._bundle = load_bundle(self.cfg.knowledge_bundle_dir)
        return self._bundle

    # ---- index lifecycle ------------------------------------------------
    def ensure_index(self) -> int:
        """Load the persisted index, rebuilding it when it is stale.

        The index is stale when the knowledge bundle content changed or when
        the embedding mode / model changed (both are part of the signature
        stored next to the FAISS index).
        """
        signature = self.cfg.index_signature(self.bundle_hash)
        if self.store.n_total > 0 and self.store.signature == signature:
            self.index_rebuilt = False
            return self.store.n_total
        loaded = 0
        if self.store.n_total == 0:
            try:
                self.store = VectorStore.load(self.cfg, self.embedder)
                loaded = self.store.n_total
            except Exception:  # pragma: no cover - corrupted index file
                self.store = VectorStore(self.cfg, self.embedder)
                loaded = 0
        if loaded and self.store.signature == signature:
            self.index_rebuilt = False
            return loaded
        if loaded:
            logger.info(
                "rag: persisted index is stale (bundle or embedding mode changed) - rebuilding"
            )
        return self.build_index()

    def build_index(self) -> int:
        """Force a rebuild of the FAISS index from the OKF bundle."""
        chunks = load_bundle_chunks(self.cfg, self.bundle)
        if not chunks:
            return 0
        signature = self.cfg.index_signature(self.bundle_hash)
        self.store.build(chunks, signature=signature)
        self.index_rebuilt = True
        try:
            self.store.save()
        except Exception:
            logger.warning("rag: could not persist the FAISS index", exc_info=True)
        return self.store.n_total

    def load_index(self) -> int:
        """Load a previously saved index. Returns chunk count or 0."""
        self.store = VectorStore.load(self.cfg, self.embedder)
        return self.store.n_total

    # ---- retrieval ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        prediction: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Retrieve chunks for a query honouring the configured RAG_MODE.

        Every mode finishes with the same step: the concepts the predicted stage
        and the clinical scores require are admitted, so the report has evidence
        for its risk and next-step sections whatever the question retrieved.
        """
        try:
            required = self._required_concept_ids(prediction)
            if self.cfg.rag_mode == RAG_MODE_OKF_ONLY:
                direct = self._select_concepts_from_payload(
                    query, prediction, top_k, reserve=len(required)
                )
            else:
                direct = self._vector_concepts(query, top_k, reserve=len(required))
            found = {hit.concept_id for hit in direct}
            missing = [cid for cid in required if cid not in found]
            if self.cfg.rag_mode == RAG_MODE_RAG_ONLY:
                hits = list(direct)
            else:
                hits = self._expand_links(direct, reserve=len(missing))
            hits.extend(self._stage_required_hits(missing))
            return self._enforce_cap(hits, set(required))
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, IndexError, TypeError):
            logger.warning("rag: retrieval failed", exc_info=True)
            return []

    def _required_concept_ids(self, prediction: dict[str, Any] | None) -> tuple[str, ...]:
        """Concept ids the predicted stage and the clinical scores require.

        Ordered by priority (the caveat concept first, since every report needs
        it) and deduplicated, so a concept required by two conditions is only
        reserved once.
        """
        ordered: list[str] = [CAVEATS_CONCEPT]
        stage = ""
        cdr: float | None = None
        if isinstance(prediction, dict):
            stage = str(
                prediction.get("predicted_stage", prediction.get("stage", ""))
            ).lower()
            cdr = _clinical_score(prediction, "cdr")

        if "mild" in stage or (cdr is not None and cdr >= 0.5):
            ordered.append(EARLY_STAGE_CONCEPT)
        if "no impairment" in stage:
            ordered.extend(PREVENTION_CONCEPTS)
        if "moderate" in stage or (cdr is not None and cdr >= 2.0):
            ordered.extend(ADVANCED_STAGE_CONCEPTS)

        seen: set[str] = set()
        ids: list[str] = []
        for concept_id in ordered:
            if concept_id in seen or self.bundle.get(concept_id) is None:
                continue
            seen.add(concept_id)
            ids.append(concept_id)
        if len(ids) > self.cfg.max_concepts:
            logger.warning(
                "rag: %d stage-required concepts exceed RAG_MAX_CONCEPTS=%d, "
                "keeping the highest-priority ones",
                len(ids),
                self.cfg.max_concepts,
            )
            ids = ids[: self.cfg.max_concepts]
        return tuple(ids)

    def _stage_required_hits(self, concept_ids: Iterable[str]) -> list[Hit]:
        """Build the hits for stage-required concepts that exist in the bundle.

        These carry the ``stage-required`` origin and a sentinel score: the
        stage demands them, so they are never dropped to satisfy the cap.
        """
        hits: list[Hit] = []
        for concept_id in concept_ids:
            hit = self._hit_for_concept(
                concept_id,
                STAGE_REQUIRED_SCORE,
                direct=True,
                origin=ORIGIN_STAGE_REQUIRED,
            )
            if hit is None:
                logger.info("rag: cannot admit stage-required concept %r", concept_id)
                continue
            hits.append(hit)
        return hits

    def _enforce_cap(self, hits: list[Hit], required: set[str]) -> list[Hit]:
        """Trim ``hits`` to ``RAG_MAX_CONCEPTS``, keeping the required concepts.

        The lowest-scoring non-required linked concepts are dropped first, then
        the lowest-scoring non-required direct ones, and only a required concept
        is given up when the cap is smaller than the stage requires.
        """
        cap = self.cfg.max_concepts
        if len(hits) <= cap:
            return hits
        # lowest score first, insertion order breaks ties deterministically
        ranked = sorted(enumerate(hits), key=lambda item: (-item[1].score, item[0]))
        keep = {index for index, _ in ranked}
        overage = len(hits) - cap
        tiers = (
            lambda hit: hit.concept_id not in required
            and origin_of(hit) == ORIGIN_LINKED,
            lambda hit: hit.concept_id not in required,
            lambda hit: True,
        )
        for tier in tiers:
            for index, hit in ranked:
                if overage <= 0:
                    break
                if index in keep and tier(hit):
                    keep.discard(index)
                    overage -= 1
        if overage > 0:  # pragma: no cover - only when the cap is sub-required
            logger.warning(
                "rag: RAG_MAX_CONCEPTS=%d is smaller than the %d concepts this "
                "stage requires; some required concepts were dropped",
                cap,
                len(required),
            )
        return [hit for index, hit in enumerate(hits) if index in keep]

    def _direct_limit(self, top_k: int | None, *, reserve: int = 0) -> int:
        """How many direct hits to keep, after reserving room for the rest.

        ``reserve`` counts the stage-required concepts; one further slot is
        always held back so the link-expanding modes can still place a graph
        neighbour within the same cap.
        """
        requested = self.cfg.top_k if top_k is None else top_k
        requested = max(1, requested)
        budget = self.cfg.max_concepts - max(0, reserve) - LINK_SLOT_RESERVE
        return max(1, min(requested, budget))

    def _vector_concepts(
        self, query: str, top_k: int | None, *, reserve: int = 0
    ) -> list[Hit]:
        """FAISS search collapsed to one best chunk per concept."""
        if self.ensure_index() <= 0:
            return []
        limit = self._direct_limit(top_k, reserve=reserve)
        candidates = max(self.cfg.top_k, limit) * CANDIDATE_FACTOR
        hits = self.store.search(query, top_k=candidates)
        best: dict[str, Hit] = {}
        for hit in hits:
            if not hit.concept_id:
                continue
            current = best.get(hit.concept_id)
            if current is None or hit.score > current.score:
                best[hit.concept_id] = hit
        ranked = sorted(best.values(), key=lambda h: (-h.score, h.concept_id))
        return ranked[:limit]

    def _select_concepts_from_payload(
        self,
        query: str,
        prediction: dict[str, Any] | None,
        top_k: int | None,
        *,
        reserve: int = 0,
    ) -> list[Hit]:
        """``okf_only`` mode: rank concepts by payload tags / types / keywords."""
        bundle = self.bundle
        if not len(bundle):
            return []
        payload_terms = self._payload_terms(prediction)
        query_terms = {
            term: 0.4 for term in _words(query) if term not in _QUERY_STOPWORDS
        }
        scored: list[tuple[float, str]] = []
        for concept in bundle:
            score = self._lexical_score(concept, payload_terms, query_terms)
            if score > 0:
                scored.append((score, concept.concept_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = self._direct_limit(top_k, reserve=reserve)
        hits: list[Hit] = []
        for raw_score, concept_id in scored[:limit]:
            # squash the lexical score into the 0..1 range used by the vector
            # hits so both modes produce comparable confidence numbers
            score = raw_score / (raw_score + 5.0)
            hit = self._hit_for_concept(concept_id, score, direct=True)
            if hit is not None:
                hits.append(hit)
        return hits

    def _expand_links(self, direct: list[Hit], *, reserve: int = 0) -> list[Hit]:
        """Add concepts one hop along the outgoing OKF links of the hits."""
        cap = max(0, self.cfg.max_concepts - max(0, reserve))
        hits = list(direct)
        seen = {h.concept_id for h in hits if h.concept_id}
        for parent in direct:
            if len(hits) >= cap:
                break
            parent_id = parent.concept_id
            if not parent_id:
                continue
            for link_id in self.bundle.outgoing(parent_id):
                if len(hits) >= cap:
                    break
                if link_id in seen:
                    continue
                seen.add(link_id)
                score = max(0.0, float(parent.score)) * LINK_SCORE_DECAY
                hit = self._hit_for_concept(
                    link_id,
                    score,
                    direct=False,
                    link_source=parent_id,
                    origin=ORIGIN_LINKED,
                )
                if hit is not None:
                    hits.append(hit)
        return hits

    def _hit_for_concept(
        self,
        concept_id: str,
        score: float,
        *,
        direct: bool,
        link_source: str = "",
        origin: str = ORIGIN_DIRECT,
    ) -> Hit | None:
        concept = self.bundle.get(concept_id)
        if concept is None:
            return None
        text = ""
        for meta in self.store.chunks_for_concept(concept_id):
            text = meta.get("text", "")
            if text:
                break
        if not text:
            chunks = chunk_concept(concept, self.cfg.chunk_size, self.cfg.chunk_overlap)
            text = chunks[0].text if chunks else concept.description or concept.title
        return Hit(
            text=text,
            source=concept.source,
            title=concept.title,
            score=float(score),
            concept_id=concept.concept_id,
            concept_type=concept.type,
            direct=direct,
            link_source=link_source,
            origin=origin,
        )

    # ---- okf_only keyword matching --------------------------------------
    def _payload_terms(self, prediction: dict[str, Any] | None) -> dict[str, float]:
        """Weighted keywords extracted from the prediction payload."""
        terms: dict[str, float] = {}

        def add(term: Any, weight: float = 1.0) -> None:
            for token in _words(str(term)):
                terms[token] = max(terms.get(token, 0.0), weight)

        if not isinstance(prediction, dict):
            return terms

        stage = str(prediction.get("predicted_stage", prediction.get("stage", "")))
        add(stage, 1.0)
        for needle, keywords in _STAGE_KEYWORDS:
            if needle in stage.lower():
                # the predicted stage is the strongest payload signal
                for kw in keywords:
                    add(kw, 1.4)
                break

        clinical = prediction.get("clinical_inputs")
        if isinstance(clinical, dict):
            for name, raw in clinical.items():
                key = str(name).lower()
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    add(key, 1.0)
                    continue
                if key.startswith("cdr"):
                    add(_cdr_band(value), 1.0)
                    add("cdr", 0.8)
                elif key.startswith("mmse"):
                    add("mmse", 1.0)
                    add(_mmse_band(value), 0.8)
                elif key.startswith("nwbv"):
                    add("nwbv", 1.0)
                    add("whole brain volume", 0.35)
                    add("atrophy", 0.8 if value < 0.73 else 0.4)
                    add("ventricular", 0.4 if value < 0.73 else 0.2)
                else:
                    add(key, 0.6)

        for field_name in (
            "clinical_diagnosis",
            "fusion_diagnosis",
            "mri_stage_implies_dementia",
            "needs_clinical_review",
            "mri_clinical_conflict",
        ):
            value = prediction.get(field_name)
            if isinstance(value, str):
                add(value, 0.7)
            elif value:
                add(field_name, 0.5)

        probabilities = prediction.get("probabilities")
        if isinstance(probabilities, dict):
            for name, _value in probabilities.items():
                add(name, 0.5)
        return terms

    @staticmethod
    def _lexical_score(
        concept: Concept,
        payload_terms: dict[str, float],
        query_terms: dict[str, float],
    ) -> float:
        tags = {tag.lower() for tag in concept.tags}
        type_slug = re.sub(r"[^a-z0-9]+", "-", concept.type.lower()).strip("-")
        title_blob = concept.title.lower()
        desc_blob = concept.description.lower()
        body_blob = concept.body.lower()
        score = 0.0
        for term, weight in list(payload_terms.items()) + list(query_terms.items()):
            matched = False
            if term in tags or (len(term) > 3 and _contains(term, " ".join(tags))):
                score += 3.0 * weight
                matched = True
            if term == type_slug or (len(term) > 3 and _contains(term, type_slug.replace("-", " "))):
                score += 2.0 * weight
                matched = True
            if _contains(term, title_blob):
                score += 1.5 * weight
                matched = True
            if _contains(term, desc_blob):
                score += 0.5 * weight
                matched = True
            if not matched and len(term) > 3:
                hits_count = min(_count(term, body_blob), 8)
                if hits_count:
                    score += 0.25 * weight * hits_count
        return score

    # ---- reporting ------------------------------------------------------
    def concepts(self, hits: Iterable[Hit]) -> list[dict[str, Any]]:
        """Deduplicated concept view of the hits, ordered as retrieved."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            key = hit.concept_id or hit.source
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "concept_id": hit.concept_id or hit.source,
                    "title": hit.title,
                    "type": hit.concept_type,
                    "score": round(float(hit.score), 4),
                    "direct": hit.direct,
                    "link_source": hit.link_source,
                    "origin": origin_of(hit),
                }
            )
        return out

    def generate(
        self,
        prediction: dict[str, Any],
        query: str,
        *,
        allow_llm: bool = True,
    ) -> dict[str, Any]:
        """Retrieve context and generate the markdown report.

        Returns ``{"report": str, "used_llm": bool, "hits": list[Hit],
        "concepts": list[dict], "index_size": int, "rag_mode": str,
        "embedding_provider": str, "llm_blocked": bool}``.

        ``allow_llm=False`` skips the LLM entirely and returns the deterministic
        template report. It is how a metered deployment enforces a per-session
        call budget without the caller having to know whether a key exists.
        """
        hits = self.retrieve(query, prediction=prediction)
        report, used_llm = generate_report(
            self.cfg, prediction, query, hits, allow_llm=allow_llm
        )
        return {
            "report": report,
            "used_llm": used_llm,
            "llm_blocked": bool(allow_llm is False),
            "hits": hits,
            "concepts": self.concepts(hits),
            "index_size": self.store.n_total,
            "rag_mode": self.cfg.rag_mode,
            "embedding_provider": self.embedder.provider,
        }