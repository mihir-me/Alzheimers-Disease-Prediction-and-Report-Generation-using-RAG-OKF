"""Loader for the OKF (Open Knowledge Format) markdown bundle.

The bundle lives in ``knowledge/okf_bundle`` and contains one markdown file per
knowledge concept::

    knowledge/okf_bundle/
        index.md                     <- bundle index, never a concept
        scores/cdr_scale.md          <- concept_id "scores/cdr_scale"
        scores/mmse.md
        ...

Every non-``index.md`` file becomes a :class:`Concept`. Each concept carries its
front-matter metadata (``type``, optional ``tags`` / ``description`` /
``sources``), its body text, and the outgoing links resolved against the bundle
so that the retrieval layer can expand along the knowledge graph.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

# Files that live inside the bundle for navigation only, never concepts.
RESERVED_FILENAMES = {"index.md", "log.md"}

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_LINK_RE = re.compile(r"(?<!\!)\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_H1_RE = re.compile(r"(?m)^#[ \t]+(.+?)[ \t]*$")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:")

# Keyword -> tag map used to derive tags when a concept has no explicit ones.
# These tags are what the "okf_only" mode matches against the prediction
# payload (CDR / MMSE / nWBV / stage ...).
_KEYWORD_TAGS: dict[str, tuple[str, ...]] = {
    "cdr": ("cdr", "staging", "clinical-score"),
    "clinical dementia rating": ("cdr", "staging", "clinical-score"),
    "mmse": ("mmse", "cognitive-testing", "clinical-score"),
    "mini-mental": ("mmse", "cognitive-testing", "clinical-score"),
    "nwbv": ("nwbv", "atrophy", "clinical-score"),
    "whole brain volume": ("nwbv", "atrophy"),
    "mci": ("mci", "early-stage"),
    "mild cognitive impairment": ("mci", "early-stage"),
    "early-stage": ("early-stage",),
    "early stage": ("early-stage",),
    "stage": ("staging",),
    "staging": ("staging",),
    "atrophy": ("atrophy", "imaging"),
    "hippocampus": ("atrophy", "mtl", "imaging"),
    "entorhinal": ("atrophy", "mtl", "imaging"),
    "medial temporal": ("mtl", "imaging"),
    "ventricul": ("atrophy", "imaging"),
    "mri": ("mri", "imaging"),
    "imaging": ("imaging",),
    "biomarker": ("biomarker", "imaging"),
    "amyloid": ("amyloid", "biomarker"),
    "referral": ("referral", "red-flags"),
    "red flag": ("red-flags", "referral"),
    "differential": ("diagnosis", "differential"),
    "nia-aa": ("diagnosis", "nia-aa"),
    "biomarker-defined": ("diagnosis",),
    "preclinical": ("preclinical", "progression"),
    "progression": ("progression",),
    "risk factor": ("risk-factors",),
    "age": ("risk-factors", "age"),
    "hypertension": ("risk-factors", "vascular"),
    "diabetes": ("risk-factors", "vascular"),
    "cholinesterase": ("treatment", "pharmacotherapy"),
    "donepezil": ("treatment", "pharmacotherapy"),
    "rivastigmine": ("treatment", "pharmacotherapy"),
    "galantamine": ("treatment", "pharmacotherapy"),
    "memantine": ("treatment", "pharmacotherapy"),
    "nmda": ("treatment", "pharmacotherapy"),
    "lecanemab": ("treatment", "disease-modifying"),
    "antibody": ("treatment", "disease-modifying"),
    "lifestyle": ("lifestyle", "risk-modification"),
    "exercise": ("lifestyle", "risk-modification"),
    "cognitive stimulation": ("lifestyle", "non-pharmacological"),
    "report": ("reporting",),
    "template": ("reporting",),
    "citation": ("reporting",),
}


def _as_list(value: object) -> list[str]:
    """Normalize a front-matter value (str / list / None) into a string list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip().strip("'\"").strip() for p in text.split(",")]
    return [p for p in parts if p]


def _parse_frontmatter(raw: str) -> dict[str, object]:
    """Parse the small YAML subset used by the bundle front matter.

    Supports ``key: scalar``, inline lists (``[a, b]``) and block lists
    (``- item``). Nested mappings (such as ``generated: { by: ..., at: ... }``)
    are intentionally ignored: no loader field needs them.
    """
    data: dict[str, object] = {}
    key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and key is not None:
            item = stripped[2:].strip().strip("'\"")
            if item:
                current = _as_list(data.get(key))
                data[key] = current + [item]
            continue
        if ":" not in stripped:
            continue
        name, _, value = stripped.partition(":")
        name = name.strip().lower()
        value = value.strip()
        if not name:
            continue
        if value == "":
            data[name] = []
            key = name
            continue
        key = None
        if value.startswith("{") or value.startswith("["):
            inner = value.strip("{}[]").strip()
            data[name] = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
        else:
            data[name] = value.strip("'\"").strip()
    return data


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


@dataclass
class Concept:
    """One knowledge concept from the OKF bundle."""

    concept_id: str
    title: str
    type: str
    body: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def category(self) -> str:
        """Top-level bundle folder of the concept (``scores``, ``imaging``...)."""
        return self.concept_id.split("/", 1)[0] if "/" in self.concept_id else ""

    @property
    def source(self) -> str:
        """Bundle-relative markdown path, used as the chunk ``source``."""
        return f"{self.concept_id}.md"

    @property
    def keyword_blob(self) -> str:
        """Lowercased text used for lexical matching in ``okf_only`` mode."""
        parts = [self.concept_id.replace("/", " "), self.title, self.type]
        parts.extend(self.tags)
        parts.append(self.description)
        parts.append(self.body)
        return "\n".join(parts).lower()


class OkfBundle:
    """In-memory view of an OKF bundle: concepts plus the outgoing link graph."""

    def __init__(self, root: Path, concepts: dict[str, Concept]):
        self.root = Path(root)
        self.concepts = concepts

    # ---- lookup helpers -------------------------------------------------
    def __len__(self) -> int:
        return len(self.concepts)

    def __contains__(self, concept_id: object) -> bool:
        return concept_id in self.concepts

    def __iter__(self) -> Iterator[Concept]:
        for cid in sorted(self.concepts):
            yield self.concepts[cid]

    def get(self, concept_id: str) -> Concept | None:
        return self.concepts.get(concept_id)

    def by_id(self) -> list[str]:
        return sorted(self.concepts)

    def outgoing(self, concept_id: str) -> list[str]:
        """Resolved outgoing link targets of a concept (unknown ids dropped)."""
        concept = self.concepts.get(concept_id)
        if concept is None:
            return []
        return [link for link in concept.links if link in self.concepts]

    def linked_titles(self, concept_id: str) -> dict[str, str]:
        """Mapping of outgoing link id -> target title."""
        return {link: self.concepts[link].title for link in self.outgoing(concept_id)}

    # ---- integrity ------------------------------------------------------
    def unresolved_links(self) -> list[tuple[str, str]]:
        """(concept_id, link) pairs whose target is not a known concept."""
        out: list[tuple[str, str]] = []
        for cid in sorted(self.concepts):
            for link in self.concepts[cid].links:
                if link not in self.concepts:
                    out.append((cid, link))
        return out

    def content_hash(self) -> str:
        """Stable sha256 over every markdown file in the bundle."""
        digest = hashlib.sha256()
        for path in sorted(self.markdown_files(), key=lambda p: str(p).lower()):
            rel = path.relative_to(self.root).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update(path.read_bytes())
            except OSError:  # pragma: no cover - unreadable file
                digest.update(b"<unreadable>")
            digest.update(b"\0")
        return digest.hexdigest()

    def markdown_files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return [p for p in self.root.rglob("*.md") if p.is_file()]


def _split_body(raw_text: str) -> tuple[dict[str, object], str]:
    """Return (front matter, body) for one markdown document."""
    match = _FRONTMATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text
    return _parse_frontmatter(match.group(1)), raw_text[match.end():]


def _resolve_link(raw_url: str, concept_path: Path, root: Path) -> str | None:
    """Resolve one markdown link to a bundle-relative concept id."""
    url = raw_url.strip()
    if not url or url.startswith("#") or url.lower().startswith(_EXTERNAL_PREFIXES):
        return None
    url = url.split("#", 1)[0].split("?", 1)[0]
    if not url:
        return None
    if url.startswith("/"):
        # Bundle-absolute reference, e.g. /scores/mmse.md
        candidate = root / url.lstrip("/")
    else:
        candidate = concept_path.parent / url
    try:
        resolved = candidate.resolve()
        rel = resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    rel_posix = rel.as_posix()
    if not rel_posix.endswith(".md"):
        return None
    concept_id = rel_posix[: -len(".md")]
    if concept_id.endswith("/index") or concept_id == "index":
        return None
    return concept_id


def _strip_link_lines(text: str) -> str:
    """Drop trailing "Related: [a](b), [c](d)" navigation lines."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("Related:", "See also", "Also see")):
            continue
        without_links = _LINK_RE.sub("", stripped)
        if stripped and not without_links.strip().strip(".,;:-"):
            continue
        kept.append(without_links)
    return "\n".join(kept)


def _derive_tags(concept_id: str, ctype: str, title: str, body: str) -> list[str]:
    """Tags from the concept id + category + type + keywords in title/body."""
    tags: list[str] = []
    if "/" in concept_id:
        tags.append(concept_id.split("/", 1)[0])
    # tokens of the concept id itself are strong identity signals
    # ("treatment/nmda_antagonist" -> nmda, antagonist)
    tags.extend(part for part in re.split(r"[/_-]+", concept_id) if len(part) > 2)
    if ctype:
        tags.append(_slug(ctype))
    blob = f"{title}\n{_strip_link_lines(body)}".lower()
    for needle, produced in _KEYWORD_TAGS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", blob):
            tags.extend(produced)
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        tag = _slug(tag)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _first_paragraph(body: str) -> str:
    """First prose paragraph of a body, used as a fallback description."""
    for block in body.split("\n\n"):
        candidate = block.strip()
        if not candidate or candidate.startswith(("#", "-", "*", ">", "|")):
            continue
        # skip list/table lines inside the block
        lines = [ln for ln in candidate.splitlines() if ln.strip()]
        if not lines:
            continue
        text = " ".join(ln.strip() for ln in lines).strip()
        if text:
            return text[:400]
    return ""


def parse_concept(path: Path, root: Path, known_ids: Iterable[str] | None = None) -> Concept | None:
    """Parse a single bundle markdown file into a :class:`Concept`."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.warning("okf_loader: cannot read %s", path)
        return None
    rel = path.relative_to(root).as_posix()
    if rel.endswith("/index.md") or rel == "index.md":
        return None
    front, body = _split_body(raw)

    concept_id = rel[: -len(".md")] if rel.endswith(".md") else rel
    heading = _H1_RE.search(body)
    title = str(front.get("title") or "").strip()
    if not title and heading:
        title = heading.group(1).strip()
    if not title:
        title = concept_id.replace("/", " ").replace("_", " ").strip().title()

    ctype = str(front.get("type") or "").strip() or "Concept"
    description = str(front.get("description") or "").strip() or _first_paragraph(body)
    tags = _as_list(front.get("tags"))
    if not tags:
        tags = _derive_tags(concept_id, ctype, title, body)
    sources = _as_list(front.get("sources"))

    links: list[str] = []
    seen: set[str] = set()
    for _, url in _LINK_RE.findall(body):
        target = _resolve_link(url, path, root)
        if target is None or target == concept_id or target in seen:
            continue
        seen.add(target)
        links.append(target)

    if known_ids is not None:
        known = set(known_ids)
        for link in links:
            if link not in known:
                logger.warning(
                    "okf_loader: %s links to unknown concept %r", concept_id, link
                )

    return Concept(
        concept_id=concept_id,
        title=title,
        type=ctype,
        body=body.strip(),
        description=description,
        tags=tags,
        links=links,
        sources=sources,
        path=rel,
    )


def load_bundle(root: Path | str) -> OkfBundle:
    """Load every non-index markdown file under ``root`` into an OkfBundle."""
    root = Path(root)
    concepts: dict[str, Concept] = {}
    if not root.is_dir():
        logger.warning("okf_loader: bundle directory %s does not exist", root)
        return OkfBundle(root, concepts)

    files = [
        p
        for p in root.rglob("*.md")
        if p.is_file() and p.name.lower() not in RESERVED_FILENAMES
    ]
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        concept = parse_concept(path, root)
        if concept is None:
            continue
        concepts[concept.concept_id] = concept

    bundle = OkfBundle(root, concepts)
    for concept_id, link in bundle.unresolved_links():
        logger.warning(
            "okf_loader: unresolved link %r in concept %r", link, concept_id
        )
    return bundle


def bundle_hash(root: Path | str) -> str:
    """sha256 over the bundle files, without parsing them."""
    return load_bundle(root).content_hash()