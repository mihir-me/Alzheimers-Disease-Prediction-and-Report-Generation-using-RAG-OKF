"""Checkpoint and artifact loader for CPU-only deployments.

Local checkpoints always win: whatever is already present under ``models/`` is
used untouched, so training runs and local development behave exactly as before.
A file that is missing -- the normal situation on a fresh Hugging Face Space,
where the app repository has no room for 1.1 GB of weights -- is downloaded once
from the Hugging Face Hub *model* repository named by ``HF_MODEL_REPO`` into that
same ``models/`` directory. Every consumer therefore keeps resolving plain
relative paths and no code downstream of this module needs to know where the
bytes came from.

The download is per file, not a repository snapshot: a Space that only needs the
clinical branch still fetches the clinical branch, and an interrupted start
resumes at the file that is still missing instead of re-fetching 1.1 GB.

Environment variables
---------------------
``HF_MODEL_REPO``
    Hub repository id, e.g. ``alice/alz-mri-weights``. Required only when an
    artifact is missing locally.
``HF_MODEL_REVISION``
    Branch, tag or commit to fetch. Defaults to ``main``.
``HF_TOKEN``
    Hugging Face token, needed only for private repositories. Public
    repositories work without it.
``MODELS_DIR``
    Overrides the ``models/`` directory.
``HF_MODEL_FETCH_OPTIONAL``
    When truthy, also fetch the ``*_info.json`` provenance files. They are not
    read at inference time, so they are skipped by default.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIRNAME = "models"

#: Floor used to recognise a usable file. Well below the real checkpoints, but
#: high enough to reject a truncated download or an HTML error page saved under
#: a ``.pth`` name.
MIN_CHECKPOINT_BYTES = 1 << 20
MIN_SMALL_FILE_BYTES = 1


@dataclass(frozen=True)
class Artifact:
    """One file the app needs in order to predict."""

    name: str
    kind: str
    min_bytes: int = MIN_SMALL_FILE_BYTES
    required: bool = True

    def usable(self, path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size >= self.min_bytes
        except OSError:  # pragma: no cover - unreadable path
            return False


#: The four MRI backbones. ``torchvision``/``timm`` build the architecture, so a
#: checkpoint is only the ``state_dict``.
MRI_CHECKPOINTS: tuple[Artifact, ...] = (
    Artifact("resnet152_best.pth", "checkpoint", MIN_CHECKPOINT_BYTES),
    Artifact("vgg16_best.pth", "checkpoint", MIN_CHECKPOINT_BYTES),
    Artifact("efficientnet_b4_best.pth", "checkpoint", MIN_CHECKPOINT_BYTES),
    Artifact("vit_best.pth", "checkpoint", MIN_CHECKPOINT_BYTES),
)

#: Small torch heads and the scikit-learn meta-learner / scalers, which are
#: joblib pickles rather than ``torch.save`` files.
HEADS_AND_PICKLES: tuple[Artifact, ...] = (
    Artifact("ensemble_meta_learner.pkl", "pickle"),
    Artifact("clinical_branch_best.pth", "checkpoint"),
    Artifact("fusion_model.pth", "checkpoint"),
    Artifact("fusion_cdr_scaler.pkl", "pickle"),
    Artifact("fusion_nwbv_scaler.pkl", "pickle"),
)

#: Training provenance written by ``train_*.py``. Not read at inference time;
#: upload them to the Hub repo so the weights are traceable, and fetch them only
#: on request.
INFO_FILES: tuple[Artifact, ...] = tuple(
    Artifact(
        name,
        "info",
        required=False,
    )
    for name in (
        "resnet152_info.json",
        "vgg16_info.json",
        "efficientnet_b4_info.json",
        "vit_info.json",
        "clinical_branch_info.json",
        "fusion_info.json",
        "ensemble_info.json",
    )
)

REQUIRED_ARTIFACTS: tuple[Artifact, ...] = MRI_CHECKPOINTS + HEADS_AND_PICKLES
ALL_ARTIFACTS: tuple[Artifact, ...] = REQUIRED_ARTIFACTS + INFO_FILES

_TRUTHY = {"1", "true", "yes", "on"}


def models_dir(base_dir: Path | str | None = None) -> Path:
    """Directory the checkpoints live in, honouring ``MODELS_DIR``."""
    override = os.getenv("MODELS_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else Path(base_dir or BASE_DIR) / path
    return Path(base_dir or BASE_DIR) / DEFAULT_MODELS_DIRNAME


def repo_id() -> str:
    """Hub repository the missing artifacts are fetched from (``""`` if unset)."""
    return os.getenv("HF_MODEL_REPO", "").strip()


def revision() -> str:
    """Hub branch / tag / commit to fetch."""
    return os.getenv("HF_MODEL_REVISION", "").strip() or "main"


def fetch_optional() -> bool:
    """Whether the ``*_info.json`` provenance files should be fetched too."""
    return os.getenv("HF_MODEL_FETCH_OPTIONAL", "").strip().lower() in _TRUTHY


def missing_artifacts(
    directory: Path | str | None = None,
    artifacts: Iterable[Artifact] | None = None,
) -> list[Artifact]:
    """Artifacts that are absent or too small under ``directory``."""
    target = Path(directory) if directory is not None else models_dir()
    wanted = REQUIRED_ARTIFACTS if artifacts is None else tuple(artifacts)
    return [a for a in wanted if not a.usable(target / a.name)]


def describe(directory: Path | str | None = None) -> str:
    """One-line-per-artifact inventory, for logs and the benchmark script."""
    target = Path(directory) if directory is not None else models_dir()
    lines = [f"artifacts in {target}:"]
    for artifact in ALL_ARTIFACTS:
        path = target / artifact.name
        if artifact.usable(path):
            size = path.stat().st_size
            mark = "ok     "
        else:
            size = 0
            mark = "MISSING"
        required = "required" if artifact.required else "optional"
        lines.append(
            f"  [{mark}] {artifact.name:28s} {required:8s} {size:>13,d} bytes"
        )
    return "\n".join(lines)


def _download(artifact: Artifact, target: Path, repo: str) -> Path:
    """Fetch one artifact from the Hub into ``target``, then verify it."""
    from huggingface_hub import hf_hub_download

    destination = target / artifact.name
    logger.info("models_loader: fetching %s from %s@%s", artifact.name, repo, revision())
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=artifact.name,
            revision=revision(),
            local_dir=str(target),
            token=os.getenv("HF_TOKEN") or None,
        )
    )

    if downloaded.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(destination))

    if not artifact.usable(destination):
        size = destination.stat().st_size if destination.is_file() else -1
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded {artifact.name} from {repo}@{revision()} is unusable "
            f"({size} bytes, expected at least {artifact.min_bytes}); the file was "
            f"removed so the next start retries the download"
        )
    return destination


def ensure_artifacts(
    directory: Path | str | None = None,
    artifacts: Iterable[Artifact] | None = None,
) -> list[Path]:
    """Make every required artifact available locally and return their paths.

    Present files are left alone. Missing files are downloaded from
    ``$HF_MODEL_REPO``; when that variable is unset the function raises with the
    full list of what is missing, which is the actionable error on a Space that
    has neither committed weights nor a Hub repository configured.
    """
    target = Path(directory) if directory is not None else models_dir()
    target.mkdir(parents=True, exist_ok=True)

    wanted = list(REQUIRED_ARTIFACTS if artifacts is None else artifacts)
    if fetch_optional():
        wanted += [a for a in INFO_FILES if a not in wanted]

    absent = [a for a in wanted if not a.usable(target / a.name)]
    if not absent:
        return [target / a.name for a in wanted]

    repo = repo_id()
    if not repo:
        names = "\n".join(f"  - {a.name}" for a in absent)
        raise FileNotFoundError(
            f"{len(absent)} model artifact(s) are missing from {target} and "
            f"HF_MODEL_REPO is not set. Either restore them locally, or set "
            f"HF_MODEL_REPO to a Hugging Face Hub model repository that contains "
            f"them (HF_TOKEN only for a private repo). Missing:\n{names}"
        )

    for artifact in absent:
        _download(artifact, target, repo)

    return [target / a.name for a in wanted]