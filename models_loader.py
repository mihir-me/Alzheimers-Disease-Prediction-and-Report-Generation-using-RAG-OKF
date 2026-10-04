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
Every setting below is read from the process environment first and from
``st.secrets`` second, which is the order a Streamlit Community Cloud app sees:
its secrets arrive as environment variables, while a local run may only have a
``.streamlit/secrets.toml``. A missing secrets file is not an error, it simply
yields no value.

``HF_MODEL_REPO``
    Hub repository id, e.g. ``alice/alz-mri-weights``. Required only when an
    artifact is missing locally.
``HF_MODEL_REVISION``
    Branch, tag or commit to fetch. Defaults to ``main``.
``HF_TOKEN``
    Hugging Face token, needed only for private repositories. Public
    repositories work without it. Its value is never logged or printed.
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
import sys
import tempfile
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

#: Prefix of the per-download staging directories. Anything left over from a
#: process that was killed mid-download is removed by ``ensure_artifacts``.
STAGING_PREFIX = ".staging-"

#: Suffix of a half-written file. A partial download never reaches the models
#: directory under this name -- the staging directory is what an interrupted
#: transfer leaves behind -- but a stray ``.part`` from an older build or a
#: manual copy is removed on start rather than mistaken for a checkpoint.
PART_SUFFIX = ".part"


def _secrets_value(name: str) -> str:
    """Value of ``name`` from ``st.secrets``, or ``''``.

    Never raises: outside a Streamlit runtime, and on a checkout without a
    secrets file, ``st.secrets`` either has no such key or refuses to load at
    all. Both cases mean "not configured".
    """
    streamlit = sys.modules.get("streamlit")
    if streamlit is None:
        try:
            import streamlit
        except Exception:  # pragma: no cover - streamlit is a hard app dep
            return ""
    try:
        value = streamlit.secrets.get(name)
    except Exception:
        return ""
    return str(value).strip() if value is not None else ""


def setting(name: str, default: str = "") -> str:
    """Deployment setting from the environment, falling back to ``st.secrets``."""
    return os.getenv(name, "").strip() or _secrets_value(name) or default


def current_rss_mb() -> float | None:
    """Resident set size of this process in MB, or ``None`` when unavailable.

    ``psutil`` is a dev-only dependency (see ``requirements-dev.txt``), so the
    reading comes from ``/proc/self/status`` on Linux. Anything else -- Windows,
    a container without procfs -- returns ``None`` instead of raising, so the
    callers can stay unconditional.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        return None
    return None


def log_rss(label: str) -> float | None:
    """Log the current RSS after ``label``, e.g. ``after loading ResNet-152``."""
    rss = current_rss_mb()
    if rss is None:
        logger.info("models_loader: RSS unavailable after %s", label)
    else:
        logger.info("models_loader: RSS after %s: %.0f MB", label, rss)
    return rss


def models_dir(base_dir: Path | str | None = None) -> Path:
    """Directory the checkpoints live in, honouring ``MODELS_DIR``."""
    override = setting("MODELS_DIR")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else Path(base_dir or BASE_DIR) / path
    return Path(base_dir or BASE_DIR) / DEFAULT_MODELS_DIRNAME


def repo_id() -> str:
    """Hub repository the missing artifacts are fetched from (``""`` if unset)."""
    return setting("HF_MODEL_REPO")


def revision() -> str:
    """Hub branch / tag / commit to fetch."""
    return setting("HF_MODEL_REVISION") or "main"


def hf_token() -> str | None:
    """Hub token for a private repository (``None`` when not configured).

    Only ever passed to ``hf_hub_download``: the value is never logged, printed
    or put in an error message.
    """
    return setting("HF_TOKEN") or None


def fetch_optional() -> bool:
    """Whether the ``*_info.json`` provenance files should be fetched too."""
    return setting("HF_MODEL_FETCH_OPTIONAL").lower() in _TRUTHY


def _is_writable(directory: Path) -> bool:
    """Whether a file can actually be created in ``directory``."""
    probe = directory / ".write-probe"
    try:
        with open(probe, "wb") as handle:
            handle.write(b"0")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def writable_models_dir(base_dir: Path | str | None = None) -> Path:
    """``models_dir()`` if it accepts writes, otherwise a temp-directory copy.

    A read-only app directory is the normal failure mode on a host that mounts
    the checkout read-only, and it would otherwise surface as an opaque
    ``OSError`` in the middle of the first download.
    """
    target = models_dir(base_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("models_loader: cannot create %s", target)
    if _is_writable(target):
        return target

    fallback = Path(tempfile.gettempdir()) / DEFAULT_MODELS_DIRNAME
    fallback.mkdir(parents=True, exist_ok=True)
    if not _is_writable(fallback):
        raise RuntimeError(
            f"neither {target} nor {fallback} is writable, so the checkpoints "
            "cannot be stored; set MODELS_DIR to a writable directory"
        )
    logger.warning(
        "models_loader: %s is not writable, falling back to %s", target, fallback
    )
    return fallback


def _clean_partials(target: Path) -> None:
    """Drop staging directories and ``*.part`` files left by a killed process.

    Called on every start: a Space that was hibernated mid-download comes back
    with a half-written file, and the only safe response to that is to throw it
    away and fetch the artifact again.
    """
    try:
        entries = list(target.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir() and entry.name.startswith(STAGING_PREFIX):
            shutil.rmtree(entry, ignore_errors=True)
            logger.info("models_loader: removed stale staging dir %s", entry.name)
        elif entry.name.endswith(PART_SUFFIX):
            entry.unlink(missing_ok=True)
            logger.info("models_loader: removed partial download %s", entry.name)


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
    """Fetch one artifact from the Hub into ``target``, then verify it.

    The bytes land in a per-file staging directory and are moved into place with
    a single ``os.replace``, which is atomic on one filesystem. So a Space that
    is hibernated (or killed) halfway through a 537 MB checkpoint never leaves a
    truncated ``.pth`` behind for the next start to load: the staging directory
    is discarded and the file is simply fetched again.
    """
    from huggingface_hub import hf_hub_download

    destination = target / artifact.name
    staging = Path(tempfile.mkdtemp(prefix=f"{STAGING_PREFIX}{artifact.name}-", dir=str(target)))
    logger.info(
        "models_loader: fetching %s from %s@%s (token: %s)",
        artifact.name,
        repo,
        revision(),
        "set" if hf_token() else "none",
    )
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo,
                filename=artifact.name,
                revision=revision(),
                local_dir=str(staging),
                token=hf_token(),
            )
        )
        if not artifact.usable(downloaded):
            size = downloaded.stat().st_size if downloaded.is_file() else -1
            raise RuntimeError(
                f"downloaded {artifact.name} from {repo}@{revision()} is unusable "
                f"({size} bytes, expected at least {artifact.min_bytes}); nothing "
                f"was installed, so the next start retries the download"
            )
        os.replace(downloaded, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log_rss(f"downloading {artifact.name}")
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
    target = Path(directory) if directory is not None else writable_models_dir()
    target.mkdir(parents=True, exist_ok=True)
    _clean_partials(target)

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