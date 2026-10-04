"""Deployment tests: weight resolution, keyless FAISS build, per-session LLM cap."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
import models_loader  # noqa: E402
from rag_pipeline.config import load_config  # noqa: E402
from rag_pipeline.service import RagReportService  # noqa: E402


# --------------------------------------------------------------------------
# models_loader: local checkpoints win, Hub is only a fallback
# --------------------------------------------------------------------------
def test_local_artifacts_are_never_downloaded(tmp_path, monkeypatch):
    for artifact in models_loader.REQUIRED_ARTIFACTS:
        (tmp_path / artifact.name).write_bytes(b"x" * artifact.min_bytes)

    def explode(*_a, **_k):
        raise AssertionError("nothing should be downloaded when the files exist")

    monkeypatch.setattr(models_loader, "_download", explode)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)

    paths = models_loader.ensure_artifacts(tmp_path)
    assert len(paths) == len(models_loader.REQUIRED_ARTIFACTS)
    assert models_loader.missing_artifacts(tmp_path) == []


def test_missing_artifacts_without_repo_id_name_every_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        models_loader.ensure_artifacts(tmp_path)

    message = str(excinfo.value)
    assert "HF_MODEL_REPO" in message
    for artifact in models_loader.REQUIRED_ARTIFACTS:
        assert artifact.name in message


def test_only_the_missing_file_is_fetched(tmp_path, monkeypatch):
    present = models_loader.REQUIRED_ARTIFACTS[0]
    (tmp_path / present.name).write_bytes(b"x" * present.min_bytes)

    fetched: list[str] = []

    def fake_download(artifact, target, repo):
        fetched.append(artifact.name)
        (target / artifact.name).write_bytes(b"x" * artifact.min_bytes)
        return target / artifact.name

    monkeypatch.setattr(models_loader, "_download", fake_download)
    monkeypatch.setenv("HF_MODEL_REPO", "someone/weights")

    models_loader.ensure_artifacts(tmp_path)
    assert fetched == [a.name for a in models_loader.REQUIRED_ARTIFACTS[1:]]


def test_truncated_checkpoint_counts_as_missing(tmp_path, monkeypatch):
    checkpoint = next(
        a for a in models_loader.REQUIRED_ARTIFACTS if a.kind == "checkpoint"
    )
    (tmp_path / checkpoint.name).write_bytes(b"")
    assert models_loader.missing_artifacts(tmp_path, [checkpoint]) == [checkpoint]
    assert checkpoint in models_loader.missing_artifacts(tmp_path)


def test_every_required_artifact_is_resolvable_locally():
    """The repo checkout the app is developed against has all nine artifacts."""
    assert models_loader.missing_artifacts() == []
    names = {a.name for a in models_loader.REQUIRED_ARTIFACTS}
    assert names == {
        "resnet152_best.pth",
        "vgg16_best.pth",
        "efficientnet_b4_best.pth",
        "vit_best.pth",
        "ensemble_meta_learner.pkl",
        "clinical_branch_best.pth",
        "fusion_model.pth",
        "fusion_cdr_scaler.pkl",
        "fusion_nwbv_scaler.pkl",
    }


def test_info_files_are_optional():
    assert all(not a.required for a in models_loader.INFO_FILES)
    assert not set(a.name for a in models_loader.INFO_FILES) & {
        a.name for a in models_loader.REQUIRED_ARTIFACTS
    }


# --------------------------------------------------------------------------
# CPU-only inference
# --------------------------------------------------------------------------
def test_device_is_pinned_to_cpu():
    assert app.DEVICE.type == "cpu"
    assert "cuda" not in str(app.DEVICE)


def test_gradients_are_disabled_globally():
    import torch

    assert torch.is_grad_enabled() is False


# --------------------------------------------------------------------------
# FAISS builds keyless, from the bundle, with hashing embeddings
# --------------------------------------------------------------------------
def test_forced_local_embeddings_win_over_a_configured_key():
    import dataclasses

    cfg = dataclasses.replace(
        load_config(),
        api_key="sk-not-a-real-key",
        force_local_embeddings=True,
    )
    assert cfg.embedding_provider == "local-hashing"

    without_flag = dataclasses.replace(cfg, force_local_embeddings=False)
    assert without_flag.embedding_provider == "openai"


def test_index_builds_from_the_bundle_with_no_api_key(tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = dataclasses.replace(
        load_config(),
        api_key="",
        force_local_embeddings=True,
        faiss_index_path=tmp_path / "faiss_index",
    )
    service = RagReportService(cfg)
    service.embedder.force_local()

    assert service.store.n_total == 0, "a fresh Space has no persisted index"
    chunks = service.ensure_index()
    assert chunks > 0
    assert service.index_rebuilt is True
    assert service.embedder.provider == "local-hashing"
    assert (tmp_path / "faiss_index.index").is_file()


def test_keyless_report_retrieves_and_cites(tmp_path):
    import dataclasses

    cfg = dataclasses.replace(
        load_config(),
        api_key="",
        force_local_embeddings=True,
        faiss_index_path=tmp_path / "faiss_index",
    )
    service = RagReportService(cfg)
    service.embedder.force_local()
    service.ensure_index()

    payload = {
        "predicted_stage": "Mild Impairment",
        "probabilities": {"Mild Impairment": 0.7},
        "clinical_inputs": {"MMSE": 20, "CDR": 1.0, "nWBV": 0.70},
    }
    out = service.generate(payload, "what should we do next?", allow_llm=True)
    assert out["used_llm"] is False
    assert out["hits"]
    assert "[concept:" in out["report"]


# --------------------------------------------------------------------------
# per-session LLM cap
# --------------------------------------------------------------------------
class _FakeSession:
    def __init__(self):
        self.data: dict[str, int] = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __setitem__(self, key, value):
        self.data[key] = value


@pytest.fixture
def session(monkeypatch):
    state = _FakeSession()
    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    monkeypatch.delenv("LLM_REPORT_CALL_CAP", raising=False)
    return state


def test_default_cap_is_five(session):
    assert app.llm_call_budget() == 5


def test_cap_is_overridable_by_environment(session, monkeypatch):
    monkeypatch.setenv("LLM_REPORT_CALL_CAP", "2")
    assert app.llm_call_budget() == 2
    monkeypatch.setenv("LLM_REPORT_CALL_CAP", "-1")
    assert app.llm_call_budget() == 0


def test_budget_is_exhausted_after_exactly_cap_calls(session):
    budget = app.llm_call_budget()
    for _ in range(budget):
        assert app.llm_report_allowed() is True
        app.note_llm_report_call()
    assert app.llm_report_allowed() is False


def test_beyond_the_cap_the_service_returns_the_template_report(tmp_path):
    import dataclasses

    cfg = dataclasses.replace(
        load_config(),
        api_key="test-key-not-used",
        force_local_embeddings=True,
        faiss_index_path=tmp_path / "faiss_index",
    )
    service = RagReportService(cfg)
    service.embedder.force_local()
    service.ensure_index()
    payload = {
        "predicted_stage": "Mild Impairment",
        "probabilities": {"Mild Impairment": 0.7},
        "clinical_inputs": {"MMSE": 20, "CDR": 1.0, "nWBV": 0.70},
    }

    def explode(*_a, **_k):
        raise AssertionError("the LLM must not be called once the budget is spent")

    import openai

    original = openai.OpenAI
    openai.OpenAI = explode
    try:
        out = service.generate(payload, "what next?", allow_llm=False)
    finally:
        openai.OpenAI = original

    assert out["used_llm"] is False
    assert out["llm_blocked"] is True
    assert "[concept:" in out["report"]


# --------------------------------------------------------------------------
# secrets come from st.secrets / the environment only
# --------------------------------------------------------------------------
def test_secrets_are_copied_into_the_environment(monkeypatch):
    fake = _FakeSecrets({"OPENAI_API_KEY": "sk-from-st-secrets", "LLM_MODEL": "gpt-x"})
    monkeypatch.setattr(app.st, "secrets", fake, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    import os

    resolved = app.configure_from_secrets()
    assert resolved["OPENAI_API_KEY"] == "sk-from-st-secrets"
    assert os.environ["OPENAI_API_KEY"] == "sk-from-st-secrets"
    assert os.environ["LLM_MODEL"] == "gpt-x"


def test_rag_service_forces_keyless_hashing_embeddings(monkeypatch):
    """The app never lets an LLM key turn the index into an API-backed one."""
    captured = {}

    class _StubService:
        def __init__(self, cfg):
            captured["cfg"] = cfg
            self.embedder = _StubEmbedder()
            self.index_rebuilt = False

        def ensure_index(self):
            captured["ensured"] = True
            return 0

    class _StubEmbedder:
        provider = "local-hashing"

        def force_local(self):
            captured["forced_local"] = True

    monkeypatch.setattr(app, "RagReportService", _StubService)
    try:
        app.get_rag_service()
    finally:
        app.get_rag_service.clear()

    assert captured["cfg"].force_local_embeddings is True
    assert captured["cfg"].embedding_provider == "local-hashing"
    assert captured["forced_local"] is True
    assert captured["ensured"] is True


def test_environment_wins_over_secrets(monkeypatch):
    fake = _FakeSecrets({"LLM_MODEL": "from-secrets"})
    monkeypatch.setattr(app.st, "secrets", fake, raising=False)
    monkeypatch.setenv("LLM_MODEL", "from-environment")

    assert app.configure_from_secrets()["LLM_MODEL"] == "from-environment"


def test_empty_secrets_are_ignored(monkeypatch):
    monkeypatch.setattr(app.st, "secrets", _FakeSecrets({}), raising=False)
    for key in ("OPENAI_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    assert app.configure_from_secrets()["OPENAI_API_KEY"] == ""


class _FakeSecrets:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)