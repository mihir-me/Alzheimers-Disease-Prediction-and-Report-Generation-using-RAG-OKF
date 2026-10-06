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


# --------------------------------------------------------------------------
# B4: ensemble agreement is derived from the four individual argmaxes
# --------------------------------------------------------------------------
def _probs(**by_class):
    """One model's 4-class probability vector, from the class names it maps to."""
    return [float(by_class.get(name, 0.0)) for name in app.CLASSES]


def _one_hot(name):
    return _probs(**{name: 1.0})


def test_agreement_counts_the_models_predicting_the_ensemble_stage():
    outputs = [
        _one_hot("Moderate Impairment"),
        _one_hot("Moderate Impairment"),
        _one_hot("Moderate Impairment"),
        _one_hot("No Impairment"),
    ]
    # plurality is Moderate, and three models are on it
    assert app.model_agreement(outputs, app.CLASSES) == 3
    assert app.model_agreement(outputs, app.CLASSES, "Moderate Impairment") == 3
    assert app.model_agreement(outputs, app.CLASSES, "No Impairment") == 1


def test_unanimous_agreement_is_four():
    outputs = [_one_hot("Mild Impairment") for _ in range(4)]
    assert app.model_agreement(outputs, app.CLASSES) == 4
    assert app.model_agreement(outputs, app.CLASSES, "Mild Impairment") == 4


def test_agreement_is_case_insensitive_and_zero_when_no_model_agrees():
    outputs = [_one_hot("No Impairment"), _one_hot("No Impairment")]
    assert app.model_agreement(outputs, app.CLASSES, "no impairment") == 2
    assert app.model_agreement(outputs, app.CLASSES, "MILD IMPAIRMENT") == 0


def test_agreement_uses_the_argmax_not_a_threshold():
    """A model that merely mentions a class at 0.3 does not count as agreeing."""
    outputs = [
        _probs(**{"Mild Impairment": 0.4, "Moderate Impairment": 0.6}),
        _one_hot("Moderate Impairment"),
    ]
    assert app.model_agreement(outputs, app.CLASSES, "Mild Impairment") == 0
    assert app.model_agreement(outputs, app.CLASSES, "Moderate Impairment") == 2


def test_agreement_of_no_models_is_zero():
    assert app.model_agreement([], app.CLASSES) == 0
    assert app.model_agreement([], app.CLASSES, "Moderate Impairment") == 0


def test_stacking_label_and_calibration_caption_are_shown():
    """The exact strings the public UI must carry (B4)."""
    assert app.STACKING_LABEL == "Ensemble (stacking) probability of top class"
    assert app.STACKING_CALIBRATION_NOTE == (
        "Stacking probabilities are not calibrated; if the individual models "
        "agree but this value is low, treat the result as low confidence."
    )
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    # rendered from the constants, so the notice cannot drift from what is tested
    assert "STACKING_LABEL" in source
    assert "STACKING_CALIBRATION_NOTE" in source
    assert "Models agreeing on the stage:" in source
    # the old wording that implied a calibrated vote is gone
    assert "Ensemble predicted probability" not in source


# --------------------------------------------------------------------------
# C6: the app never quotes a hardcoded accuracy
# --------------------------------------------------------------------------
def test_metrics_are_read_from_the_evaluation_summary():
    metrics = app.load_metrics()
    import json

    raw = json.loads(
        (PROJECT_ROOT / "results" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert metrics["ensemble_test"] == raw["test"]["ensemble_accuracy"]
    assert metrics["clinical_val"] == raw["clinical"]["validation_accuracy"]
    assert metrics["individual_test"] == raw["test"]["individual"]


def test_superseded_accuracy_figures_are_absent_from_the_app():
    """The 70.69% / 94.67% pair came from an earlier run and must not be quoted."""
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "70.69" not in source
    assert "94.67" not in source


def test_methodology_reports_the_real_figures_and_the_limitations():
    lines = "\n".join(app.methodology_lines())
    assert app.fmt_pct(app.load_metrics()["ensemble_test"]) in lines
    for name in app.MODEL_NAMES:
        assert name in lines
    # the limitations a reader needs before trusting a number
    assert "not" in lines.lower() and "evidence of real-world performance" in lines
    assert "2 OASIS subjects are Moderate" in lines
    assert "Research prototype" in lines
    assert "not calibrated" in lines
    assert "sampled values" in lines


def test_missing_metrics_file_degrade_instead_of_inventing(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "results_path", lambda: tmp_path / "absent.json")
    app.load_metrics.clear()
    try:
        metrics = app.load_metrics()
        assert metrics["ensemble_test"] is None
        assert metrics["individual_test"] == {}
        assert app.fmt_pct(metrics["ensemble_test"]) == "not available"
        # the methodology still renders, with the figures marked unavailable
        assert "not available" in "\n".join(app.methodology_lines())
    finally:
        app.load_metrics.clear()


# --------------------------------------------------------------------------
# C7: the internal class labels never reach the UI
# --------------------------------------------------------------------------
def test_internal_labels_map_to_screening_wording():
    assert app.signal_headline("DEMENTED") == app.HIGHER_IMPAIRMENT_LABEL
    assert app.signal_headline("NON-DEMENTED") == app. LOWER_IMPAIRMENT_LABEL
    # anything unexpected is passed through rather than forced into a verdict
    assert app.signal_headline("SOMETHING ELSE") == "SOMETHING ELSE"
    assert app.signal_headline(None) == ""


def test_screening_wording_is_what_the_ui_shows():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "Higher likelihood of dementia-related impairment" in source
    assert "Lower likelihood of dementia-related impairment" in source
    assert "Model screening signal" in source
    assert "Final Diagnosis" not in source
    # the raw labels survive only as internal constants
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("'''"):
            continue
        if "DEMENTED" in stripped:
            assert "DEMENTED_LABEL" in stripped or "NON_DEMENTED_LABEL" in stripped, stripped


# --------------------------------------------------------------------------
# C8: the research-prototype notice is at the top of the page
# --------------------------------------------------------------------------
def test_research_prototype_notice_is_exact_and_present():
    assert app.RESEARCH_NOTICE == (
        "Research prototype. Not a medical device and not a diagnosis. "
        "Uploaded images are processed in memory and not stored."
    )
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    # rendered from the constant, so the notice cannot drift from what is tested
    assert "st.warning(f\"**{RESEARCH_NOTICE}**\")" in source

    # it has to be rendered by main(), above the upload flow
    main_body = source.split("def main():", 1)[1]
    warning_at = main_body.index("st.warning(f\"**{RESEARCH_NOTICE}**\")")
    assert warning_at < main_body.index("if uploaded_file")
    assert warning_at < main_body.index("Get Prediction")

    # and the methodology repeats it, so the limitation survives being expanded
    assert app.RESEARCH_NOTICE in "\n".join(app.methodology_lines())


# --------------------------------------------------------------------------
# C10: upload validation
# --------------------------------------------------------------------------
class _FakeUpload:
    """Minimal stand-in for a Streamlit ``UploadedFile``."""

    def __init__(self, name, data=b"", mime="image/png", size=None):
        self.name = name
        self._data = data
        self.type = mime
        self.size = len(data) if size is None else size
        self._pos = 0

    def seek(self, pos, whence=0):
        self._pos = pos

    def read(self, *_a):
        return self._data[self._pos :]


def _png_bytes():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_png_and_jpeg_uploads_are_accepted():
    assert app.validate_upload(_FakeUpload("scan.png", _png_bytes())) == (None, None)
    assert app.validate_upload(_FakeUpload("scan.jpg", _jpeg_bytes(), "image/jpeg")) == (
        None,
        None,
    )
    assert app.validate_upload(_FakeUpload("scan.JPEG", _jpeg_bytes(), "image/jpeg")) == (
        None,
        None,
    )


def test_missing_upload_is_reported():
    error, _ = app.validate_upload(None)
    assert error and "no file" in error.lower()


def test_disallowed_extension_is_refused():
    for name in ("scan.gif", "scan.bmp", "scan.tiff", "scan.pdf", "scan"):
        error, _ = app.validate_upload(_FakeUpload(name, _png_bytes()))
        assert error, name
        assert "png" in error.lower()


def test_upload_over_five_megabytes_is_refused():
    assert app.MAX_UPLOAD_BYTES == 5 * 1024 * 1024
    oversize = _FakeUpload("scan.png", b"", size=app.MAX_UPLOAD_BYTES + 1)
    error, _ = app.validate_upload(oversize)
    assert error and "5 MB" in error

    # a declared size under the cap is still checked against the real bytes
    lying = _FakeUpload("scan.png", b"x" * 16, size=app.MAX_UPLOAD_BYTES // 2)
    error, _ = app.validate_upload(lying)
    assert error and "could not be read" in error


def test_upload_at_exactly_the_limit_is_accepted():
    at_limit = _FakeUpload("scan.png", _png_bytes(), size=app.MAX_UPLOAD_BYTES)
    assert app.validate_upload(at_limit) == (None, None)


def test_non_image_mime_is_refused():
    error, _ = app.validate_upload(
        _FakeUpload("scan.png", _png_bytes(), mime="application/x-msdownload")
    )
    assert error and "not a supported image" in error


def test_corrupt_or_non_image_content_is_refused():
    error, _ = app.validate_upload(_FakeUpload("scan.png", b"not an image at all"))
    assert error and "could not be read" in error
    # a real PNG that is truncated part-way through
    error, _ = app.validate_upload(_FakeUpload("scan.png", _png_bytes()[:20]))
    assert error and "could not be read" in error


def test_the_uploader_offers_exactly_the_allowed_types():
    assert app.ALLOWED_UPLOAD_TYPES == ["png", "jpg", "jpeg"]
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "type=list(ALLOWED_UPLOAD_TYPES)" in source
    assert app.MAX_UPLOAD_BYTES // (1024 * 1024) == 5