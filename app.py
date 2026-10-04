# app.py — Streamlit Web App with 4 MRI Models + Clinical Branch + RAG Report
# Pipeline matches notebooks/09_prediction_demo.ipynb (verified working)
#   - MRI Ensemble: ResNet-152, VGG16, EfficientNet-B4, ViT
#   - Stacking meta-learner (ensemble_meta_learner.pkl) -> 4-class MRI stage
#   - Clinical branch (clinical_branch_best.pth) -> DEMENTED / NON-DEMENTED
#   - RAG report generator (rag_pipeline) -> markdown clinical report from
#     prediction + user query + FAISS-retrieved knowledge base
#
# Deployment target: a free CPU-only host (Hugging Face Space). The app never
# touches a GPU, every checkpoint is loaded exactly once into a single cached
# resource, weights come from models_loader (local first, Hub as a fallback),
# and nothing at all is required from a .env file: every setting comes from
# st.secrets or the process environment.

import dataclasses
import gc
import os
import platform

import matplotlib

matplotlib.use("Agg")

import streamlit as st

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from models_loader import (
    current_rss_mb,
    describe as describe_artifacts,
    ensure_artifacts,
    log_rss,
    models_dir,
)
from rag_pipeline.config import load_config
from rag_pipeline.service import RagReportService
from rag_pipeline.vector_store import ORIGIN_LINKED, ORIGIN_STAGE_REQUIRED, origin_of

st.set_page_config(
    page_title="Alzheimer's Disease Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .demented {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .non-demented {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    .header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

BASE_PATH = Path(__file__).resolve().parent

CLASSES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

# Free CPU hosts have no GPU and the weights are far too large to commit, so the
# device is pinned rather than probed.
DEVICE = torch.device("cpu")

# LLM report calls are metered per browser session: a shared public Space must
# not let one visitor spend the whole API quota. Past the cap the deterministic
# template report is used instead.
LLM_REPORT_CALL_CAP = 5

# Every deployment setting, read from st.secrets first and the process
# environment second. Nothing is read from a .env file, so a Space configured
# with secrets.toml (or with Space variables) starts without one.
SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "RAG_MODE",
    "RAG_MAX_CONCEPTS",
    "RAG_MIN_SCORE",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "KNOWLEDGE_DIR",
    "OKF_BUNDLE_DIR",
    "FAISS_INDEX_PATH",
    "MODELS_DIR",
    "HF_MODEL_REPO",
    "HF_MODEL_REVISION",
    "HF_TOKEN",
)

_THREADS = os.getenv("TORCH_NUM_THREADS", "").strip()
if _THREADS.isdigit() and int(_THREADS) > 0:
    torch.set_num_threads(int(_THREADS))
torch.set_grad_enabled(False)


def _secret(name: str) -> str:
    """Value of ``name`` from st.secrets, or '' when it is not configured."""
    try:
        value = st.secrets.get(name)
    except Exception:
        return ""
    return str(value).strip() if value is not None else ""


def configure_from_secrets() -> dict[str, str]:
    """Bridge st.secrets / environment variables into ``os.environ``.

    ``rag_pipeline`` and ``models_loader`` read plain environment variables, so
    resolving them once here keeps a single configuration path for a local run,
    a Space with secrets.toml and a Space with Space variables. Returns the
    resolved values so the caller can show what the app is actually using.
    """
    resolved: dict[str, str] = {}
    for name in SECRET_ENV_KEYS:
        value = os.environ.get(name, "").strip() or _secret(name)
        if value:
            os.environ[name] = value
        resolved[name] = value
    return resolved


def llm_call_budget() -> int:
    """How many LLM report calls this deployment allows per session."""
    raw = os.getenv("LLM_REPORT_CALL_CAP", "").strip()
    if raw.lstrip("-").isdigit():
        return max(0, int(raw))
    return LLM_REPORT_CALL_CAP


def llm_report_allowed() -> bool:
    """Whether this session may still spend an LLM call."""
    return int(st.session_state.get("llm_report_calls", 0)) < llm_call_budget()


def note_llm_report_call() -> None:
    st.session_state["llm_report_calls"] = (
        int(st.session_state.get("llm_report_calls", 0)) + 1
    )


def build_resnet152():
    model = models.resnet152(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 4))
    return model


def build_vgg16():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4))
    return model


def build_efficientnet():
    return timm.create_model("efficientnet_b4", pretrained=False, num_classes=4)


def build_vit():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=4)


def load_checkpoint(model, filename, device):
    """Load one state_dict into ``model`` without keeping a second copy around.

    ``map_location`` is pinned to ``"cpu"`` so a checkpoint never lands on an
    accelerator the host does not have. ``weights_only=True`` refuses to
    unpickle arbitrary objects, and the transient dict is dropped and collected
    before the next checkpoint is read, so peak RAM stays at one model plus one
    checkpoint instead of all of them at once.
    """
    path = models_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        checkpoint = torch.load(path, map_location="cpu")
    try:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)
    finally:
        del checkpoint
        gc.collect()
    model.to(device)
    model.eval()
    return model


class ClinicalFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


class MultimodalFusionNet(nn.Module):
    """True multimodal fusion: MRI (16 probs) + clinical (3 scores) -> 2-class."""

    def __init__(self, n_mri=16, n_clinical=3, n_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_mri + n_clinical, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.network(x)


def build_transforms():
    """Inference transforms, built once and reused for every prediction."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return {
        size: transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
        for size in (224, 380)
    }


def system_info_lines() -> list[str]:
    """Runtime facts about this deployment, RSS included.

    The free CPU host caps the app at roughly 2.7 GB, so the resident set size
    is the number to watch when the Space is close to being killed. It comes
    from ``/proc/self/status`` and reads "unavailable" where that does not exist
    (a local Windows run), never an exception.
    """
    rss = current_rss_mb()
    rss_text = f"{rss:,.0f} MB" if rss is not None else "unavailable on this platform"
    return [
        f"python              : {platform.python_version()} "
        f"({platform.system()} {platform.machine()})",
        f"device              : {DEVICE} (torch {torch.__version__})",
        f"torch threads       : {torch.get_num_threads()}",
        f"process RSS         : {rss_text}",
        f"models directory    : {models_dir()}",
    ]


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load the 4 MRI models, ensemble meta-learner, clinical branch and fusion.

    Runs once per server process. Weights are resolved by ``models_loader``
    first, so a checkout that already has ``models/*.pth`` behaves exactly as
    before and a fresh Space fetches them from the Hub once.

    The RSS of the server process is logged after every model: a free Space caps
    the app at roughly 2.7 GB, so the log line is the only way to tell a
    borderline deployment from an OOM kill. Nothing is printed to stdout by the
    app itself; Streamlit surfaces the log lines in its own console.
    """
    device = DEVICE

    st.info("Resolving model checkpoints...")
    ensure_artifacts()
    log_rss("resolving model checkpoints")

    st.info("Loading ResNet-152...")
    resnet = load_checkpoint(build_resnet152(), "resnet152_best.pth", device)
    log_rss("loading ResNet-152")

    st.info("Loading VGG16...")
    vgg = load_checkpoint(build_vgg16(), "vgg16_best.pth", device)
    log_rss("loading VGG16")

    st.info("Loading EfficientNet-B4...")
    efficient = load_checkpoint(build_efficientnet(), "efficientnet_b4_best.pth", device)
    log_rss("loading EfficientNet-B4")

    st.info("Loading Vision Transformer...")
    vit = load_checkpoint(build_vit(), "vit_best.pth", device)
    log_rss("loading Vision Transformer")

    st.info("Loading Ensemble Meta-learner...")
    meta_learner = joblib.load(models_dir() / "ensemble_meta_learner.pkl")
    gc.collect()
    log_rss("loading the ensemble meta-learner")

    st.info("Loading Clinical Branch...")
    clinical_model = load_checkpoint(ClinicalFusionNet(), "clinical_branch_best.pth", device)
    log_rss("loading the clinical branch")

    st.info("Loading Multimodal Fusion (MRI + Clinical)...")
    fusion_model = load_checkpoint(MultimodalFusionNet(), "fusion_model.pth", device)
    log_rss("loading the multimodal fusion")

    cdr_scaler = joblib.load(models_dir() / "fusion_cdr_scaler.pkl")
    nwbv_scaler = joblib.load(models_dir() / "fusion_nwbv_scaler.pkl")

    # Every model went through load_checkpoint, which already moved it to the
    # device and put it in eval mode; this only drops the joblib transients.
    gc.collect()
    log_rss("loading every model")

    return {
        'resnet': resnet,
        'vgg': vgg,
        'efficient': efficient,
        'vit': vit,
        'meta_learner': meta_learner,
        'clinical_model': clinical_model,
        'fusion_model': fusion_model,
        'cdr_scaler': cdr_scaler,
        'nwbv_scaler': nwbv_scaler,
        'device': device,
        'class_names': CLASSES,
        'transforms': build_transforms(),
    }


@st.cache_resource(show_spinner=False)
def get_rag_service():
    """Build the FAISS knowledge-base index once and reuse it.

    The index is always built with the deterministic local hashing embedder, even
    when an LLM key is configured, so a Space with no API key still retrieves
    knowledge and the index is a pure function of the bundle content.
    """
    configure_from_secrets()
    cfg = dataclasses.replace(load_config(), force_local_embeddings=True)
    service = RagReportService(cfg)
    service.embedder.force_local()
    service.ensure_index()
    return service


def build_report_payload(results, mmse_score, cdr_score, nwbv_score):
    """Structured prediction dict passed to the RAG report generator."""
    class_names = results['_class_names']
    ensemble_probs = results['Ensemble']['probs']
    probabilities = {
        name: float(p) for name, p in zip(class_names, ensemble_probs)
    }
    final = results['Final']
    return {
        'predicted_stage': results['Ensemble']['class'],
        'probabilities': probabilities,
        'ensemble_confidence': round(results['Ensemble']['confidence'], 2),
        'clinical_inputs': {
            'MMSE': int(mmse_score),
            'CDR': float(cdr_score),
            'nWBV': float(nwbv_score),
        },
        'clinical_diagnosis': final['clinical_diagnosis'],
        'clinical_demented_probability': round(final['clinical_demented_prob'], 2),
        'clinical_non_demented_probability': round(final['clinical_non_demented_prob'], 2),
        'fusion_diagnosis': final['diagnosis'],
        'fusion_confidence': round(final['confidence'], 2),
        'fusion_demented_probability': round(final['demented_prob'], 2),
        'fusion_non_demented_probability': round(final['non_demented_prob'], 2),
        'mri_stage_implies_dementia': bool(final['mri_stage_demented']),
        'mri_clinical_conflict': bool(final['conflict']),
        'needs_clinical_review': bool(final['conflict']),
    }


def predict_alzheimers(models_dict, image_file, mmse_score, cdr_score, nwbv_score):
    """Full prediction pipeline: 4-model MRI ensemble + clinical branch."""

    if image_file is None:
        raise ValueError("An MRI image is required")
    try:
        mmse_score = float(mmse_score)
        cdr_score = float(cdr_score)
        nwbv_score = float(nwbv_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Clinical scores must be numeric") from exc
    if not 0 <= mmse_score <= 30:
        raise ValueError("MMSE must be between 0 and 30")
    if cdr_score not in (0.0, 0.5, 1.0, 2.0, 3.0):
        raise ValueError("CDR must be one of 0, 0.5, 1, 2, or 3")
    if not 0.60 <= nwbv_score <= 0.85:
        raise ValueError("nWBV must be between 0.60 and 0.85")

    resnet = models_dict['resnet']
    vgg = models_dict['vgg']
    efficient = models_dict['efficient']
    vit = models_dict['vit']
    meta_learner = models_dict['meta_learner']
    clinical_model = models_dict['clinical_model']
    fusion_model = models_dict['fusion_model']
    cdr_scaler = models_dict['cdr_scaler']
    nwbv_scaler = models_dict['nwbv_scaler']
    device = models_dict['device']
    class_names = models_dict['class_names']
    precomputed = models_dict['transforms']

    transform_224 = precomputed[224]
    transform_380 = precomputed[380]

    image = Image.open(image_file).convert('RGB')
    image_224 = transform_224(image).unsqueeze(0).to(device)
    image_380 = transform_380(image).unsqueeze(0).to(device)

    results = {}

    with torch.inference_mode():
        out_resnet = torch.softmax(resnet(image_224), dim=1)[0].cpu().numpy()
        out_vgg = torch.softmax(vgg(image_224), dim=1)[0].cpu().numpy()
        out_eff = torch.softmax(efficient(image_380), dim=1)[0].cpu().numpy()
        out_vit = torch.softmax(vit(image_224), dim=1)[0].cpu().numpy()

    model_names = ['ResNet-152', 'VGG16', 'EfficientNet-B4', 'Vision Transformer']
    outputs = [out_resnet, out_vgg, out_eff, out_vit]

    for name, output in zip(model_names, outputs):
        results[name] = {
            'class': class_names[output.argmax()],
            'confidence': output.max() * 100,
            'probs': output
        }

    # MRI Ensemble: 16 features (4 probabilities x 4 models)
    features = np.hstack(outputs).reshape(1, -1)

    ensemble_pred = meta_learner.predict(features)[0]
    ensemble_probs = meta_learner.predict_proba(features)[0]
    ensemble_conf = ensemble_probs.max() * 100
    mri_class = class_names[ensemble_pred]

    results['Ensemble'] = {
        'class': mri_class,
        'confidence': ensemble_conf,
        'probs': ensemble_probs
    }

    # Clinical Branch: mmse_norm, cdr_norm, nwbv_norm
    mmse_norm = mmse_score / 30.0
    cdr_norm = cdr_scaler.transform([[cdr_score]])[0][0]
    nwbv_norm = nwbv_scaler.transform([[nwbv_score]])[0][0]

    clinical_input = np.array([mmse_norm, cdr_norm, nwbv_norm], dtype=np.float32)
    clinical_tensor = torch.tensor(clinical_input).unsqueeze(0).to(device)

    with torch.inference_mode():
        clinical_output = clinical_model(clinical_tensor)
        clinical_probs = torch.softmax(clinical_output, dim=1)[0].cpu().numpy()

    clinical_pred = clinical_output.argmax(dim=1).item()
    clinical_diagnosis = 'DEMENTED' if clinical_pred == 0 else 'NON-DEMENTED'
    clinical_confidence = clinical_probs.max() * 100

    results['Clinical'] = {
        'diagnosis': clinical_diagnosis,
        'demented_prob': clinical_probs[0] * 100,
        'non_demented_prob': clinical_probs[1] * 100,
        'confidence': clinical_confidence
    }

    # ---- Multimodal Fusion: 16 MRI probs + 3 clinical features ----
    fusion_features = np.hstack([features.reshape(-1), [mmse_norm, cdr_norm, nwbv_norm]]).astype(np.float32)
    fusion_tensor = torch.tensor(fusion_features).unsqueeze(0).to(device)

    with torch.inference_mode():
        fusion_output = fusion_model(fusion_tensor)
        fusion_probs = torch.softmax(fusion_output, dim=1)[0].cpu().numpy()

    fusion_pred = fusion_output.argmax(dim=1).item()
    fusion_diagnosis = 'DEMENTED' if fusion_pred == 0 else 'NON-DEMENTED'
    fusion_confidence = fusion_probs.max() * 100

    # ---- Reconciliation: does the MRI stage agree with the clinical branch? ----
    mri_stage_demented = mri_class != 'No Impairment'
    clinical_demented = (clinical_pred == 0)
    conflict = bool(mri_stage_demented != clinical_demented)

    results['Fusion'] = {
        'diagnosis': fusion_diagnosis,
        'demented_prob': fusion_probs[0] * 100,
        'non_demented_prob': fusion_probs[1] * 100,
        'confidence': fusion_confidence
    }

    results['Final'] = {
        'diagnosis': fusion_diagnosis,
        'demented_prob': fusion_probs[0] * 100,
        'non_demented_prob': fusion_probs[1] * 100,
        'confidence': fusion_confidence,
        'conflict': conflict,
        'mri_stage': mri_class,
        'mri_stage_demented': mri_stage_demented,
        'clinical_diagnosis': clinical_diagnosis,
        'clinical_demented_prob': clinical_probs[0] * 100,
        'clinical_non_demented_prob': clinical_probs[1] * 100,
    }

    results['_class_names'] = class_names

    return results, image


def main():

    st.markdown('<p class="header">🧠 Alzheimer\'s Disease Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")

    settings = configure_from_secrets()

    with st.spinner("Loading AI models..."):
        models_dict = load_all_models()
        rag_service = get_rag_service()

    # Rendered after the load, so the RSS below is what the process actually
    # holds rather than the interpreter's size before the weights arrived.
    with st.expander("System info"):
        st.caption(
            "Runtime facts about this deployment, measured after the models were "
            "loaded. The RSS is the resident set size of the server process, "
            "which is what the free CPU host caps."
        )
        st.code("\n".join(system_info_lines()), language="text")

    st.sidebar.title("📋 Patient Information")

    uploaded_file = st.sidebar.file_uploader(
        "Upload MRI Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI image (JPG or PNG)"
    )

    st.sidebar.subheader("Clinical Data")

    mmse_score = st.sidebar.slider("MMSE Score", min_value=0, max_value=30, value=24)
    cdr_score = st.sidebar.select_slider("CDR Score", options=[0, 0.5, 1, 2, 3], value=0.5)
    nwbv_score = st.sidebar.slider("nWBV Score", min_value=0.60, max_value=0.85, value=0.73, step=0.01)

    st.sidebar.subheader("💬 Report Question (RAG)")
    user_query = st.sidebar.text_area(
        "Ask a question about this patient",
        value="What do these results suggest about this patient's cognitive status and what should we do next?",
        help="The report is generated from this question plus retrieved medical knowledge (RAG)."
    )

    if uploaded_file is not None:

        if st.sidebar.button("🔮 Get Prediction", use_container_width=True):

            with st.spinner("Analyzing MRI and clinical data..."):
                results, image = predict_alzheimers(
                    models_dict,
                    uploaded_file,
                    mmse_score,
                    cdr_score,
                    nwbv_score
                )

            # ---- RAG clinical report (shown first) ----
            budget = llm_call_budget()
            used = int(st.session_state.get("llm_report_calls", 0))
            with st.spinner("Generating clinical report (RAG)..."):
                report_payload = build_report_payload(results, mmse_score, cdr_score, nwbv_score)
                rag_output = rag_service.generate(
                    report_payload, user_query, allow_llm=used < budget
                )
            if rag_output['used_llm']:
                note_llm_report_call()
                used += 1

            st.subheader("📄 Automated Clinical Report")
            rag_mode = rag_output.get('rag_mode', rag_service.cfg.rag_mode)
            n_concepts = len(rag_output.get('concepts') or rag_output['hits'])
            if rag_output['used_llm']:
                st.caption(
                    f"Generated with LLM ({rag_service.cfg.llm_model}) + "
                    f"{rag_output['index_size']} knowledge chunks · "
                    f"{n_concepts} concepts (RAG mode: {rag_mode}) · "
                    f"embeddings: {rag_output['embedding_provider']} · "
                    f"LLM calls left this session: {max(0, budget - used)}"
                )
            else:
                if rag_output.get("llm_blocked"):
                    reason = (
                        f"session LLM cap of {budget} reached, falling back to the "
                        f"template report"
                    )
                elif not rag_service.cfg.has_api_key:
                    reason = "no LLM API key configured"
                else:
                    reason = "LLM call failed (check API key / model / base URL)"
                st.caption(
                    f"Template report ({reason}) · "
                    f"{rag_output['index_size']} knowledge chunks · "
                    f"{n_concepts} concepts (RAG mode: {rag_mode}) · "
                    f"embeddings: {rag_output['embedding_provider']} · "
                    f"LLM calls left this session: {max(0, budget - used)}"
                )
            st.markdown(rag_output['report'])
            with st.expander("🔎 Retrieved knowledge sources"):
                for h in rag_output['hits']:
                    origin = origin_of(h)
                    if origin == ORIGIN_STAGE_REQUIRED:
                        label = f"stage-required, score {h.score:.3f}"
                    elif origin == ORIGIN_LINKED:
                        label = f"linked from {h.link_source}, score {h.score:.3f}"
                    else:
                        label = f"direct, score {h.score:.3f}"
                    st.markdown(
                        f"**`{h.concept_id or h.source}`** — {h.title} "
                        f"*({label})*"
                    )
                    st.markdown(h.text)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📸 Uploaded MRI Image")
                st.image(image)

            with col2:
                st.subheader("🎯 Final Diagnosis (Fused MRI + Clinical)")

                final_diagnosis = results['Final']['diagnosis']
                confidence = results['Final']['confidence']
                demented_prob = results['Final']['demented_prob']
                non_demented_prob = results['Final']['non_demented_prob']

                if final_diagnosis == 'DEMENTED':
                    css_class = 'demented'
                    emoji = '⚠️'
                else:
                    css_class = 'non-demented'
                    emoji = '✅'

                diagnosis_html = f"""
                <div class="prediction-box {css_class}">
                    <h2>{emoji} {final_diagnosis}</h2>
                    <p><strong>Predicted class probability:</strong> {confidence:.2f}%</p>
                    <p><strong>Demented Prob:</strong> {demented_prob:.2f}%</p>
                    <p><strong>Non-Demented Prob:</strong> {non_demented_prob:.2f}%</p>
                </div>
                """
                st.markdown(diagnosis_html, unsafe_allow_html=True)

                if results['Final']['conflict']:
                    mri_stage = results['Final']['mri_stage']
                    clinical_dx = results['Final']['clinical_diagnosis']
                    st.error(
                        f"⚠️ **Conflicting signals — needs clinician review.** "
                        f"The MRI models predict **{mri_stage}** (a {'' if results['Final']['mri_stage_demented'] else 'non-'}demented "
                        f"stage) while the clinical branch predicts **{clinical_dx}**. "
                        f"The fused verdict above combines both, but the disagreement "
                        f"means neither modality alone is decisive."
                    )
                else:
                    st.success("MRI stage and clinical branch agree — signals are consistent.")

            st.markdown("---")

            st.subheader("🤖 Individual Model Predictions (4 Models)")

            col1, col2, col3, col4 = st.columns(4)

            models_to_show = ['ResNet-152', 'VGG16', 'EfficientNet-B4', 'Vision Transformer']
            cols = [col1, col2, col3, col4]

            for model_name, col in zip(models_to_show, cols):
                with col:
                    pred_class = results[model_name]['class']
                    confidence = results[model_name]['confidence']
                    st.metric(label=model_name, value=pred_class, delta=f"{confidence:.1f}% predicted probability")

            st.markdown("---")

            st.subheader("🗳️ Ensemble Prediction (4 Models Vote)")

            ensemble_class = results['Ensemble']['class']
            ensemble_conf = results['Ensemble']['confidence']

            st.info(f"**MRI Stage:** {ensemble_class}\n**Ensemble predicted probability:** {ensemble_conf:.2f}%")

            st.markdown("---")

            st.subheader("🧬 Fused Model (MRI + Clinical)")

            fusion = results['Fusion']
            clinical = results['Clinical']

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Fused Diagnosis", fusion['diagnosis'], delta=f"{fusion['confidence']:.1f}% predicted probability")
            col_b.metric("Clinical Branch", clinical['diagnosis'], delta=f"{clinical['confidence']:.1f}% predicted probability")
            col_c.metric("Demented Prob", f"{fusion['demented_prob']:.1f}%")

            st.caption(
                "The fused model takes the 16 MRI probabilities plus MMSE / CDR / nWBV "
                "and outputs a single DEMENTED / NON-DEMENTED verdict."
            )

            st.markdown("---")

            st.subheader("📊 Probability Distributions")

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Model Predictions Distribution', fontsize=14, fontweight='bold')

            class_names = models_dict['class_names']

            for idx, model_name in enumerate(models_to_show):
                ax = axes[idx // 2, idx % 2]
                probs = results[model_name]['probs']
                colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff']
                ax.bar(class_names, probs, color=colors)
                ax.set_title(model_name, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.tick_params(axis='x', rotation=45)
                for i, v in enumerate(probs):
                    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')

            ax = axes[1, 1]
            final_probs = [results['Final']['demented_prob'] / 100, results['Final']['non_demented_prob'] / 100]
            ax.bar(['Demented', 'Non-Demented'], final_probs, color=['#ff6b6b', '#6bcf7f'])
            ax.set_title('Final Diagnosis (Fusion)', fontweight='bold')
            ax.set_ylim([0, 1])
            for i, v in enumerate(final_probs):
                ax.text(i, v + 0.02, f'{v * 100:.1f}%', ha='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("👈 Upload an MRI image to get started")

        st.subheader("📝 How to Use")
        st.markdown("""
        1. Upload a brain MRI scan (JPG or PNG)
        2. Enter patient clinical data (MMSE, CDR, nWBV)
        3. Optionally ask a question for the report (RAG)
        4. Click 'Get Prediction'
        5. View the auto-generated clinical report, then all 4 models + ensemble + clinical branch
        """)

        st.subheader("🧠 System Architecture")
        st.markdown("""
        **4 Deep Learning Models (MRI):**
        - ResNet-152
        - VGG16
        - EfficientNet-B4
        - Vision Transformer

        **Stacking Ensemble:**
        - 16 features (4 classes x 4 models) fed to a Logistic Regression meta-learner
        - OASIS test accuracy: **70.69%** (subject-disjoint split)

        **Clinical Branch:**
        - Combines MMSE, CDR and nWBV clinical scores
        - Predicts DEMENTED vs NON-DEMENTED (OASIS val accuracy: **94.67%**)

        **Multimodal Fusion (MRI + Clinical):**
        - 16 MRI probabilities + MMSE/CDR/nWBV -> single fused verdict
        - Trained subject-disjoint on OASIS; reconciles imaging & clinical signals
        - Flags **conflicting signals** when the MRI stage and clinical branch disagree

        **RAG Report Generator:**
        - FAISS vector index over a medical knowledge base
        - Retrieves context on your question, then an LLM writes the markdown report
        - Deterministic hashing embeddings, so no API key is needed to build the index
        - Works offline with a template report when no API key is configured
        """)

        with st.expander("⚙️ Deployment status"):
            st.code(
                "\n".join([
                    f"device                : {DEVICE} (torch {torch.__version__})",
                    f"torch threads         : {torch.get_num_threads()}",
                    f"models directory      : {models_dir()}",
                    f"LLM report call cap   : {llm_call_budget()} per session, "
                    f"{int(st.session_state.get('llm_report_calls', 0))} used",
                    f"LLM endpoint          : "
                    f"{rag_service.cfg.base_url if rag_service.cfg.has_api_key else '(none — template report)'}",
                    f"RAG mode              : {rag_service.cfg.rag_mode}",
                    f"embeddings            : {rag_service.embedder.provider}",
                    f"FAISS index chunks    : {rag_service.store.n_total} "
                    f"(rebuilt at startup: {rag_service.index_rebuilt})",
                    f"HF_MODEL_REPO         : {settings['HF_MODEL_REPO'] or '(not set)'}",
                    f"HF_TOKEN              : {'configured' if settings['HF_TOKEN'] else '(not set)'}",
                    "",
                    *system_info_lines(),
                    "",
                    describe_artifacts(),
                ]),
                language="text",
            )


if __name__ == "__main__":
    main()
