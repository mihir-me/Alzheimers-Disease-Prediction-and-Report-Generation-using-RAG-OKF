# app.py — Streamlit Web App with 4 MRI Models + Clinical Branch + RAG Report
# Pipeline matches notebooks/09_prediction_demo.ipynb (verified working)
#   - MRI Ensemble: ResNet-152, VGG16, EfficientNet-B4, ViT
#   - Stacking meta-learner (ensemble_meta_learner.pkl) -> 4-class MRI stage
#   - Clinical branch (clinical_branch_best.pth) -> DEMENTED / NON-DEMENTED
#   - RAG report generator (rag_pipeline) -> markdown clinical report from
#     prediction + user query + FAISS-retrieved knowledge base

import streamlit as st

import torch
import torch.nn as nn
import torchvision.models as models
import timm
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from pathlib import Path

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
MODELS_PATH = BASE_PATH / "models"

CLASSES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]


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
    path = MODELS_PATH / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
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


@st.cache_resource
def load_all_models():
    """Load the 4 MRI models, ensemble meta-learner and clinical branch."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.info("Loading ResNet-152...")
    resnet = load_checkpoint(build_resnet152(), "resnet152_best.pth", device)

    st.info("Loading VGG16...")
    vgg = load_checkpoint(build_vgg16(), "vgg16_best.pth", device)

    st.info("Loading EfficientNet-B4...")
    efficient = load_checkpoint(build_efficientnet(), "efficientnet_b4_best.pth", device)

    st.info("Loading Vision Transformer...")
    vit = load_checkpoint(build_vit(), "vit_best.pth", device)

    st.info("Loading Ensemble Meta-learner...")
    meta_learner = joblib.load(MODELS_PATH / "ensemble_meta_learner.pkl")

    st.info("Loading Clinical Branch...")
    clinical_model = load_checkpoint(ClinicalFusionNet(), "clinical_branch_best.pth", device)

    st.info("Loading Multimodal Fusion (MRI + Clinical)...")
    fusion_model = load_checkpoint(MultimodalFusionNet(), "fusion_model.pth", device)

    cdr_scaler = joblib.load(MODELS_PATH / "fusion_cdr_scaler.pkl")
    nwbv_scaler = joblib.load(MODELS_PATH / "fusion_nwbv_scaler.pkl")

    for model in [resnet, vgg, efficient, vit, clinical_model, fusion_model]:
        model.to(device)
        model.eval()

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
        'class_names': CLASSES
    }


@st.cache_resource(show_spinner=False)
def get_rag_service():
    """Build/load the FAISS knowledge-base index once and reuse it."""
    service = RagReportService()
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

    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_380 = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file).convert('RGB')
    image_224 = transform_224(image).unsqueeze(0).to(device)
    image_380 = transform_380(image).unsqueeze(0).to(device)

    results = {}

    with torch.no_grad():
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

    with torch.no_grad():
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

    with torch.no_grad():
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

    with st.spinner("Loading AI models..."):
        models_dict = load_all_models()

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
            with st.spinner("Generating clinical report (RAG)..."):
                report_payload = build_report_payload(results, mmse_score, cdr_score, nwbv_score)
                rag_output = get_rag_service().generate(report_payload, user_query)

            st.subheader("📄 Automated Clinical Report")
            rag_mode = rag_output.get('rag_mode', get_rag_service().cfg.rag_mode)
            n_concepts = len(rag_output.get('concepts') or rag_output['hits'])
            if rag_output['used_llm']:
                st.caption(
                    f"Generated with LLM ({get_rag_service().cfg.llm_model}) + "
                    f"{rag_output['index_size']} knowledge chunks · "
                    f"{n_concepts} concepts (RAG mode: {rag_mode}) · "
                    f"embeddings: {rag_output['embedding_provider']}"
                )
            else:
                reason = (
                    "no LLM API key configured"
                    if not get_rag_service().cfg.has_api_key
                    else "LLM call failed (check API key / model / base URL)"
                )
                st.caption(
                    f"Template report ({reason}) · "
                    f"{rag_output['index_size']} knowledge chunks · "
                    f"{n_concepts} concepts (RAG mode: {rag_mode}) · "
                    f"embeddings: {rag_output['embedding_provider']}"
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
        - Works offline with a template report when no API key is configured
        """)


if __name__ == "__main__":
    main()
