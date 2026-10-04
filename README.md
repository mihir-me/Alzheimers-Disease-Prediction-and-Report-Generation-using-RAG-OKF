# Alzheimer's Disease Prediction System

> **Research prototype, not a medical device.** Nothing in this repository is
> cleared, approved or validated for clinical use, for diagnosis, or for any
> treatment decision. The outputs are the output of student-grade models
> evaluated on a small academic dataset. Do not use them to inform the care of a
> real person. Always consult a qualified clinician.

A CPU-only Streamlit app that classifies brain MRI scans into four cognitive
staging classes, fuses that result with clinical scores (MMSE, CDR, nWBV), and
generates a cited clinical report with retrieval-augmented generation (RAG) over
a versioned medical knowledge bundle.

## Pipeline

| Stage | What it does |
| --- | --- |
| MRI backbones | ResNet-152, VGG16, EfficientNet-B4, ViT-B/16 — 4-class staging (No / Very Mild / Mild / Moderate Impairment) |
| Stacking ensemble | A scikit-learn meta-learner over the 16 concatenated probabilities (4 classes × 4 models) |
| Clinical branch | Small torch head over MMSE / CDR / nWBV → DEMENTED / NON-DEMENTED |
| Multimodal fusion | 16 MRI probabilities + 3 clinical features → one fused verdict, and a flag when imaging and clinical signals disagree |
| RAG report | FAISS index over the OKF knowledge bundle → retrieved, concept-cited markdown report. Works offline with a deterministic template; uses an LLM when a key is configured |

The prediction path matches `notebooks/09_prediction_demo.ipynb`.

## Accuracy

From `results/evaluation_summary.json`, subject-disjoint OASIS splits:

| Model | Validation | Test |
| --- | --- | --- |
| ResNet-152 | 63.12% | 58.60% |
| VGG16 | 60.94% | 55.47% |
| EfficientNet-B4 | 59.75% | 60.13% |
| ViT-B/16 | 67.25% | 63.40% |
| **Stacking ensemble** | **70.56%** | **62.20%** |

Clinical branch: 97.33% validation accuracy. Fusion model: 100% test accuracy
across 52 test subjects, with 11 flagged imaging/clinical conflicts — a small
sample, so read it as a sanity check rather than a result.

## Repository layout

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit entry point: models, inference, RAG report, UI |
| `models_loader.py` | Resolves checkpoints (local first, Hugging Face Hub as fallback) and reports process RSS |
| `rag_pipeline/` | Chunking, embeddings, FAISS vector store, OKF loader, report generator, config |
| `knowledge/okf_bundle/` | Versioned medical knowledge bundle (27 concepts) consumed by the RAG stage |
| `train_*.py`, `splits.py`, `evaluate.py` | Training, subject-disjoint splitting and evaluation |
| `scripts/` | CPU benchmark, RAG mode comparison, OKF bundle validators |
| `tests/` | pytest suite (`pip install -r requirements-dev.txt`) |
| `docs/` | Generated documentation, including `rag_mode_comparison.md` |

## Running locally

```bash
python -m pip install -r requirements.txt   # CPU-only wheels, Python 3.11
streamlit run app.py
```

The checkpoints live in `models/` and are git-ignored (~1.2 GB). Either restore
them there yourself, or point `HF_MODEL_REPO` at a Hugging Face Hub model
repository that contains them and let `models_loader.py` fetch them on first
start. See `.env.example` for every optional setting.

```bash
python -m pytest tests -q                    # needs requirements-dev.txt
python scripts/benchmark_cpu.py              # per-model latency and peak RSS
python scripts/compare_rag_modes.py          # regenerates docs/rag_mode_comparison.md
```

## Deploy

The app is built for a **free, CPU-only host with no GPU** — Streamlit Community
Cloud is the intended target. There is no Docker image and no GPU code path.

### Streamlit Community Cloud settings

| Setting | Value |
| --- | --- |
| Repository | this repository |
| Branch | `main` |
| Main file path | `app.py` |
| Python version | `3.11` |
| Requirements file | `requirements.txt` (auto-detected, leave as-is) |
| Package manager | leave empty — there is deliberately no `packages.txt`, see below |
| Secrets | add the keys listed below, one per entry |

`.streamlit/config.toml` is committed and already sets `headless = true` and
`gatherUsageStats = false`, so nothing else needs configuring. It also disables
the file watcher, because `models/` holds ~1.2 GB of weights that never change.

### Secrets

Every setting is read from the environment first and from `st.secrets` second,
so on Community Cloud each secret must be a **top-level** key (a key nested in a
`[table]` is exported as `TABLE_KEY`, not `KEY`):

```toml
HF_MODEL_REPO = "your-hugging-face-username/alz-mri-weights"
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o-mini"
```

| Secret | Required | Purpose |
| --- | --- | --- |
| `HF_MODEL_REPO` | yes | Hub repository holding the nine checkpoints, when they are not committed locally |
| `HF_TOKEN` | private repos only | Read token for that repository. Never logged or printed by the app |
| `OPENAI_API_KEY` | no | Enables LLM-written reports. Without it the app still runs and returns the deterministic template report |
| `OPENAI_BASE_URL` | no | Any OpenAI-compatible endpoint; defaults to `https://api.openai.com/v1` |
| `LLM_MODEL` | no | Chat model for the report |
| `EMBEDDING_MODEL` | no | Informational: the deployed app always builds the index with the local hashing embedder, so the index is identical on every host |

Optional overrides, same format: `MODELS_DIR`, `HF_MODEL_REVISION` (default
`main`), `RAG_MODE` (`okf_rag` | `rag_only` | `okf_only`), `RAG_MAX_CONCEPTS`,
`TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RAG_MIN_SCORE`, `LLM_REPORT_CALL_CAP`.

### What the first start does

1. Resolves the checkpoints: anything already in `models/` is used untouched,
   anything missing is downloaded from `HF_MODEL_REPO` into a staging directory
   and moved into place with a single atomic rename. A Space that is hibernated
   mid-download never leaves a truncated `.pth` behind — the partial file is
   discarded and fetched again.
2. Writes the weights to `./models` if that directory is writable, and falls
   back to a temp directory if it is not.
3. Builds the FAISS index from `knowledge/okf_bundle` (41 chunks) with the
   deterministic hashing embedder — no API key and no network needed.
4. Loads ResNet-152 → VGG16 → EfficientNet-B4 → ViT → meta-learner → clinical
   branch → fusion, one at a time, each pinned to `map_location="cpu"` with the
   transient `state_dict` dropped and collected before the next one is read.
   The process RSS is logged after every load and shown in the **System info**
   expander.

Expect a slow first start (the weights download) and treat the free tier's
~2.7 GB memory cap as the binding constraint. Measured on Linux CPython 3.11,
6 threads: **1.81 GB peak** while loading, **~1.55 GB resident** once all seven
models are loaded, and **~1.3 s** per prediction. The **System info** expander
and the `models_loader: RSS after ...` log lines are how you watch it on the
deployed Space. If the Space is killed on load, the fix is fewer resident
models, not a bigger host.

### Why there is no `packages.txt`

Nothing in the dependency set needs a system library that the base image lacks.
The PyTorch CPU wheels (`torch==2.0.1+cpu`) bundle their own OpenMP runtime, and
FAISS, scikit-learn, SciPy and Pillow all ship manylinux wheels that carry what
they need — which is exactly why `requirements.txt` pins versions that publish
`cp311` manylinux / abi3 / `py3-none-any` wheels. Add `packages.txt` with
`libgomp1` only if the app log shows `ImportError: libgomp.so.1: cannot open
shared object file`, and note that `libgomp1` must also be installed locally for
a Linux test to reproduce the same environment.
