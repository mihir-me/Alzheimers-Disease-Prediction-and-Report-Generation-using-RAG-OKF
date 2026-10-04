#!/usr/bin/env python3
"""Cold-start, peak-RAM and per-prediction latency benchmark for the CPU app.

Measures exactly what the Streamlit app does, in the same order, on the CPU
device the app pins:

1. artifact resolution (``models_loader``: local checkpoints, Hub only if any
   are missing) and the load of all four MRI backbones plus the meta-learner,
   clinical branch and fusion head;
2. the FAISS knowledge-base index build;
3. one full prediction on a real OASIS test slice, split per model so it is
   obvious which backbone dominates.

Run it with the same interpreter the app uses::

    python scripts/benchmark_cpu.py
    python scripts/benchmark_cpu.py --threads 2      # emulate a 2-vCPU Space
    python scripts/benchmark_cpu.py --repeats 5 --image path/to/slice.jpg

Nothing here changes weights or precision; it only reads them.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MB = 1024 * 1024
GB = 1024 * MB


class PeakMemory:
    """Samples process RSS on a background thread while the app runs."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._proc = psutil.Process()
            self._rss = lambda: self._proc.memory_info().rss
            self.backend = "psutil"
        except ImportError:  # pragma: no cover - psutil is in requirements-dev
            self._rss = self._rss_from_proc
            self.backend = "procfs"

    def _rss_from_proc(self) -> int:
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self.peak = self._rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak = max(self.peak, self._rss())

    @property
    def rusage_peak(self) -> int | None:
        """Kernel-reported peak RSS. Linux only, and authoritative when present."""
        try:
            import resource
        except ImportError:  # pragma: no cover - Windows
            return None
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform != "darwin" else peak // 1024


def rss_now() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0


def pick_test_image(explicit: str | None) -> Path:
    """A real OASIS test slice, so the number is not from a synthetic tensor."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--image does not exist: {path}")
        return path

    manifest_path = REPO_ROOT / "results" / "split_manifest.json"
    datasets = REPO_ROOT / "datasets" / "OASIS DATASET" / "Data"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entries in manifest.get("test", {}).values():
            for rel in entries:
                candidate = datasets / rel
                if candidate.is_file():
                    return candidate
    for candidate in sorted(datasets.rglob("*.jpg")):
        return candidate
    raise FileNotFoundError(
        "no test image found; the OASIS dataset is not present, pass --image"
    )


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def measure_cold_start(memory: PeakMemory) -> tuple[dict, dict]:
    """Resolve artifacts, load every model, build the FAISS index."""
    import app
    from models_loader import ensure_artifacts

    steps: dict[str, float] = {}
    timings: dict[str, float] = {}

    start = time.perf_counter()
    ensure_artifacts()
    steps["artifacts"] = time.perf_counter() - start

    start = time.perf_counter()
    models_dict = app.load_all_models()
    steps["models"] = time.perf_counter() - start
    timings["load_models"] = steps["models"]
    gc.collect()

    start = time.perf_counter()
    service = app.get_rag_service()
    steps["faiss_index"] = time.perf_counter() - start
    timings["build_faiss_index"] = steps["faiss_index"]

    return models_dict, {
        **steps,
        "total": sum(steps.values()),
        "rss_after_load": rss_now(),
        "peak_rss": memory.peak,
        "rusage_peak_rss": memory.rusage_peak,
        "memory_backend": memory.backend,
        "faiss_chunks": service.store.n_total,
        "faiss_rebuilt": service.index_rebuilt,
        "embedding_provider": service.embedder.provider,
    }


def per_model_latency(models_dict: dict, image: Path, repeats: int) -> dict:
    """Forward-pass time per backbone, plus the whole pipeline."""
    import numpy as np
    import torch

    transforms = models_dict["transforms"]
    device = models_dict["device"]
    from PIL import Image

    rgb = Image.open(image).convert("RGB")
    tensors = {
        224: transforms[224](rgb).unsqueeze(0).to(device),
        380: transforms[380](rgb).unsqueeze(0).to(device),
    }
    plan = [
        ("ResNet-152", models_dict["resnet"], 224),
        ("VGG16", models_dict["vgg"], 224),
        ("EfficientNet-B4", models_dict["efficient"], 380),
        ("Vision Transformer", models_dict["vit"], 224),
    ]
    samples: dict[str, list[float]] = {name: [] for name, _, _ in plan}
    samples["Clinical branch"] = []
    samples["Fusion head"] = []

    clinical = torch.tensor([[0.8, -1.0, -0.4]], dtype=torch.float32, device=device)
    fusion = torch.tensor([np.full(16, 0.25, dtype=np.float32).tolist() + [0.8, -1.0, -0.4]],
                          dtype=torch.float32, device=device).to(device)

    with torch.inference_mode():
        for _ in range(repeats):
            for name, model, size in plan:
                start = time.perf_counter()
                torch.softmax(model(tensors[size]), dim=1)
                samples[name].append(time.perf_counter() - start)
            start = time.perf_counter()
            models_dict["clinical_model"](clinical)
            samples["Clinical branch"].append(time.perf_counter() - start)
            start = time.perf_counter()
            models_dict["fusion_model"](fusion)
            samples["Fusion head"].append(time.perf_counter() - start)

    return {name: values for name, values in samples.items() if values}


def measure_prediction(models_dict: dict, image: Path, repeats: int) -> dict:
    """End-to-end latency of one 'Get Prediction' click, template report included."""
    import app

    mmse, cdr, nwbv = 24, 0.5, 0.73
    query = (
        "What do these results suggest about this patient's cognitive status "
        "and what should we do next?"
    )

    start = time.perf_counter()
    results, _ = app.predict_alzheimers(models_dict, image, mmse, cdr, nwbv)
    first = time.perf_counter() - start

    payload = app.build_report_payload(results, mmse, cdr, nwbv)
    service = app.get_rag_service()
    report_start = time.perf_counter()
    output = service.generate(payload, query, allow_llm=False)
    first_report = time.perf_counter() - report_start

    predict_times: list[float] = []
    report_times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        results, _ = app.predict_alzheimers(models_dict, image, mmse, cdr, nwbv)
        predict_times.append(time.perf_counter() - start)

        payload = app.build_report_payload(results, mmse, cdr, nwbv)
        start = time.perf_counter()
        service.generate(payload, query, allow_llm=False)
        report_times.append(time.perf_counter() - start)

    return {
        "first_prediction": first,
        "first_report": first_report,
        "predict_median": statistics.median(predict_times),
        "predict_min": min(predict_times),
        "predict_max": max(predict_times),
        "report_median": statistics.median(report_times),
        "click_median": statistics.median(predict_times) + statistics.median(report_times),
        "stage": results["Ensemble"]["class"],
        "diagnosis": results["Final"]["diagnosis"],
        "concepts": len(output.get("concepts") or []),
        "used_llm": output["used_llm"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="path to one MRI slice (default: an OASIS test slice)")
    parser.add_argument("--repeats", type=int, default=3, help="timed warm runs (default 3)")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads; 0 keeps the default. Use 2 to emulate a free Space.",
    )
    args = parser.parse_args()

    if args.threads > 0:
        import torch

        torch.set_num_threads(args.threads)

    banner("ENVIRONMENT")
    import torch

    import app

    print(f"platform            : {platform.platform()}")
    print(f"python              : {platform.python_version()}")
    print(f"torch               : {torch.__version__}")
    print(f"logical CPUs        : {os.cpu_count()}")
    print(f"torch threads       : {torch.get_num_threads()}")
    print(f"device pinned by app: {app.DEVICE}")
    print(f"cuda available      : {torch.cuda.is_available()} (ignored on purpose)")

    image = pick_test_image(args.image)
    print(f"test image          : {image.relative_to(REPO_ROOT) if image.is_relative_to(REPO_ROOT) else image}")

    with PeakMemory() as memory:
        banner("COLD START (first process, warm page cache excluded)")
        models_dict, cold = measure_cold_start(memory)

        print(f"resolve artifacts   : {cold['artifacts']:8.2f} s")
        print(f"load 7 checkpoints  : {cold['models']:8.2f} s")
        print(f"build FAISS index   : {cold['faiss_index']:8.2f} s "
              f"({cold['faiss_chunks']} chunks, rebuilt={cold['faiss_rebuilt']}, "
              f"embeddings={cold['embedding_provider']})")
        print(f"TOTAL cold start    : {cold['total']:8.2f} s")

        banner("MEMORY")
        rusage = cold["rusage_peak_rss"]
        print(f"backend             : {cold['memory_backend']}")
        print(f"RSS after load      : {cold['rss_after_load'] / GB:8.2f} GB "
              f"({cold['rss_after_load'] / MB:,.0f} MB)")
        print(f"peak RSS sampled    : {cold['peak_rss'] / GB:8.2f} GB "
              f"({cold['peak_rss'] / MB:,.0f} MB)")
        if rusage:
            print(f"peak RSS (getrusage): {rusage / GB:8.2f} GB ({rusage / MB:,.0f} MB)")

        banner("PER-MODEL FORWARD PASS (batch of 1, inference_mode)")
        stages = per_model_latency(models_dict, image, args.repeats)
        total_stage = 0.0
        for name, values in stages.items():
            median = statistics.median(values)
            total_stage += median
            print(f"{name:22s} median {median * 1000:8.1f} ms   "
                  f"(min {min(values) * 1000:7.1f} / max {max(values) * 1000:7.1f})")
        print(f"{'sum of forward passes':22s}        {total_stage * 1000:8.1f} ms")

        banner("END-TO-END PREDICTION (report generation included, no LLM)")
        pred = measure_prediction(models_dict, image, args.repeats)
        print(f"first (cold) predict: {pred['first_prediction']:8.2f} s")
        print(f"warm predict median : {pred['predict_median']:8.2f} s "
              f"(min {pred['predict_min']:.2f} / max {pred['predict_max']:.2f})")
        print(f"warm report median  : {pred['report_median']:8.2f} s "
              f"({pred['concepts']} concepts, used_llm={pred['used_llm']})")
        print(f"warm click median   : {pred['click_median']:8.2f} s")
        print(f"prediction          : stage={pred['stage']} diagnosis={pred['diagnosis']}")

        peak = max(cold["peak_rss"], rusage or 0)
        banner("VERDICT AGAINST THE 12 GB / 20 s BUDGET")
        print(f"peak RAM            : {peak / GB:.2f} GB  (budget 12 GB) -> "
              f"{'OK' if peak <= 12 * GB else 'OVER'}")
        print(f"per-prediction      : {pred['click_median']:.2f} s   (budget 20 s) -> "
              f"{'OK' if pred['click_median'] <= 20 else 'OVER'}")
        print(f"cold start          : {cold['total']:.2f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())