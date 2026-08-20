from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from smart_turn.constants import N_FRAMES, N_MELS
from smart_turn.inference import resolve_model_path


def benchmark_onnx(model_path: str | None = None, repeats: int = 50) -> dict[str, float]:
    import onnxruntime as ort

    path = resolve_model_path(model_path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, N_MELS, N_FRAMES), dtype=np.float32)
    session.run(None, {input_name: dummy})
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(times)
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    return {
        "model_path": path,
        "size_mb": float(size_mb),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": float(array.mean()),
        "repeats": float(repeats),
    }
