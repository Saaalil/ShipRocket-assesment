from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

from smart_turn.audio import prepare_audio
from smart_turn.constants import (
    DEFAULT_THRESHOLD,
    HF_MODEL_REPO,
    HF_ONNX_FILENAME,
    SAMPLE_RATE,
)
from smart_turn.features import extract_log_mel


class TurnPrediction(TypedDict):
    probability_complete: float
    is_complete: bool
    threshold: float
    audio_duration_seconds: float
    inference_ms: float


def _load_onnx_session(model_path: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _download_hf_onnx(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_model_path(explicit: str | None = None, allow_official_fallback: bool = True) -> str:
    if explicit:
        return explicit
    import os

    raw = os.environ.get("SMART_TURN_ONNX_PATH", "").strip()
    if raw:
        env_path = Path(raw)
        if env_path.exists():
            return str(env_path)
    for local in (Path("artifacts/model_fp32.onnx"), Path("artifacts/model_int8.onnx")):
        if local.exists():
            return str(local)
    assigned_repo = os.environ.get("SMART_TURN_HF_REPO", HF_MODEL_REPO)
    assigned_file = os.environ.get("SMART_TURN_HF_ONNX", HF_ONNX_FILENAME)
    try:
        return _download_hf_onnx(assigned_repo, assigned_file)
    except Exception:
        pass
    if allow_official_fallback:
        return _download_hf_onnx("pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx")
    raise FileNotFoundError(
        "No ONNX model found. Train/export first, publish to Hugging Face, "
        "or set SMART_TURN_ONNX_PATH."
    )


def predict_turn(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float | None = None,
    model_path: str | None = None,
) -> TurnPrediction:
    prepared = prepare_audio(audio, sample_rate)
    features = extract_log_mel(prepared, SAMPLE_RATE)[None, ...]
    session = _load_onnx_session(resolve_model_path(model_path))
    input_name = session.get_inputs()[0].name
    started = time.perf_counter()
    outputs = session.run(None, {input_name: features.astype(np.float32)})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    probability = float(np.asarray(outputs[0]).reshape(-1)[0])
    cutoff = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    return {
        "probability_complete": probability,
        "is_complete": probability >= cutoff,
        "threshold": cutoff,
        "audio_duration_seconds": float(prepared.shape[0] / SAMPLE_RATE),
        "inference_ms": elapsed_ms,
    }


def predict_file(
    path: str,
    threshold: float | None = None,
    model_path: str | None = None,
) -> TurnPrediction:
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=False)
    return predict_turn(np.asarray(audio), int(sample_rate), threshold, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run turn detection on a wav file")
    parser.add_argument("audio")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    prediction = predict_file(args.audio, args.threshold, args.model)
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
