from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np

from smart_turn.constants import DEFAULT_THRESHOLD
from smart_turn.inference import predict_turn

HF_MODEL_REPO = "Saalil/Assesment_SR-model"

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(function=None, **_kwargs):
            if function is not None:
                return function

            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()

EXAMPLES = [
    "A pause is not the same as a finished thought.",
    "Complete: short replies like 'haan, that works.'",
    "Incomplete: trail-offs like 'haan, but actually...'",
]


def _model_label() -> str:
    raw = os.environ.get("SMART_TURN_ONNX_PATH", "").strip()
    env = Path(raw) if raw else None
    if env and env.exists():
        return f"Serving `{env.name}`"
    for local in (Path("artifacts/model_fp32.onnx"), Path("artifacts/model_int8.onnx")):
        if local.exists():
            return f"Serving `{local}`"
    return f"Will download `{HF_MODEL_REPO}` (or the official Pipecat fallback if that repo is empty)."


def _to_waveform(audio) -> tuple[np.ndarray, int]:
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        data, sample_rate = sf.read(str(audio), always_2d=False)
        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, int(sample_rate)
    sample_rate, data = audio
    waveform = np.asarray(data, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.dtype == np.int16 or np.max(np.abs(waveform)) > 1.5:
        waveform = waveform / 32768.0
    return waveform, int(sample_rate)


@spaces.GPU(duration=60)
def _run(audio, threshold: float) -> str:
    if audio is None:
        return "Record or upload audio first."
    waveform, sample_rate = _to_waveform(audio)
    prediction = predict_turn(waveform, sample_rate, threshold=float(threshold))
    label = "COMPLETE" if prediction["is_complete"] else "INCOMPLETE"
    return (
        f"{label}\n"
        f"probability={prediction['probability_complete']:.3f}\n"
        f"threshold={prediction['threshold']:.2f}\n"
        f"inference={prediction['inference_ms']:.1f} ms"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Shiprocket Turn Detection") as demo:
        gr.Markdown(
            "# Shiprocket turn detection\n"
            "Audio-native **complete / incomplete** classifier. "
            "Run it after VAD sees a pause, not as a speech/silence detector.\n\n"
            "Trained on official Smart Turn v3.2 English + Hindi/Marathi. "
            "Validation: accuracy **0.618**, macro-F1 **0.617**, ROC-AUC **0.670**."
        )
        gr.Markdown(_model_label())
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Utterance")
            threshold = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.05, label="Threshold")
        output = gr.Textbox(label="Decision", lines=5)
        gr.Button("Predict").click(_run, [audio, threshold], output, api_name=False)
        gr.Markdown("\n".join(f"- {item}" for item in EXAMPLES))
    return demo


if __name__ == "__main__":
    import os

    os.environ["GRADIO_SSR_MODE"] = "0"
    Path("artifacts").mkdir(exist_ok=True)
    build_demo().launch(ssr_mode=False, share=False)
