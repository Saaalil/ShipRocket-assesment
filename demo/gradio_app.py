from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np

from smart_turn.constants import DEFAULT_THRESHOLD
from smart_turn.inference import predict_turn

EXAMPLES = [
    "A pause is not the same as a finished thought.",
    "The model should say complete for: 'haan, that works.'",
    "The model should say incomplete for: 'haan, but actually...'",
]


def _run(audio, threshold: float) -> str:
    if audio is None:
        return "Record or upload audio first."
    sample_rate, data = audio
    waveform = np.asarray(data, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if waveform.dtype == np.int16 or waveform.max() > 1.5:
        waveform = waveform / 32768.0
    prediction = predict_turn(waveform, int(sample_rate), threshold=float(threshold))
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
            "# Shiprocket Turn Detection\n"
            "Audio-native complete/incomplete classifier. "
            "Intended to run after VAD sees a pause, not as a speech/silence detector."
        )
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Utterance")
            threshold = gr.Slider(0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.05, label="Threshold")
        output = gr.Textbox(label="Decision", lines=5)
        gr.Button("Predict").click(_run, [audio, threshold], output)
        gr.Markdown("\n".join(f"- {item}" for item in EXAMPLES))
    return demo


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    build_demo().launch()
