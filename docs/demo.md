# Demo

## Local Gradio

```bash
python demo/gradio_app.py
```

Microphone and file upload are supported. The slider is the completeness threshold.

## Static browser demo

Open `demo/web/index.html` via any static host. Inference runs in the browser with ONNX Runtime Web. Python Gradio remains the numerically matched reference because browser log-mel is approximate.

## Space

https://huggingface.co/spaces/Saalil/Assesment_SR

See `OWNER_STEPS.md` to push the Gradio app into that Space.
