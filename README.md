# Shiprocket Turn Detection

Tiny audio-native classifier that decides whether a paused speaker has **finished their turn** or is only hesitating.

This is the Shiprocket open Data Scientist assignment: a small, fast, CPU-friendly model trained only on the official [Smart Turn v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train) data. Indian-language evidence uses official Hindi/Marathi slices plus English filler/short utterances. No extra recordings.

## What it does

Voice activity detection finds a pause. This model then looks at up to 8 seconds of 16 kHz audio and returns `P(complete)`.

- Too early → interruption
- Too late → sluggish response

## Quickstart

```bash
python -m pip install -e ".[dev,demo,train]"
python -m pytest
python demo/gradio_app.py
```

Before a trained checkpoint exists, the demo falls back to the public Pipecat Smart Turn v3.2 ONNX model.

## Train on free Colab

1. Open `notebooks/02_train_colab.ipynb`
2. Mount Drive for checkpoints
3. Start with `configs/head_only.yaml`, then `configs/partial_unfreeze.yaml`

Do not tune on `pipecat-ai/smart-turn-data-v3.2-test`. That split is release-only.

## Repository map

| Path | Role |
| --- | --- |
| `src/smart_turn/` | Model, data, train, eval, export, inference |
| `configs/` | Experiment settings |
| `scripts/` | CLI entry points |
| `demo/` | Gradio + static browser demo |
| `STAFF_ENGINEERING_PLAN.md` | Full engineering contract |
| `OWNER_STEPS.md` | Manual GitHub / Hugging Face / Colab steps |

## Links

- Code: https://github.com/Saaalil/ShipRocket-assesment
- Demo Space: https://huggingface.co/spaces/Saalil/Assesment_SR
