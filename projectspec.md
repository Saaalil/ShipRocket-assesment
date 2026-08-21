# Project Spec — Shiprocket Turn Detection

## Product

Audio-native binary classifier that runs **after VAD detects a pause** and estimates `P(turn is complete)`.

- Complete → assistant may respond  
- Incomplete → keep listening  

Target setting: Indian English / Hindi / Hinglish-like conversational speech (short replies, fillers, mid-thought pauses).

## Links

| Artifact | URL |
| --- | --- |
| Code | https://github.com/Saaalil/ShipRocket-assesment |
| Demo | https://huggingface.co/spaces/Saalil/Assesment_SR |
| Model | https://huggingface.co/Saalil/Assesment_SR-model |

## Inspiration

- **OpenAI GPT Live** — interruption vs delay are different failures; live path must stay small; full-duplex large models are out of scope for this assignment.  
  https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- **Pipecat Smart Turn v3.2** — Whisper Tiny encoder + attention pooling + compact head after VAD.  
  https://github.com/pipecat-ai/smart-turn
- **Whisper** (Radford et al.) — multilingual encoder reused via transfer learning.
- Turn-taking literature (Sacks / Schegloff / Jefferson): silence ≠ turn end.
- ONNX Runtime quantization practice: ship FP32 first; INT8 only after real calibration + parity.

## Constraints

- Train only on official `pipecat-ai/smart-turn-data-v3.2-train`.
- No extra Hinglish collection.
- Free Colab T4 compute.
- Public test set unused for training and threshold selection.
- Apache-2.0.

## Data

- Subset: **84,223** English + Hindi/Marathi clips  
- Validation: **8,422** clips  
- Metrics file: `reports/partial_unfreeze_eval.json`

## Architecture

1. Mono → 16 kHz → peak-normalize → last 8 s → Whisper log-Mel **80×800**  
2. Whisper Tiny **encoder only**  
3. Attention pooling  
4. MLP head → sigmoid `P(complete)`  

## Training (final)

Config: `configs/partial_unfreeze.yaml`

| Setting | Value |
| --- | --- |
| Unfrozen encoder layers | 2 |
| Head LR / encoder LR | 5e-5 / 8e-6 |
| Effective batch | 128 (8 × 16) |
| Precision | FP16 |
| Epochs | **2** (1,444 steps) |
| Hardware | Colab T4 |
| Wall time | ~4 h 51 m |

### Why 2 epochs

One partial-unfreeze run already cost ~5 hours on free Colab. Macro F1 improved only **0.42 pp** from epoch 1.66 → 2.00. Further epochs were deferred due to diminishing returns, disconnect risk, and the need to spend remaining compute on export and demo.

### Validation @ threshold 0.50

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6178 |
| Macro F1 | 0.6174 |
| ROC-AUC | 0.6697 |
| Complete F1 | 0.6288 |
| Incomplete F1 | 0.6060 |

Default product threshold: **0.50**. Conservative agent setting to evaluate: **0.55–0.60**.

## Export

| Artifact | Size | Status |
| --- | ---: | --- |
| FP32 ONNX | ~32.1 MB | **Served in demo** |
| INT8 candidate | ~8.29 MB | Not validated (needs real calibration) |

Serve FP32 until INT8 passes parity on validation.

## Repo layout

```text
projectspec.md          # this file (sole project document)
app.py                  # Hugging Face Space entry
pyproject.toml          # package + deps
LICENSE
configs/
  data.yaml
  partial_unfreeze.yaml
src/smart_turn/         # model, train, eval, export, inference
scripts/                # CLI: train, export, evaluate, publish, bench
demo/                   # Gradio + optional static web demo
spaces/                 # HF Space metadata + requirements
notebooks/
  02_train_colab.ipynb
  04_save_publish_demo_colab.ipynb
reports/
  partial_unfreeze_eval.json
model_card/README.md    # HF model card payload
```

## Quickstart

```bash
python -m pip install -e ".[demo,train]"
python demo/gradio_app.py
```

Train on Colab: `notebooks/02_train_colab.ipynb`  
Export + publish: `notebooks/04_save_publish_demo_colab.ipynb`

```bash
python scripts/train.py --config configs/partial_unfreeze.yaml
python scripts/export_onnx.py --config configs/partial_unfreeze.yaml --checkpoint path/to/final_model
python scripts/evaluate.py --config configs/partial_unfreeze.yaml --checkpoint path/to/final_model
```

## Scope honesty

This is a compact audio-native endpoint detector, not GPT Live, not ASR, and not a claim of production barge-in quality on Shiprocket traffic. It is a reproducible proof of concept under free-compute and official-data constraints.
