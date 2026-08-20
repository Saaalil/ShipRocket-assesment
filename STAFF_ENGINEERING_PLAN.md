# Smart Turn Detection — Staff Engineering Development Plan

## 0. Document purpose

This document is the implementation contract for an AI coding agent or engineer building the Shiprocket turn-detection assignment. It is intentionally explicit about scope, architecture, experiments, interfaces, quality gates, expected runtime, free-service constraints, deliverables, and the decisions that must not be improvised.

The implementation agent should:

1. Treat this document as the source of truth.
2. Record deviations in `docs/decisions.md` before implementing them.
3. Keep every experiment reproducible from configuration.
4. Never tune against the public test set.
5. Prefer a smaller, defensible, reproducible result over an unverified complex model.

No implementation exists at the time this plan was written.

---

## 1. Executive summary

### 1.1 Problem statement

Build a tiny audio-native binary classifier that runs when a voice activity detector observes a pause and predicts:

- `1 / complete`: the user has finished their conversational turn and the agent may respond.
- `0 / incomplete`: the user is pausing, hesitating, using a filler, or otherwise intends to continue.

The detector must be particularly credible on:

- Indian English and Hindi-English code-switching (“Hinglish”).
- Mid-thought fillers and hesitation.
- End fillers.
- Short but complete responses such as “yes”, “haan”, “okay”, and “no”.
- Background noise and ordinary microphones.
- Pauses that are acoustically silent but semantically incomplete.

### 1.2 Product objective

Produce the strongest assignment submission under free-compute constraints, not an academic model with unbounded compute. The submission must demonstrate:

- Sound data preparation.
- Reproducible experiments and ablations.
- Balanced treatment of premature interruption and delayed response.
- A small, fast CPU-deployable model.
- A public model artifact and a working demonstration.
- A report that clearly separates measured results from assumptions.

### 1.3 Recommended solution

Use an audio-native Whisper Tiny encoder with attention pooling and a compact classification head, initialized from open pretrained weights. Establish the official Pipecat Smart Turn v3.2 model as the external baseline. Train in stages:

1. Reproduce baseline evaluation.
2. Train only the new head while the encoder is frozen.
3. Unfreeze the top encoder blocks for efficient adaptation.
4. Fine-tune the best candidate on targeted Indian-English/Hinglish hard cases.
5. Calibrate the output threshold.
6. Export to ONNX and statically quantize to INT8.

The primary public demo should be a static, client-side browser application using ONNX Runtime Web. This avoids paid inference hosting. A local Gradio demo must also be included.

---

## 2. Evidence and fixed assumptions

### 2.1 Supplied data

Dataset: `pipecat-ai/smart-turn-data-v3.2-train`

Verified public metadata:

- 270,946 training rows.
- Approximately 41.4 GB.
- One published `train` split.
- Fields: `audio`, `id`, `language`, `endpoint_bool`, `midfiller`, `endfiller`, `synthetic`, `spoken_text`, and `dataset`.
- The public test dataset contains 31,527 rows and is approximately 4.84 GB.
- The data supports 23 languages, including English, Hindi, and Marathi.
- Hinglish is not documented as an independent language category.

### 2.2 Published baseline

The current Pipecat training implementation uses:

- `openai/whisper-tiny`.
- Whisper encoder only, configured for 8-second inputs.
- Attention pooling over encoder time steps.
- A compact MLP binary-classification head.
- Binary cross-entropy with logits.
- Four training epochs in the published configuration.
- Up to 8 seconds of 16 kHz mono PCM audio.
- VAD as a trigger: Smart Turn runs after silence is detected.
- FP32 and statically quantized INT8 ONNX exports.

The implementation agent must inspect and pin an exact upstream commit before reusing code. Do not silently track `main`.

### 2.3 Interpretation of “from scratch”

Assume “from scratch” means building the complete solution and experimentation pipeline, not training an audio encoder from random initialization. This is supported by the assignment’s explicit suggestion to start with Whisper Tiny and a classification head.

### 2.4 OpenAI research lesson

The OpenAI GPT-Live article removes a standalone detector by using a large full-duplex voice model. That architecture is outside this assignment. Relevant lessons for this project are:

- Premature interruption and delayed response are different failure modes.
- End-to-end conversational latency matters more than isolated model latency.
- The live audio path must remain lightweight.
- Stateful audio context, prosody, pace, and hesitation are important.
- Deeper work should stay off the latency-critical audio path.

The OpenAI article is systems guidance, not a training recipe for this classifier.

### 2.5 Fixed constraints

- Primary goal: a strong Shiprocket assignment submission.
- Error preference: balanced interruption and delay errors.
- Training budget: free Google Colab where possible.
- Paid APIs, paid GPU services, and mandatory paid hosting are prohibited.
- **v1 data scope is locked:** train and evaluate only on the official Smart Turn v3.2 train/test datasets. Do not collect extra Hinglish recordings unless a later iteration is explicitly approved.
- Indian-language evidence for v1 comes from official `language` slices, especially English, Hindi, and Marathi, plus filler/short-utterance slices.
- Public artifacts should use permissive/open components with documented licenses.
- The final model must run on commodity CPU.

### 2.6 Locked destinations

| Artifact | Destination |
| --- | --- |
| Source code | https://github.com/Saaalil/ShipRocket-assesment.git |
| Public demo Space | https://huggingface.co/spaces/Saalil/Assesment_SR |
| Model repo | Create later as a public Hugging Face **model** repo, not the Space |

The GitHub repository currently has no commits. The Space is a default Gradio template. Hugging Face compute Spaces may require a paid plan; if the Space cannot run, ship GitHub Pages static demo plus local Gradio as the guaranteed free path.

---

## 3. Scope

### 3.1 In scope

- Dataset profiling and validation.
- Leakage-aware local train/validation splits.
- Official baseline evaluation.
- Whisper Tiny encoder experiments.
- Compact classification heads.
- Indian-language slices constructed from the official dataset.
- Noise and channel augmentation.
- Calibration and threshold selection.
- Error analysis and ablations.
- ONNX FP32 and INT8 export.
- CPU benchmarking.
- Browser and local demos.
- Hugging Face model repository.
- GitHub source repository.
- Human-readable technical report.

### 3.2 Out of scope

- Training a full speech foundation model from random initialization.
- Building a general full-duplex speech-to-speech model.
- Paid transcription, TTS, inference, storage, or GPU APIs.
- Production telephony integrations.
- Real customer voice collection.
- Claiming production readiness from clip-level metrics alone.
- Uploading the full 41.4 GB upstream dataset into the project repository.

---

## 4. Success criteria and quality gates

### 4.1 Primary model-selection criteria

Candidates must be selected using a local validation set, not the public test set.

The default selection score should combine:

```text
selection_score =
    0.40 * macro_f1
  + 0.25 * complete_recall
  + 0.25 * incomplete_recall
  + 0.10 * indic_official_slice_f1
```

This score is a project convention, not an author-provided metric. The report must also publish all underlying metrics so reviewers can apply different priorities.

### 4.2 Required classification metrics

- Accuracy.
- Macro F1.
- Precision, recall, and F1 for each class.
- Confusion matrix.
- False-complete rate: incomplete speech predicted complete. This approximates interruption risk.
- False-incomplete rate: complete speech predicted incomplete. This approximates response-delay risk.
- ROC-AUC and PR-AUC.
- Expected calibration error or Brier score.
- Metrics at the chosen threshold and at threshold `0.5`.

### 4.3 Required slices

- Language.
- Dataset/source.
- Synthetic versus human.
- Mid-filler.
- End-filler.
- Duration buckets: `<1s`, `1–3s`, `3–5s`, `5–8s`.
- Clean versus augmented/noisy.
- Short complete responses.
- Indian English.
- Hinglish/code-switched speech.

Do not report a slice with fewer than 30 examples without clearly marking it as directional.

### 4.4 Deployment gates

The final candidate should meet:

- INT8 ONNX artifact ideally no larger than 15 MB; hard ceiling 30 MB.
- Single-clip CPU latency:
  - target p50 under 50 ms;
  - target p95 under 100 ms;
  - measured after warm-up;
  - measured on at least one ordinary laptop CPU.
- Peak demo memory below 500 MB.
- No server-side inference dependency for the primary public demo.
- Bitwise reproducible preprocessing within reasonable floating-point tolerance.
- FP32-to-INT8 macro-F1 regression no greater than 1.5 percentage points unless the size/latency trade-off is explicitly justified.

These are engineering targets inferred from the post and published Smart Turn behavior, not official hidden-test requirements.

### 4.5 Submission gates

The submission is not complete until a fresh machine can:

1. Install dependencies from pinned files.
2. Run one smoke-test prediction.
3. Reproduce evaluation on a small fixture set.
4. Launch the local demo.
5. Download the published model.
6. View the static demo without an API key.

---

## 5. System architecture

### 5.1 Runtime flow

```text
Microphone / uploaded audio
        |
        v
16 kHz mono normalization
        |
        v
VAD observes speech and then a pause
        |
        v
Take the latest <= 8 seconds of the current user turn
Left-pad shorter clips with zeros
        |
        v
80-bin Whisper log-Mel feature extraction
        |
        v
Whisper Tiny encoder
        |
        v
Attention pooling + compact MLP head
        |
        v
Calibrated completion probability
        |
        v
Threshold policy
  complete -> allow response
  incomplete -> continue listening
```

### 5.2 Separation of responsibilities

- VAD answers: “Is speech currently present?”
- Turn detector answers: “Given that speech paused, is the user’s thought complete?”
- Threshold policy answers: “At what confidence should this application respond?”
- The demo visualizes each stage separately to prevent conflating VAD and semantic endpointing.

### 5.3 Model interface

Canonical Python inference interface:

```python
class TurnPrediction(TypedDict):
    probability_complete: float
    is_complete: bool
    threshold: float
    audio_duration_seconds: float
    inference_ms: float

def predict_turn(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float | None = None,
) -> TurnPrediction:
    ...
```

Canonical ONNX input and output:

- Input name: `input_features`.
- Input dtype: `float32`.
- Input shape: `[batch, 80, 800]`.
- Output name: `probability_complete`.
- Output range: `[0, 1]`.

The export metadata must contain:

- model version;
- git commit;
- training config hash;
- preprocessing version;
- selected threshold;
- class mapping.

---

## 6. Repository design

The implementation agent should create this structure:

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.lock
├── Makefile
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── head_only.yaml
│   ├── partial_unfreeze.yaml
│   └── final.yaml
├── notebooks/
│   ├── 01_data_audit_colab.ipynb
│   ├── 02_train_colab.ipynb
│   └── 03_export_and_publish_colab.ipynb
├── src/smart_turn/
│   ├── __init__.py
│   ├── audio.py
│   ├── data.py
│   ├── splits.py
│   ├── augment.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── export.py
│   ├── inference.py
│   └── benchmark.py
├── demo/
│   ├── gradio_app.py
│   └── web/
│       ├── index.html
│       ├── app.js
│       ├── audio-worklet.js
│       └── styles.css
├── tests/
│   ├── fixtures/
│   ├── test_audio.py
│   ├── test_model.py
│   ├── test_inference.py
│   ├── test_export_parity.py
│   └── test_demo_smoke.py
├── scripts/
│   ├── audit_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── benchmark_cpu.py
│   └── publish_model.py
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── final_report.md
├── model_card/
│   └── README.md
└── docs/
    ├── decisions.md
    ├── data_card.md
    ├── experiment_log.md
    └── demo.md
```

Generated checkpoints, raw data, Hugging Face caches, audio recordings, and secrets must be ignored by Git.

---

## 7. Data engineering plan

### 7.1 Data audit before training

Run an audit over metadata first, without decoding every audio file. Produce:

- Total records and duplicate IDs.
- Class distribution.
- Language distribution.
- Source/dataset distribution.
- Synthetic/human distribution.
- Filler-label distribution by class.
- Cross-tabs for language × class, source × class, and synthetic × class.
- Missing/invalid metadata.

Then decode a deterministic stratified sample and inspect:

- Sample rate and channel count.
- Duration distribution.
- Clipping rate.
- RMS loudness.
- Leading/trailing silence.
- Corrupt or empty audio.
- Duplicate waveforms using a robust audio fingerprint.

Quality gate: training must not begin until label counts, corrupt-file policy, and split strategy are recorded in `docs/data_card.md`.

### 7.2 Split policy

The upstream training set exposes only one split. Create:

- `train`: 90%.
- `validation`: 10%.
- `public_test`: upstream v3.2 test dataset, never used for tuning.
- `challenge_indic`: official Hindi/Marathi plus English filler/short-utterance rows, held out from local training.

Preferred grouping priority:

1. Speaker ID if recoverable.
2. Recording/session ID.
3. Source dataset plus stable ID prefix.
4. Exact/near-duplicate audio fingerprint.

Use grouped stratification across `endpoint_bool`, `language`, `synthetic`, and source where practical.

If speaker/session grouping is impossible, explicitly document the leakage risk. Do not claim the random split measures unseen-speaker generalization.

### 7.3 Label semantics

- `endpoint_bool=True`: complete turn.
- `endpoint_bool=False`: incomplete turn.
- `midfiller` and `endfiller`: metadata/possible auxiliary labels.
- `synthetic`: provenance metadata.

Never feed ground-truth filler flags, language labels, source IDs, or synthetic flags as inference inputs. That would create train-serving mismatch and possible leakage.

### 7.4 Indian-language evaluation from the official dataset only

v1 does **not** collect extra Hinglish recordings. The official corpus is sufficient to train, compare against Smart Turn v3.2, and report Indic performance.

Build an **internal Indic challenge slice** from the official train/test metadata, never from new recordings:

- All Hindi rows.
- All Marathi rows.
- A stratified English subset, emphasizing short utterances, `midfiller=True`, and `endfiller=True`.
- Keep this slice disjoint from the local training split.
- Do not claim true Hinglish/code-switch coverage unless `spoken_text` or language labels actually show mixed Hindi-English. If they do not, report Hindi + Indian-language English as the closest official proxy and state that limitation in the report.

This is enough for the assignment because the author provided this dataset, Hindi is already a published Smart Turn weak slice (~90% on CPU v3.2), and filler/short-response labels already exist. Extra collection is a later upgrade, not a v1 blocker.

### 7.5 Augmentation

Apply augmentations online and probabilistically:

- Background ambience at controlled SNR values.
- Gain variation.
- Mild room impulse response/reverberation.
- Band-limiting and telephone-like filtering.
- Resampling artifacts.
- Light clipping/compression.
- Small temporal shifts.

Do not use transformations that change completion semantics, such as cutting the final word or removing the prosodic ending.

Recommended policy:

- 50% clean.
- 30% one augmentation.
- 20% two compatible augmentations.
- SNR primarily between 10 and 30 dB, with a small 5–10 dB stress slice.

Keep noise sources license-compatible and list them in the data card.

### 7.6 Hard-negative mining

After the first trained model:

1. Score the local validation set.
2. Select high-confidence false completes and false incompletes.
3. Categorize each error.
4. Oversample valid hard cases at no more than 2×.
5. Add targeted Indian/Hinglish examples matching dominant error categories.
6. Retrain once.

Avoid repeated manual tuning on the same validation examples. Preserve a final local holdout.

---

## 8. Model development plan

### 8.1 Baselines

Implement and freeze results for:

#### B0 — pause heuristic

Predict complete after a fixed silence duration. This establishes why semantic turn detection is needed. Evaluate using a replay simulation if timestamps allow; otherwise describe it qualitatively.

#### B1 — official Smart Turn v3.2 INT8

Evaluate the published checkpoint with the project’s exact metrics and preprocessing.

#### B2 — Whisper Tiny reproduction

Reproduce the upstream-compatible encoder, attention pooling, and MLP head. This validates the training and export pipeline.

### 8.2 Candidate experiments

Run experiments in this order and stop when gains no longer justify compute:

#### E1 — head-only adaptation

- Initialize Whisper Tiny encoder.
- Freeze all encoder parameters.
- Train attention pooling and MLP head.
- Purpose: fast pipeline validation and a cheap baseline.
- Expected runtime: 1–3 GPU hours for one full pass, highly dependent on decoding throughput.

#### E2 — partial unfreeze

- Start from E1.
- Unfreeze the top one or two encoder blocks.
- Use a lower encoder learning rate than head learning rate.
- Purpose: adapt high-level acoustic/semantic features without full training cost.
- This is the recommended Colab candidate.

#### E3 — full encoder fine-tuning

- Fine-tune all encoder layers only if E2 underperforms and runtime permits.
- Use mixed precision, gradient clipping, and checkpointed resume.
- Treat this as optional because free Colab is not guaranteed.

#### E4 — auxiliary filler prediction

- Shared encoder and pooled representation.
- Main head: complete/incomplete.
- Auxiliary heads: mid-filler and end-filler.
- Auxiliary labels affect training only.
- Start auxiliary loss weight at `0.1` each.
- Retain only if the primary holdout improves.

#### E5 — domain-balanced adaptation

- Fine-tune the best candidate with the Indic challenge training subset.
- Mix general and Indic examples to avoid catastrophic forgetting.
- Starting sampling ratio: 75% general, 25% Indic/hard cases.
- Select using general validation plus untouched Indic holdout.

#### E6 — teacher-assisted distillation, optional

Only attempt if all earlier stages are stable:

- Use a larger open audio model or an ensemble as an offline teacher.
- Distil soft endpoint probabilities into the tiny model.
- Never introduce a teacher dependency at inference.
- Skip this experiment if licensing, compute, or time is unclear.

### 8.3 Recommended head

Start with the published attention-pooling pattern:

```text
encoder hidden states
  -> learned temporal attention
  -> weighted pooled vector
  -> Linear
  -> LayerNorm
  -> GELU
  -> Dropout
  -> Linear
  -> GELU
  -> binary logit
```

Do not introduce recurrent layers or cross-attention until the baseline is verified. Complexity must earn its place through an ablation.

### 8.4 Loss

Primary loss:

- `BCEWithLogitsLoss`.
- Compute stable class weights from the training split, not each individual batch.
- If classes are close to balanced, use no weighting.

Optional auxiliary loss:

```text
total_loss =
    endpoint_bce
  + 0.1 * midfiller_bce
  + 0.1 * endfiller_bce
```

Avoid focal loss unless analysis shows a genuine hard-example problem and BCE is insufficient.

### 8.5 Training configuration starting point

For free Colab T4-class hardware:

- Precision: FP16 where numerically stable.
- Physical batch size: 8–24, discovered by a memory probe.
- Effective batch size: 128–384 using gradient accumulation.
- Head learning rate: `1e-4` for head-only, then `5e-5`.
- Encoder learning rate: `5e-6` to `1e-5` for partial unfreeze.
- Warm-up: 5–10%.
- Weight decay: `0.01`.
- Gradient clipping: `1.0`.
- Epochs:
  - head-only: 1–2;
  - partial unfreeze: 1–3;
  - full fine-tune: no more than 4.
- Early stopping: validation macro-F1 with patience measured in evaluation intervals.
- Seed set: at least `42`; rerun the final candidate with two additional seeds if compute allows.

These are starting values, not guaranteed optima.

### 8.6 Training safeguards

- Save resumable checkpoints every 30–45 minutes to Google Drive.
- Keep only the best two and latest one checkpoints.
- Save optimizer, scheduler, RNG, and sampler state.
- Write metrics to local JSONL/CSV; use TensorBoard locally.
- Weights & Biases must be optional and disabled by default because it is unnecessary for this scope.
- Catch Colab disconnects by persisting checkpoints outside the ephemeral VM.
- Run a 500–2,000-example smoke experiment before any full run.
- Verify overfitting on a 64-example subset as a model/pipeline sanity test.

---

## 9. Evaluation design

### 9.1 Threshold selection

The model outputs a probability, not just a class. Select the threshold on validation data.

Procedure:

1. Fit optional temperature scaling on a calibration split.
2. Sweep thresholds from `0.05` to `0.95`.
3. Plot false-complete versus false-incomplete rates.
4. Choose the threshold maximizing the project selection score.
5. Publish at least one conservative threshold favoring fewer interruptions.
6. Freeze the threshold before public-test evaluation.

The demo should expose the threshold control while clearly showing the recommended default.

### 9.2 Public test protocol

- Run only after model and threshold are frozen.
- Do not train on public-test failures.
- Report overall and published slices.
- Save predictions keyed by sample ID.
- Include exact model checksum and evaluation command.

### 9.3 Operational replay test

Clip classification does not fully represent a voice conversation. Create a small replay benchmark:

- Concatenate scripted turns with pauses of 200, 350, 500, and 800 ms.
- Include incomplete pauses followed by resumed speech.
- Trigger the detector after VAD-like silence.
- Measure:
  - premature response opportunities;
  - additional waiting time after true completion;
  - model calls per minute;
  - end-to-end decision latency.

This benchmark can be small but must be reproducible.

### 9.4 Error taxonomy

Manually categorize at least the 100 highest-confidence errors:

- Short complete response.
- Filler.
- Connective ending.
- Prosodic continuation.
- Code-switching.
- Number/list continuation.
- Noise.
- Long turn truncated to 8 seconds.
- Silence/padding artifact.
- Suspected label error.
- Ambiguous even to a human.

Use this taxonomy to drive exactly one hard-negative iteration.

### 9.5 Statistical reporting

- Report counts with every percentage.
- Add bootstrap 95% confidence intervals for headline metrics.
- For model comparisons, use paired bootstrap differences on identical examples.
- Do not claim improvement when confidence intervals are inconclusive.
- Report variation across final seeds if available.

---

## 10. Compute, storage, and training-time estimate

### 10.1 Why estimates are ranges

Google states that free Colab GPU type, availability, idle timeout, and maximum runtime are dynamic and not guaranteed. Free notebooks may run for at most 12 hours, depending on availability and usage. Audio decode and feature extraction can become the bottleneck even when the model is small.

Therefore, all estimates must be treated as planning ranges and updated after a measured 2,000-step pilot.

### 10.2 Storage estimate

- Upstream training data: 41.4 GB.
- Public test data: 4.84 GB.
- Hugging Face cache and decoded temporary overhead: allow 15–35 GB.
- Checkpoints/logs/exports: 1–5 GB with retention.
- Safe ephemeral-disk target: 75–100 GB.

Do not precompute all 80×800 float32 features unless storage is measured first; uncompressed features can exceed 69 GB for the full training set.

Recommended data strategy:

1. Download/cache sharded source audio on the Colab ephemeral disk when capacity permits.
2. Decode and compute features on demand.
3. Persist only small manifests, split indices, checkpoints, and results to Drive.
4. If disk is insufficient, run experiments on deterministic source/language-stratified subsets and reserve full-data training for the final candidate.

### 10.3 Runtime estimate on free Colab

Assume a T4-like 16 GB GPU when available:

- Environment setup and dependency verification: 15–45 minutes.
- Dataset metadata audit: 10–40 minutes.
- Full data download: approximately 1–4 hours depending on network/cache.
- Audio integrity audit on a stratified subset: 30–90 minutes.
- 2,000-step throughput pilot: 15–45 minutes.
- Head-only full pass: approximately 1–3 GPU hours.
- Partial-unfreeze full pass: approximately 3–8 GPU hours per epoch.
- Full encoder fine-tune: approximately 5–12 GPU hours per epoch.
- Full public-test evaluation: approximately 30–120 minutes.
- ONNX export, static calibration, and parity tests: 20–60 minutes.
- CPU benchmark and report generation: 20–60 minutes.

Likely end-to-end compute:

- Minimum credible submission: 8–15 GPU hours.
- Recommended experiment plan: 20–40 GPU hours.
- Full plan with multi-seed and optional distillation: 45–80 GPU hours.

Likely calendar time on free Colab:

- Minimum: 3–5 days.
- Recommended: 7–12 days.
- With unreliable GPU availability or full ablations: 2–3 weeks.

The published upstream batch size of 384 is not a safe physical batch-size assumption for free T4 hardware. Use gradient accumulation.

### 10.4 Go/no-go checkpoints

- After pilot: replace estimates with measured examples/second.
- After E1: stop if the pipeline cannot reproduce sensible baseline behavior.
- After E2: if gains over official Smart Turn are below noise, focus on Indic data, calibration, demo, and analysis rather than adding architecture.
- After 40 GPU hours: stop model expansion and complete the submission.

---

## 11. Free-service strategy

### 11.1 Source control

Use a public GitHub repository:

- Code, configs, tests, report, and small fixtures only.
- GitHub Actions for lint/unit tests, within free public-repository allowances.
- GitHub Releases optionally hold a small submission bundle.
- GitHub Pages hosts the static browser demo.

### 11.2 Model hosting

Use a public Hugging Face model repository:

- INT8 ONNX model.
- FP32 ONNX model if size remains reasonable.
- Threshold/config JSON.
- Preprocessing metadata.
- Model card with metrics, limitations, intended use, and licenses.

Public Hugging Face storage is best-effort for free accounts. The compact model is small enough to be a responsible use.

### 11.3 Demo hosting

Current Hugging Face documentation states:

- CPU Basic hardware itself has no hourly charge.
- Creating new Gradio or Docker compute Spaces requires a paid plan.
- Static Spaces remain free.

Therefore:

#### Primary public demo — static and client-side

Host the same static app on:

1. GitHub Pages.
2. Hugging Face Static Space as a redundant mirror.

The browser performs:

- microphone capture or file upload;
- mono conversion/resampling;
- log-Mel preprocessing;
- ONNX Runtime Web/WASM inference;
- thresholding and visualization.

No API key, backend, account, or paid compute is required.

#### Reviewer fallback — local Gradio

Provide:

```bash
pip install -e .[demo]
python demo/gradio_app.py
```

The local app uses ONNX Runtime CPU and offers equivalent predictions.

#### Optional notebook demo

Provide a Colab badge that installs the package and launches Gradio with a temporary share link. This is a convenience, not the primary permanent demo.

### 11.4 Free observability

- Training logs: JSONL/CSV and TensorBoard.
- Experiment index: `docs/experiment_log.md`.
- CI: GitHub Actions.
- Model/version provenance: Git tags and Hugging Face commit hashes.
- No paid telemetry service.

---

## 12. Demo specification

### 12.1 User experience

The demo must support:

- Record from microphone.
- Upload WAV/MP3/FLAC where browser support allows.
- Play back selected audio.
- Show waveform or duration.
- Show `Complete` or `Incomplete`.
- Show calibrated probability.
- Show current threshold and permit adjustment.
- Show inference time excluding model download.
- Explain that the model is intended to run after a VAD-detected pause.
- Provide example clips for complete, incomplete, filler, and Hinglish cases.

### 12.2 Educational value

The demo should make the distinction between silence and semantic completion visible:

- “Speech paused” is a VAD event.
- “Turn complete” is the model decision.

Include two paired examples:

- “I want to book a flight.” — complete.
- “I want to book a…” — incomplete.

And one Hinglish pair:

- “haan, that works for me.” — complete.
- “haan, but actually…” — incomplete.

### 12.3 Browser implementation risks

Browser preprocessing parity is the highest demo risk. Before polishing UI:

1. Export ten fixed audio fixtures.
2. Generate canonical Python log-Mel features and predictions.
3. Run browser preprocessing and ONNX inference.
4. Assert maximum feature/prediction deviation within documented tolerance.
5. If parity cannot be achieved quickly, use a WASM-compatible preprocessing implementation and retain local Gradio as the reference.

### 12.4 Demo acceptance tests

- Cold load succeeds on current Chrome and Edge.
- Model download size is displayed.
- Inference runs without network calls after assets load.
- A 1-second and an 8-second clip both work.
- Stereo and non-16 kHz uploads are normalized.
- Empty, overlong, and corrupt uploads fail gracefully.
- No microphone audio leaves the browser in the static demo.
- Mobile layout remains usable, even if mobile inference is slower.

---

## 13. Development phases

### Phase 0 — bootstrap and decision capture

Estimated effort: 0.5 day.

Tasks:

- Create repository structure.
- Pin Python and dependency versions.
- Pin upstream Smart Turn commit.
- Add license and attribution.
- Add deterministic config loading and seed setup.
- Establish CI for formatting, lint, and unit tests.

Exit criteria:

- Clean environment installs.
- Tests run.
- Upstream provenance is recorded.

### Phase 1 — data audit and split

Estimated effort: 1 day plus download time.

Tasks:

- Build metadata audit.
- Validate sampled audio.
- Generate immutable split manifests.
- Document leakage limitations.
- Create a tiny fixture dataset for CI.

Exit criteria:

- Data card exists.
- Split checksums are frozen.
- No public-test access occurs during training.

### Phase 2 — baseline reproduction

Estimated effort: 1 day.

Tasks:

- Integrate official INT8 baseline.
- Implement project metric suite.
- Run overfit and smoke checks.
- Run B1 on local validation and optionally public test once as a reference.
- Verify preprocessing.

Exit criteria:

- Metrics are reproducible.
- Python and ONNX predictions match.

### Phase 3 — efficient model experiments

Estimated effort: 2–4 days, depending on Colab.

Tasks:

- E1 head-only.
- E2 partial unfreeze.
- E3 only if justified.
- Record config, git SHA, data SHA, runtime, and metrics.

Exit criteria:

- At least two candidates compared.
- Best candidate selected without public-test tuning.

### Phase 4 — Official Indic slice analysis

Estimated effort: 0.5–1 day, overlapping Phase 3.

Tasks:

- Build Hindi/Marathi/English-filler slices from official metadata.
- Confirm the slice is disjoint from local training.
- Report per-language and filler metrics.
- Optionally upsample Hindi/Marathi/hard filler examples during domain adaptation, still using only official data.

Exit criteria:

- Indic slice counts and metrics are recorded.
- No extra recordings were used.

### Phase 5 — calibration and error analysis

Estimated effort: 1 day.

Tasks:

- Threshold sweep.
- Calibration.
- Error taxonomy.
- One hard-negative iteration.
- Freeze final model and threshold.

Exit criteria:

- Balanced operating point justified.
- Dominant failures documented.

### Phase 6 — export and benchmark

Estimated effort: 0.5–1 day.

Tasks:

- Export FP32 ONNX.
- Static INT8 quantization using a representative calibration sample.
- Validate parity.
- Benchmark warm/cold CPU latency and memory.

Exit criteria:

- Deployment gates pass or exceptions are documented.

### Phase 7 — demos and publishing

Estimated effort: 1–2 days.

Tasks:

- Local Gradio demo.
- Client-side static demo.
- GitHub Pages deployment.
- Hugging Face model repository.
- Hugging Face Static Space mirror if desired.

Exit criteria:

- Public links work without paid APIs.
- Local fallback works from a clean checkout.

### Phase 8 — report and submission hardening

Estimated effort: 1 day.

Tasks:

- Final report and ablations.
- Architecture diagram.
- Model/data cards.
- Reproduction commands.
- Demo video/GIF as a fallback.
- Fresh-machine verification.

Exit criteria:

- Submission bundle satisfies every gate in Section 4.

---

## 14. Experiment matrix and stopping policy

Required experiments:

1. Official Smart Turn v3.2 INT8 baseline.
2. Head-only Whisper Tiny.
3. Partial-unfreeze Whisper Tiny.
4. Best model with versus without targeted Indic data.
5. FP32 versus INT8.
6. Threshold `0.5` versus calibrated threshold.

Optional experiments, in priority order:

1. Auxiliary filler heads.
2. Augmentation ablation.
3. Full encoder fine-tuning.
4. Teacher distillation.

Every experiment entry must include:

- ID and hypothesis.
- Parent checkpoint.
- Code commit.
- Config.
- Data/split checksum.
- GPU type.
- Wall-clock and GPU time.
- Validation metrics.
- Slice metrics.
- Artifact path/checksum.
- Decision: keep, reject, or inconclusive.

Stopping rules:

- Reject complexity that improves macro-F1 by less than 0.3 percentage points without a meaningful slice or latency benefit.
- Stop after two consecutive experiments fail to improve the frozen selection score.
- Do not spend more compute on architecture until data and labels have been inspected.

---

## 15. Testing strategy

### 15.1 Unit tests

- Audio mono conversion.
- Resampling.
- Last-8-second truncation.
- Left padding.
- Feature shape and dtype.
- Split determinism and disjointness.
- Model output shape/range.
- Threshold behavior.
- Corrupt-audio handling.

### 15.2 Integration tests

- Dataset row to model loss.
- Checkpoint save/resume.
- PyTorch to ONNX export.
- FP32/INT8 inference.
- Python/browser fixture parity.
- Local demo smoke test.

### 15.3 Regression tests

Keep 20–50 redistributable fixtures with expected probability ranges. Avoid exact probability assertions across all hardware; use tolerances.

Block release when:

- Preprocessing output changes unexpectedly.
- Export parity fails.
- INT8 regression exceeds the accepted bound.
- Model card metrics do not match generated metric files.

---

## 16. Security, privacy, and licensing

- Do not commit Hugging Face tokens or Google credentials.
- Use `.env.example`, never `.env`.
- Request only microphone permission needed by the demo.
- State that static-demo audio remains on device.
- Limit upload duration and size.
- Parse audio defensively.
- Track licenses for the upstream dataset, model weights, noise, and added recordings.
- Do not publish contributor audio without explicit redistribution permission.
- Document that turn detection can fail and must not gate safety-critical actions.

---

## 17. Risks and mitigations

### Free Colab disconnects

Mitigation: resumable checkpoints to Drive, short pilot runs, retained split/config hashes.

### Dataset does not explicitly isolate Hinglish

Mitigation: create a dedicated, speaker-separated challenge set and avoid unsupported Hinglish claims based only on Hindi/English aggregate scores.

### Random-split leakage

Mitigation: group by speaker/session/source/fingerprint where possible and document residual risk.

### Public-test overfitting

Mitigation: treat it as release-only; never use it to pick hyperparameters or thresholds.

### Synthetic-data bias

Mitigation: report synthetic/human slices and cap synthetic contribution in final domain adaptation.

### Browser feature mismatch

Mitigation: fixture parity tests before UI work; local Gradio remains reference fallback.

### Quantization regression

Mitigation: representative static calibration, per-slice parity checks, retain FP32 artifact.

### Free demo-host policy changes

Mitigation: duplicate static hosting on GitHub Pages and Hugging Face Static Space; include local demo and a short recorded walkthrough.

### Scope expansion

Mitigation: enforce experiment stopping rules and a 40-GPU-hour recommended ceiling.

---

## 18. Inputs required from the project owner

### Locked

- GitHub: https://github.com/Saaalil/ShipRocket-assesment.git
- Hugging Face Space: https://huggingface.co/spaces/Saalil/Assesment_SR
- GitHub username: `Saaalil`
- Hugging Face username: `Saalil`
- v1 data: official Smart Turn v3.2 train/test only
- Extra Hinglish collection: deferred

### Still needed later, never paste into chat

### Step 2 — account access

The owner performs login locally; credentials are never sent in chat or committed:

- GitHub authentication so this machine can push to `Saaalil/ShipRocket-assesment`.
- Hugging Face write token stored as a secret/environment variable, used only when publishing the Space and model.
- Google account for Colab and Drive checkpoint storage.

### Step 3 — licensing confirmation

Confirm:

- The submission will be public.
- The desired code license; Apache-2.0 is the recommended default unless upstream compatibility requires otherwise.

### Step 4 — challenge-data participation

Locked for v1: use only the official dataset. No extra recordings.

### Step 5 — access to free compute

Needed only when training starts. Open the training notebook in free Colab and confirm:

- A GPU runtime can be allocated.
- Approximate ephemeral disk capacity.
- Drive is mounted for checkpoints.

No paid upgrade is required; availability may delay the schedule.

### Step 6 — subjective review

Review:

- 30–50 ambiguous validation clips.
- Top model errors.
- Whether the selected threshold feels balanced.
- Demo language and visual clarity.

### Step 7 — publication approval

Before publishing, approve:

- Model card.
- Dataset/challenge-set license.
- Final metrics.
- Known limitations.
- Public demo examples.

### Step 8 — final submission

Provide or confirm the recipient/submission channel. The final package should contain:

- GitHub URL.
- Hugging Face model URL.
- Static demo URL.
- Local demo instructions.
- Short report.
- Optional 60–90 second demo recording.

---

## 19. Instructions for the implementation AI

### 19.1 First actions

1. Read this entire document.
2. Inspect the exact upstream repository and license.
3. Create `docs/decisions.md`.
4. Write a task list mapped to Phases 0–8.
5. Implement Phase 0 only.
6. Run tests before moving to Phase 1.

### 19.2 Working rules

- Do not invent metrics.
- Do not report unmeasured latency or accuracy.
- Do not change frozen splits.
- Do not train on public test data.
- Do not add dependencies without documenting why.
- Do not use paid services.
- Do not publish anything until the owner approves.
- Prefer scripts for reproducibility; notebooks should call library code rather than contain unique logic.
- Keep training and inference preprocessing in one shared implementation.
- Commit generated metric JSON, but not raw data or large checkpoints.

### 19.3 Required progress report after each phase

Return:

- Completed tasks.
- Files changed.
- Commands/tests run.
- Measured outputs.
- Risks/blockers.
- Decision needed from owner.
- Next phase estimate.

### 19.4 Definition of done for the AI

The AI is finished only when:

- all required experiments are reproducible;
- the selected model is justified by held-out evidence;
- INT8 export parity is verified;
- both demos work;
- public artifacts are published or publication-ready;
- the report clearly documents failures and limitations;
- a clean-checkout reviewer run succeeds.

---

## 20. Final report outline

1. Problem and conversational failure modes.
2. Why VAD alone is insufficient.
3. Dataset audit and split methodology.
4. Architecture and parameter count.
5. Training strategy under free-compute constraints.
6. Experiments and ablations.
7. Overall and slice metrics.
8. Threshold and calibration analysis.
9. Indic/Hinglish challenge evaluation.
10. Latency, size, and quantization.
11. Error analysis.
12. Demo and reproduction.
13. Limitations and future work.

The report should sound human, include failed experiments, and avoid claiming that clip-level accuracy proves natural full-duplex conversation quality.

---

## 21. Recommended schedule

Assuming free Colab access and one primary developer/AI agent:

- Day 1: repository, environment, data metadata audit.
- Day 2: split manifests, preprocessing, official baseline.
- Day 3: head-only pilot and throughput measurement.
- Days 4–5: partial-unfreeze training and evaluation.
- Day 6: official Hindi/Marathi/filler-slice analysis and optional upsample.
- Day 7: calibration, final model freeze, public test.
- Day 8: ONNX/INT8 export and CPU benchmark.
- Days 9–10: browser and Gradio demos.
- Day 11: report, model card, and fresh-machine validation.
- Day 12: buffer for Colab/runtime/hosting failures.

Expected plan: 7–12 working days. Keep a 2–3 week calendar window because free GPU availability is not guaranteed.

---

## 22. Sources consulted

- Dataset: https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train
- Public test set: https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test
- Smart Turn repository: https://github.com/pipecat-ai/smart-turn
- Published model: https://huggingface.co/pipecat-ai/smart-turn-v3
- Pipecat turn-detection overview: https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview
- OpenAI GPT-Live article: https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/en/spaces-overview
- Hugging Face storage limits: https://huggingface.co/docs/hub/en/storage-limits

Service policies and free-tier availability may change. Recheck them before publication.
