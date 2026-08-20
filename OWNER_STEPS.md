# Owner steps

Do these locally. Never paste tokens into chat.

## 1. Push this repo (only if `git push` failed)

In PowerShell, from this folder:

```powershell
git init
git checkout -b main
git add .
git commit -m "Initial turn-detection assignment repository"
git remote add origin https://github.com/Saaalil/ShipRocket-assesment.git
git push -u origin main
```

If GitHub asks for a password, use a [personal access token](https://github.com/settings/tokens) with `repo` scope, not your account password.

## 2. Hugging Face login

```powershell
pip install huggingface_hub
huggingface-cli login
```

Create a **write** token at https://huggingface.co/settings/tokens and paste it into the CLI prompt, not into chat.

## 3. Publish the demo Space

```powershell
git clone https://huggingface.co/spaces/Saalil/Assesment_SR hf-space
copy app.py hf-space\app.py
copy demo\gradio_app.py hf-space\demo\gradio_app.py
xcopy src hf-space\src /E /I
copy pyproject.toml hf-space\pyproject.toml
```

Add `hf-space/README.md`:

```yaml
---
title: Assesment SR
emoji: 🎙️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
---
Shiprocket turn-detection demo.
```

Then:

```powershell
cd hf-space
git add .
git commit -m "Add Gradio turn-detection demo"
git push
```

If Hugging Face blocks a free Gradio Space, use the static demo instead:

```powershell
# GitHub Pages from /demo/web
```

## 4. Train on Colab

1. Open Google Colab
2. Runtime → GPU
3. Upload this repo or `git clone https://github.com/Saaalil/ShipRocket-assesment.git`
4. Run `notebooks/01_data_audit_colab.ipynb`
5. Run `notebooks/02_train_colab.ipynb` with `configs/head_only.yaml`
6. Save checkpoints to Drive every run
7. Export with `notebooks/03_export_and_publish_colab.ipynb`

## 5. Publish model weights

```powershell
python scripts/publish_model.py --repo Saalil/Assesment_SR-model --onnx artifacts/model_int8.onnx
```

## 6. Submission links

- GitHub: https://github.com/Saaalil/ShipRocket-assesment
- Space: https://huggingface.co/spaces/Saalil/Assesment_SR
- Model: create `Saalil/Assesment_SR-model` after export
