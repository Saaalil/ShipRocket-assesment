from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login

from smart_turn.constants import HF_SPACE_REPO

SPACE_FILES = [
    ("app.py", "app.py"),
    ("demo/__init__.py", "demo/__init__.py"),
    ("demo/gradio_app.py", "demo/gradio_app.py"),
    ("spaces/README.md", "README.md"),
    ("spaces/requirements.txt", "requirements.txt"),
    ("spaces/packages.txt", "packages.txt"),
]


def _copy(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the Gradio demo to a Hugging Face Space")
    parser.add_argument("--repo", default=HF_SPACE_REPO)
    parser.add_argument("--onnx", default="artifacts/model_fp32.onnx")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    if args.token:
        login(token=args.token)
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    root = Path.cwd()
    stage = Path(tempfile.mkdtemp(prefix="hf-space-"))
    for src, dest in SPACE_FILES:
        _copy(root / src, stage / dest)
    src_dir = root / "src" / "smart_turn"
    if src_dir.exists():
        for path in src_dir.rglob("*.py"):
            _copy(path, stage / path.relative_to(root))
    onnx = Path(args.onnx)
    if onnx.exists():
        _copy(onnx, stage / "artifacts" / onnx.name)
        sidecar = onnx.with_suffix(".json")
        if sidecar.exists():
            _copy(sidecar, stage / "artifacts" / sidecar.name)
    api.upload_folder(folder_path=str(stage), repo_id=args.repo, repo_type="space")
    print(f"published https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
