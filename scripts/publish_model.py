from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login

from smart_turn.constants import HF_MODEL_REPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish ONNX weights to Hugging Face")
    parser.add_argument("--repo", default=HF_MODEL_REPO, help="e.g. Saalil/Assesment_SR-model")
    parser.add_argument("--onnx", default="artifacts/model_int8.onnx")
    parser.add_argument("--checkpoint", default=None, help="Optional PyTorch final_model folder")
    parser.add_argument("--token", default=None, help="Use env HF_TOKEN instead of passing here")
    args = parser.parse_args()
    if args.token:
        login(token=args.token)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    uploads = [
        (Path(args.onnx), Path(args.onnx).name),
        (Path(args.onnx).with_suffix(".json"), Path(args.onnx).with_suffix(".json").name),
        (Path("model_card/README.md"), "README.md"),
        (Path("reports/partial_unfreeze_eval.json"), "metrics.json"),
    ]
    fp32 = Path("artifacts/model_fp32.onnx")
    if fp32.exists():
        uploads.append((fp32, fp32.name))
        uploads.append((fp32.with_suffix(".json"), fp32.with_suffix(".json").name))
    for file_path, dest in uploads:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type="model",
            )
            print(f"uploaded {dest}")
    if args.checkpoint and Path(args.checkpoint).is_dir():
        api.upload_folder(
            folder_path=args.checkpoint,
            repo_id=args.repo,
            repo_type="model",
            path_in_repo="pytorch",
            allow_patterns=["*.safetensors", "*.json", "*.txt", "config.json", "preprocessor_config.json"],
        )
        print("uploaded pytorch/")
    print(f"published https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
