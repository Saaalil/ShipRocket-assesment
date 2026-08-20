from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish ONNX weights to Hugging Face")
    parser.add_argument("--repo", required=True, help="e.g. Saalil/Assesment_SR-model")
    parser.add_argument("--onnx", default="artifacts/model_int8.onnx")
    parser.add_argument("--token", default=None, help="Use env HF_TOKEN instead of passing here")
    args = parser.parse_args()
    if args.token:
        login(token=args.token)
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    for path in [args.onnx, Path(args.onnx).with_suffix(".json"), "model_card/README.md"]:
        file_path = Path(path)
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name if file_path.suffix != ".md" else "README.md",
                repo_id=args.repo,
                repo_type="model",
            )
    print(f"published {args.repo}")


if __name__ == "__main__":
    main()
