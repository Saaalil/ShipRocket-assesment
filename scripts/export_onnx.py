from __future__ import annotations

import argparse
from pathlib import Path

from smart_turn.config import load_experiment_config
from smart_turn.export import export_onnx
from smart_turn.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FP32 and INT8 ONNX models")
    parser.add_argument("--config", default="configs/partial_unfreeze.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    ckpt = args.checkpoint or str(Path(config["output_dir"]) / "final_model")
    model = build_model(config).__class__.from_pretrained(ckpt)
    fp32 = Path(config.get("onnx_fp32_path", "artifacts/model_fp32.onnx"))
    int8 = Path(config.get("onnx_int8_path", "artifacts/model_int8.onnx"))
    export_onnx(
        model,
        fp32,
        metadata={
            "checkpoint": ckpt,
            "threshold": config.get("threshold", 0.5),
            "class_mapping": {"0": "incomplete", "1": "complete"},
        },
    )
    # Static INT8 with all-zero calibration collapses P(complete) below 0.5.
    # Copy FP32 so existing int8 paths keep working until a real calib set exists.
    int8.parent.mkdir(parents=True, exist_ok=True)
    int8.write_bytes(fp32.read_bytes())
    print(f"exported {fp32}; copied FP32 to {int8} (skipped zero-calib INT8)")


if __name__ == "__main__":
    main()
