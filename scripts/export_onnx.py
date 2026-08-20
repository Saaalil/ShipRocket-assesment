from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smart_turn.config import load_experiment_config
from smart_turn.constants import N_FRAMES, N_MELS
from smart_turn.export import export_onnx, quantize_onnx
from smart_turn.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FP32 and INT8 ONNX models")
    parser.add_argument("--config", default="configs/final.yaml")
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
    n_calib = int(config.get("calibration_samples", 32))
    calib = np.zeros((n_calib, N_MELS, N_FRAMES), dtype=np.float32)
    quantize_onnx(fp32, int8, calib)
    print(f"exported {fp32} and {int8}")


if __name__ == "__main__":
    main()
