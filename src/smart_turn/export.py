from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smart_turn.constants import FEATURE_VERSION, N_FRAMES, N_MELS
from smart_turn.model import SmartTurnModel


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: SmartTurnModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.model(input_features)["probabilities"]


def export_onnx(
    model: SmartTurnModel,
    path: str | Path,
    opset: int = 17,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, N_MELS, N_FRAMES, dtype=torch.float32)
    wrapped = ExportWrapper(model.eval().cpu())
    torch.onnx.export(
        wrapped,
        dummy,
        str(output),
        input_names=["input_features"],
        output_names=["probability_complete"],
        dynamic_axes={"input_features": {0: "batch"}, "probability_complete": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    sidecar = output.with_suffix(".json")
    payload = {
        "feature_version": FEATURE_VERSION,
        "input_name": "input_features",
        "output_name": "probability_complete",
        "input_shape": [None, N_MELS, N_FRAMES],
        **(metadata or {}),
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def quantize_onnx(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_features: np.ndarray,
) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self, features: np.ndarray) -> None:
            self._data = [features[i : i + 1] for i in range(len(features))]
            self._iter = iter(self._data)

        def get_next(self):
            try:
                item = next(self._iter)
            except StopIteration:
                return None
            return {"input_features": item.astype(np.float32)}

    output = Path(int8_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(output),
        calibration_data_reader=Reader(calibration_features),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return output
