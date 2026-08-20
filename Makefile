.PHONY: install test lint demo audit train eval export bench

install:
	python -m pip install -e ".[dev,demo,train]"

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts demo

demo:
	python demo/gradio_app.py

audit:
	python scripts/audit_data.py --config configs/data.yaml

train:
	python scripts/train.py --config configs/head_only.yaml

eval:
	python scripts/evaluate.py --config configs/final.yaml

export:
	python scripts/export_onnx.py --config configs/final.yaml

bench:
	python scripts/benchmark_cpu.py --onnx artifacts/model_int8.onnx
