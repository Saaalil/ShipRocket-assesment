.PHONY: install lint demo train export eval bench

install:
	python -m pip install -e ".[dev,demo,train]"

lint:
	python -m ruff check src scripts demo

demo:
	python demo/gradio_app.py

train:
	python scripts/train.py --config configs/partial_unfreeze.yaml

export:
	python scripts/export_onnx.py --config configs/partial_unfreeze.yaml

eval:
	python scripts/evaluate.py --config configs/partial_unfreeze.yaml

bench:
	python scripts/benchmark_cpu.py --onnx artifacts/model_fp32.onnx
