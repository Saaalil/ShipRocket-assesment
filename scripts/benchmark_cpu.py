from __future__ import annotations

import argparse
import json

from smart_turn.benchmark import benchmark_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX CPU latency")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(benchmark_onnx(args.onnx, args.repeats), indent=2))


if __name__ == "__main__":
    main()
