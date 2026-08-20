from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from smart_turn.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official Smart Turn metadata")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)

    from datasets import load_dataset

    dataset = load_dataset(config["train_dataset"])["train"]
    if args.max_rows:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    labels = [bool(value) for value in dataset["endpoint_bool"]]
    languages = [str(value) for value in dataset["language"]]
    sources = [str(value) for value in dataset["dataset"]]
    synthetic = [bool(value) for value in dataset["synthetic"]]
    report = {
        "rows": len(dataset),
        "complete": sum(labels),
        "incomplete": len(labels) - sum(labels),
        "languages": dict(Counter(languages).most_common()),
        "sources": dict(Counter(sources).most_common(20)),
        "synthetic_true": sum(synthetic),
        "indic_proxy_count": sum(
            1
            for lang in languages
            if lang.lower() in {item.lower() for item in config["indic_languages"]}
        ),
    }
    out = Path("reports/metrics/data_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
