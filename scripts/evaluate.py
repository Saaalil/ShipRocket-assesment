from __future__ import annotations

import argparse
import json
from pathlib import Path

from smart_turn.calibrate import sweep_thresholds
from smart_turn.config import load_experiment_config
from smart_turn.data import TurnDataset
from smart_turn.evaluate import compute_metrics, slice_metrics
from smart_turn.splits import is_indic_language
from smart_turn.train import _load_hf_dataset, build_model, predict_loader, prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    _, val_ds = prepare_splits(config)
    model = build_model(config)
    ckpt = args.checkpoint or str(Path(config["output_dir"]) / "final_model")
    model = model.__class__.from_pretrained(ckpt)
    probs = predict_loader(model, val_ds, batch_size=int(config.get("eval_batch_size", 8)))
    labels = [int(val_ds[i]["labels"].item()) for i in range(len(val_ds))]
    languages = [str(val_ds[i]["language"]) for i in range(len(val_ds))]
    indic = [
        is_indic_language(lang, config.get("indic_languages", [])) for lang in languages
    ]
    threshold, sweep = sweep_thresholds(
        __import__("numpy").asarray(labels),
        probs,
        __import__("numpy").asarray(indic),
    )
    metrics = compute_metrics(labels, probs, threshold)
    metrics["selected_threshold"] = threshold
    metrics["by_language"] = slice_metrics(labels, probs, languages, threshold)
    out = Path("reports/metrics") / f"{config['name']}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "sweep": sweep[:5]}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    test_name = config.get("test_dataset")
    if test_name:
        test = TurnDataset(_load_hf_dataset(test_name), augment=False)
        # Keep public test unused during model selection. This flag writes a reminder only.
        print(f"public test {test_name} has {len(test)} clips; run after freeze.")


if __name__ == "__main__":
    main()
