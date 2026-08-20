from __future__ import annotations

import argparse

from smart_turn.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a turn-detection experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = train_from_config(args.config)
    print(f"saved model to {path}")


if __name__ == "__main__":
    main()
