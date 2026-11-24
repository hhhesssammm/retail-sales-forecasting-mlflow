import argparse
from pathlib import Path

from src.data import DATA_PATH
from src.report import generate_report
from src.train import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run training, registry and final report")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    summary = run_experiment(args.data)
    generate_report(args.data, batch_id=summary["batch_id"])


if __name__ == "__main__":
    main()
