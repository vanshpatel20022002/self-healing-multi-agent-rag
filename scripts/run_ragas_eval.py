"""CLI: run Ragas evaluation on sample Q&A pairs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.eval.ragas_eval import run_ragas_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ragas faithfulness and context precision eval")
    parser.add_argument(
        "--samples",
        default="data/eval/samples.json",
        help="Path to evaluation samples JSON",
    )
    parser.add_argument(
        "--output",
        default="data/eval/results.json",
        help="Where to write evaluation summary JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build retrieval records without calling Ragas or vLLM generation",
    )
    args = parser.parse_args()

    summary = run_ragas_evaluation(
        samples_path=Path(args.samples),
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
