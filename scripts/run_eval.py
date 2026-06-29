import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evaluator.run_eval import print_summary_table, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DocuPilot-RAG strategy evaluation.")
    parser.add_argument("--eval_file", default="eval_sets/sample_eval.jsonl")
    parser.add_argument("--strategies", nargs="+", default=["naive", "hybrid", "rerank", "agentic"])
    parser.add_argument("--chunk_strategy", choices=["sliding", "section"], default="section")
    args = parser.parse_args()

    result = run_evaluation(
        eval_file=args.eval_file,
        strategies=args.strategies,
        chunk_strategy=args.chunk_strategy,
    )
    print_summary_table(result["summary"], args.strategies)
    print(f"eval report saved to {result['report_path']}")
    print(f"raw results saved to {Path(result['json_path']).parent}")


if __name__ == "__main__":
    main()
