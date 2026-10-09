"""Command-line entry point."""
import argparse
from .runner import run_experiment


def main():
    parser = argparse.ArgumentParser(prog="jrc-moe")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume the paper-faithful 50-prompt replication")
    run.add_argument("--fixed8-csv", required=True, help="validated 50-row Fixed8 diagnostic CSV")
    run.add_argument("--output-dir", default="outputs/reflex_paper_replication50")
    run.add_argument("--limit", type=int, help="run only the first N prompts as an explicitly partial pilot")
    args = parser.parse_args()
    if args.command == "run":
        run_experiment(args.fixed8_csv, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
