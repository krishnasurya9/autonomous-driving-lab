"""Run the configured agents over one of the explicit seed splits."""

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from experiments.run_comparison import run_paired_suite


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents on shared 2D simulator seeds.")
    parser.add_argument("--seeds", choices=["test", "validation", "train"], default="test")
    parser.add_argument("--count", type=int, default=None, help="Number of seeds; default runs the full selected split")
    parser.add_argument("--agents", default="llm,rl", help="Comma-separated agents; headless human driving is unsupported")
    args = parser.parse_args()

    seed_file = os.path.join(PROJECT_DIR, "experiments", "seeds", f"{args.seeds}_seeds.json")
    with open(seed_file, "r", encoding="utf-8") as stream:
        seeds = json.load(stream)
    if args.count is not None:
        if args.count < 1:
            parser.error("--count must be at least 1")
        if args.count > len(seeds):
            parser.error(f"--count {args.count} exceeds the {len(seeds)} seeds in the {args.seeds} split")
        seeds = seeds[: args.count]

    agents = [agent.strip().lower() for agent in args.agents.split(",") if agent.strip()]
    if not agents or any(agent not in {"llm", "rl"} for agent in agents):
        parser.error("--agents must contain llm and/or rl; use run.py for an interactive human episode")
    run_paired_suite(seeds=seeds, agents=agents)


if __name__ == "__main__":
    main()
