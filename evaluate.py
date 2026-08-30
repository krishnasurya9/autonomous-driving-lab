import argparse
import sys
import os
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from experiments.run_comparison import run_paired_suite

def main():
    parser = argparse.ArgumentParser(description="Evaluate agents across held-out test seeds.")
    parser.add_argument("--seeds", type=str, default="test", choices=["test", "validation", "train"], help="Seed split")
    parser.add_argument("--count", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--agents", default="llm,rl", help="Comma-separated agent list; headless human episodes are not supported")
    args = parser.parse_args()

    seed_file = f"experiments/seeds/{args.seeds}_seeds.json"
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            seeds = json.load(f)
    else:
        seeds = [2000, 2001, 2002, 2003, 2004]

    agents = [agent.strip().lower() for agent in args.agents.split(",") if agent.strip()]
    run_paired_suite(seeds=seeds[: args.count], agents=agents)

if __name__ == "__main__":
    main()
