import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

from experiments.run_comparison import run_paired_suite

def main():
    parser = argparse.ArgumentParser(description="Evaluate agents across held-out test seeds.")
    parser.add_argument("--seeds", type=str, default="test", choices=["test", "validation", "train"], help="Seed split")
    parser.add_argument("--count", type=int, default=5, help="Number of seeds to run")
    args = parser.parse_args()

    seed_file = f"experiments/seeds/{args.seeds}_seeds.json"
    if os.path.exists(seed_file):
        with open(seed_file, "r") as f:
            seeds = json.load(f)
    else:
        seeds = [2000, 2001, 2002, 2003, 2004]

    run_paired_suite(seeds=seeds[: args.count])

if __name__ == "__main__":
    main()
