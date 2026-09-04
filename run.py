import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from experiments.run_comparison import run_paired_experiment_on_seed

def main():
    parser = argparse.ArgumentParser(description="Run single-agent evaluation with visual rendering & video recording.")
    parser.add_argument("--agent", type=str, required=True, choices=["human", "llm", "rl"], help="Agent type")
    parser.add_argument("--seed", type=int, default=2037, help="Procedural scenario seed")
    parser.add_argument("--no_render", action="store_true", help="Disable 3D visual rendering window")
    parser.add_argument("--video", action="store_true", help="Save MP4 video recording to recordings/")
    args = parser.parse_args()

    render = not args.no_render
    save_video = args.video

    run_paired_experiment_on_seed(
        seed=args.seed,
        agent_type=args.agent,
        render=render,
        save_video=save_video,
    )

if __name__ == "__main__":
    main()
