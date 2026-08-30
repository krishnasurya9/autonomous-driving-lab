import argparse
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from experiments.run_comparison import run_paired_experiment_on_seed

def main():
    parser = argparse.ArgumentParser(description="Run one agent in the top-down 2D driving simulator.")
    parser.add_argument("--agent", type=str, required=True, choices=["human", "llm", "rl"], help="Agent type")
    parser.add_argument("--seed", type=int, default=2037, help="Procedural scenario seed")
    parser.add_argument("--no_render", action="store_true", help="Run without opening the 2D window")
    parser.add_argument("--video", action="store_true", help="Reserved; 2D video export is not configured yet")
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
