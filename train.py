import argparse
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from rl.train_sac import train_sac

def main():
    parser = argparse.ArgumentParser(description="Train SAC on the configured 2D driving environment.")
    parser.add_argument("--algorithm", choices=["sac"], default="sac")
    parser.add_argument("--timesteps", type=int, default=None, help="Defaults to rl.total_timesteps in config.yaml")
    parser.add_argument("--train-seeds", default=None, help="Optional training seed manifest JSON")
    parser.add_argument("--validation-seeds", default=None, help="Optional validation seed manifest JSON")
    parser.add_argument("--checkpoint-dir", default=None, help="Output directory for a separate or normal run")
    args = parser.parse_args()

    train_sac(
        total_timesteps=args.timesteps,
        train_seeds_file=args.train_seeds,
        validation_seeds_file=args.validation_seeds,
        checkpoint_dir=args.checkpoint_dir,
    )

if __name__ == "__main__":
    main()
