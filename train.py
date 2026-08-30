import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from rl.train_sac import train_sac

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent (SAC) on MetaDrive training seeds.")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL Algorithm (sac)")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps")
    args = parser.parse_args()

    train_sac(total_timesteps=args.timesteps)

if __name__ == "__main__":
    main()
