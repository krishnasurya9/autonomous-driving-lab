import argparse
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from rl.train_sac import train_sac

def main():
    parser = argparse.ArgumentParser(description="Train a SAC agent in the 2D driving environment.")
    parser.add_argument("--algorithm", type=str, default="sac", help="RL Algorithm (sac)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    args = parser.parse_args()

    train_sac(total_timesteps=args.timesteps)

if __name__ == "__main__":
    main()
