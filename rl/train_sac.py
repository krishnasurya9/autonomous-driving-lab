import os
import sys
import json
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Ensure project root is in path
sys.path.insert(0, os.path.abspath("."))

from environment.metadrive_env import CustomMetaDriveEnv

def train_sac(
    train_seeds_file: str = "experiments/seeds/train_seeds.json",
    total_timesteps: int = 20000,
    checkpoint_dir: str = "rl/checkpoints",
):
    print("=" * 60)
    print("STARTING SAC RL TRAINING PIPELINE")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(train_seeds_file, "r") as f:
        train_seeds = json.load(f)

    start_seed = train_seeds[0]
    print(f"Training on procedural seeds starting at {start_seed} (Total seeds: {len(train_seeds)})")

    env_config = {
        "seed": start_seed,
        "render": False,
        "traffic_density": 0.1,
        "target_speed_kmh": 40.0,
    }
    env = CustomMetaDriveEnv(env_config)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        tensorboard_log="rl/tensorboard_logs/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="sac_metadrive_checkpoint",
    )

    print(f"Training SAC model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    final_path = os.path.join(checkpoint_dir, "sac_metadrive.zip")
    model.save(final_path)
    print(f"SAC model training complete! Model saved to: {final_path}")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    args = parser.parse_args()
    train_sac(total_timesteps=args.timesteps)
