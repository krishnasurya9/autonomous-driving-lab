"""Train a Stable-Baselines3 SAC policy against the lightweight 2D task."""

import argparse
import json
import os
import sys

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.abspath("."))

from environment.driving_2d_env import Driving2DEnv


class SeedSequenceDriving2DEnv(Driving2DEnv):
    """Cycle through a fixed seed split whenever an episode starts."""

    def __init__(self, config, seeds):
        self.seeds = tuple(int(seed) for seed in seeds)
        if not self.seeds:
            raise ValueError("At least one environment seed is required")
        self.episode_index = 0
        super().__init__(config)

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = self.seeds[self.episode_index % len(self.seeds)]
        self.episode_index += 1
        return super().reset(seed=seed, options=options)


def train_sac(
    train_seeds_file: str = "experiments/seeds/train_seeds.json",
    validation_seeds_file: str = "experiments/seeds/validation_seeds.json",
    total_timesteps: int = 50000,
    checkpoint_dir: str = "rl/checkpoints",
):
    with open(train_seeds_file, "r", encoding="utf-8") as stream:
        train_seeds = json.load(stream)
    with open(validation_seeds_file, "r", encoding="utf-8") as stream:
        validation_seeds = json.load(stream)
    train_seeds = [int(seed) for seed in train_seeds] or [1000]
    validation_seeds = [int(seed) for seed in validation_seeds] or [1100]
    seed = train_seeds[0]
    os.makedirs(checkpoint_dir, exist_ok=True)

    env_config = {"seed": seed, "render": False, "target_speed_kmh": 40.0}
    env = Monitor(SeedSequenceDriving2DEnv(env_config, train_seeds))
    eval_env = Monitor(
        SeedSequenceDriving2DEnv(
            {**env_config, "seed": validation_seeds[0]}, validation_seeds
        )
    )
    model = SAC(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=50000,
        batch_size=256, gamma=0.99, verbose=1,
        tensorboard_log="rl/tensorboard_logs/",
        seed=seed,
    )
    checkpoint = CheckpointCallback(
        save_freq=5000, save_path=checkpoint_dir, name_prefix="sac_2d_checkpoint"
    )
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=5000,
        n_eval_episodes=min(5, len(validation_seeds)),
        deterministic=True,
        render=False,
    )
    print(f"Training 2D SAC for {total_timesteps} steps (seed {seed})...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint, evaluation])

    final_path = os.path.join(checkpoint_dir, "sac_2d.zip")
    best_path = os.path.join(checkpoint_dir, "best_model.zip")
    model_to_save = SAC.load(best_path, env=env) if os.path.exists(best_path) else model
    model_to_save.save(final_path)
    env.close()
    eval_env.close()
    print(f"2D SAC policy saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000)
    args = parser.parse_args()
    train_sac(total_timesteps=args.timesteps)
