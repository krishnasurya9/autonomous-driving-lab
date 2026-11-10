"""Train the configured SAC policy in the configured environment."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


class CyclingSeedEnv(gym.Wrapper):
    """Use a manifest seed on every episode reset, cycling deterministically."""

    def __init__(self, env: gym.Env, seeds: List[int]):
        super().__init__(env)
        if not seeds:
            raise ValueError("At least one episode seed is required.")
        self.seeds = [int(seed) for seed in seeds]
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.seeds[self.episode_index % len(self.seeds)]
            self.episode_index += 1
        return self.env.reset(seed=seed, options=options)


def _read_seeds(path: str, fallback: List[int]) -> List[int]:
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
    with open(resolved, "r", encoding="utf-8") as stream:
        seeds = json.load(stream)
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Seed manifest must be a non-empty JSON list of integers: {resolved}")
    return seeds or fallback


def train_sac(
    train_seeds_file: Optional[str] = None,
    validation_seeds_file: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
):
    config_path = os.path.join(PROJECT_DIR, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config: Dict = yaml.safe_load(stream) or {}
    env_config = dict(config.get("environment", {}))
    if env_config.get("backend", "2d").lower() != "2d":
        raise ValueError("SAC trainer currently supports environment.backend: 2d only.")

    eval_cfg = config.get("evaluation", {})
    train_seeds = _read_seeds(train_seeds_file or eval_cfg.get("train_seeds_file", "experiments/seeds/train_seeds.json"), [1000])
    validation_seeds = _read_seeds(
        validation_seeds_file or eval_cfg.get("val_seeds_file", "experiments/seeds/validation_seeds.json"), [1100]
    )
    rl_cfg = config.get("rl", {})
    total_timesteps = int(total_timesteps if total_timesteps is not None else rl_cfg.get("total_timesteps", 50000))
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    configured_checkpoint = rl_cfg.get("checkpoint", "rl/checkpoints/sac_2d.zip")
    if not os.path.isabs(configured_checkpoint):
        configured_checkpoint = os.path.join(PROJECT_DIR, configured_checkpoint)
    checkpoint_dir = checkpoint_dir or os.path.dirname(configured_checkpoint)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env(seeds: List[int]):
        def factory():
            from environment.driving_2d_env import Driving2DEnv

            native_env = Driving2DEnv({**env_config, "render": False})
            return Monitor(CyclingSeedEnv(native_env, seeds))

        return factory

    env = make_env(train_seeds)()
    eval_env = DummyVecEnv([make_env(validation_seeds)])
    model = SAC(
        policy=rl_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(rl_cfg.get("buffer_size", 50000)),
        batch_size=int(rl_cfg.get("batch_size", 256)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        verbose=1,
        tensorboard_log=os.path.join(PROJECT_DIR, "rl", "tensorboard_logs"),
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, min(5000, total_timesteps // 10)),
        save_path=checkpoint_dir,
        name_prefix="sac_2d_checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=max(1, total_timesteps // 5),
        n_eval_episodes=len(validation_seeds),
        deterministic=True,
        render=False,
    )
    try:
        print(f"Training SAC for {total_timesteps} steps across {len(train_seeds)} train seeds.")
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
        best_path = os.path.join(checkpoint_dir, "best_model.zip")
        final_path = os.path.join(checkpoint_dir, "sac_2d.zip")
        if os.path.exists(best_path):
            import shutil

            shutil.copy2(best_path, final_path)
        else:
            model.save(final_path[:-4])
        try:
            evaluation_history = np.load(os.path.join(checkpoint_dir, "evaluations.npz"))
            timesteps = evaluation_history["timesteps"]
            rewards = evaluation_history["results"]
            means, stds = rewards.mean(axis=1), rewards.std(axis=1)
            plots_dir = os.path.join(PROJECT_DIR, "results", "plots")
            os.makedirs(plots_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(timesteps, means, color="darkorange", label="Mean validation reward")
            ax.fill_between(timesteps, means - stds, means + stds, color="darkorange", alpha=0.2)
            ax.set(title="2D SAC Validation Reward", xlabel="Training steps", ylabel="Episode reward")
            ax.grid(True)
            ax.legend()
            fig.savefig(os.path.join(plots_dir, "rl_training_curve.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
        except (OSError, KeyError, ValueError) as error:
            print(f"Could not create validation reward plot: {error}")
        print(f"Saved trained SAC policy to {final_path}")
        return final_path
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the SAC agent on the seeded 2D environment.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-seeds", default=None)
    parser.add_argument("--validation-seeds", default=None)
    args = parser.parse_args()
    train_sac(
        train_seeds_file=args.train_seeds,
        validation_seeds_file=args.validation_seeds,
        total_timesteps=args.timesteps,
    )
