import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import json
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
import numpy as np

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
    
    def make_env():
        return Monitor(CustomMetaDriveEnv(env_config))
        
    env = make_env()
    eval_env = SubprocVecEnv([make_env])

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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=checkpoint_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    print(f"Training SAC model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    final_path = os.path.join(checkpoint_dir, "sac_metadrive.zip")
    best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
    
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, final_path)
    else:
        model.save(final_path)

    # Plot training curve
    try:
        results = np.load(os.path.join(checkpoint_dir, "evaluations.npz"))
        timesteps = results["timesteps"]
        results_mean = results["results"].mean(axis=1)
        os.makedirs("results/plots", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, results_mean)
        plt.title("RL Training Curve (Evaluation Reward)")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.grid()
        plt.savefig("results/plots/rl_training.png")
        plt.close()
    except Exception as e:
        print(f"Could not plot training curve: {e}")

    print(f"SAC model training complete! Model saved to: {final_path}")
    print("=" * 60)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    args = parser.parse_args()
    train_sac(total_timesteps=args.timesteps)
