import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_rl_learning_curve(log_dir: str = None, output_dir: str = "results/plots"):
    if log_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(project_dir, "config.yaml"), encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        checkpoint = config.get("rl", {}).get("checkpoint", "rl/checkpoints/sac_2d.zip")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(project_dir, checkpoint)
        log_dir = os.path.dirname(checkpoint)
    evaluation_file = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(evaluation_file):
        print(f"[Visualization] No real SAC evaluation history found at {evaluation_file}; skipping curve.")
        return None
    data = np.load(evaluation_file)
    timesteps = data["timesteps"]
    rewards = data["results"]
    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(timesteps, mean_rewards, color="darkorange", linewidth=2.0, label="Mean evaluation reward")
    ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, color="darkorange", alpha=0.2)
    ax.set(title="2D SAC Evaluation Reward", xlabel="Training timesteps", ylabel="Episode reward")
    ax.grid(True)
    ax.legend()
    out_file = os.path.join(output_dir, "rl_training_curve.png")
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] SAC evaluation curve saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    plot_rl_learning_curve()
