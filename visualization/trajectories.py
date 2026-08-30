import os
import glob
import json
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_trajectory_overlay(telemetry_dir: str = "telemetry", output_dir: str = "results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    agents = ["human", "llm", "rl"]
    colors = {"human": "green", "llm": "blue", "rl": "red"}

    # Find common seeds
    seeds_by_agent: Dict[str, Dict[int, str]] = {a: {} for a in agents}
    for agent in agents:
        for fpath in glob.glob(os.path.join(telemetry_dir, agent, "seed_*.json")):
            with open(fpath, "r") as f:
                d = json.load(f)
                seed = d["experiment_metadata"]["seed"]
                seeds_by_agent[agent][seed] = fpath

    all_seeds = set(seeds_by_agent["human"].keys()) | set(seeds_by_agent["llm"].keys()) | set(seeds_by_agent["rl"].keys())

    for seed in all_seeds:
        plt.figure(figsize=(10, 6))
        plt.title(f"2D Trajectory Comparison - Seed {seed}")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid(True)

        plotted = False
        for agent in agents:
            if seed in seeds_by_agent[agent]:
                with open(seeds_by_agent[agent][seed], "r") as f:
                    data = json.load(f)
                series = data.get("time_series_telemetry", [])
                xs = [s["pos_x"] for s in series]
                ys = [s["pos_y"] for s in series]
                if xs and ys:
                    plt.plot(xs, ys, label=agent.upper(), color=colors[agent], linewidth=2.0)
                    plotted = True

        if plotted:
            plt.legend()
            out_file = os.path.join(output_dir, f"trajectory_seed_{seed}.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"[Visualization] Trajectory plots saved to: {output_dir}")

if __name__ == "__main__":
    plot_trajectory_overlay()
