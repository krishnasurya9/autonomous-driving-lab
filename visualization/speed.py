import glob
import json
import os

import matplotlib.pyplot as plt


def plot_speed_profiles(telemetry_dir: str = "telemetry_2d", output_dir: str = "results/plots"):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_dir, "experiments", "seeds", "test_seeds.json"), encoding="utf-8") as stream:
        allowed_seeds = {int(seed) for seed in json.load(stream)}
    os.makedirs(output_dir, exist_ok=True)
    agents = ["human", "llm", "rl"]
    colors = {"human": "green", "llm": "blue", "rl": "red"}
    files_by_agent = {agent: {} for agent in agents}
    for agent in agents:
        paths = glob.glob(os.path.join(telemetry_dir, f"{agent.upper()}_SCENARIO_*.json"))
        paths += glob.glob(os.path.join(telemetry_dir, agent, "seed_*.json"))
        for path in sorted(paths, key=os.path.getmtime):
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    data = json.load(stream)
                meta = data.get("experiment_metadata", data)
                seed = meta.get("seed")
                if seed is not None and int(seed) in allowed_seeds:
                    files_by_agent[agent][int(seed)] = data
            except (OSError, ValueError, TypeError):
                continue

    seeds = set().union(*(set(items) for items in files_by_agent.values()))
    for seed in sorted(seeds):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set(title=f"2D Speed Profile · Seed {seed}", xlabel="Time (seconds)", ylabel="Speed (km/h)")
        ax.grid(True)
        ax.axhline(y=40.0, color="gray", linestyle="--", label="Target (40 km/h)")
        for agent in agents:
            data = files_by_agent[agent].get(seed)
            if not data:
                continue
            series = data.get("time_series_telemetry", data.get("steps", []))
            times = [step.get("timestamp", step.get("step", i) * 0.05) for i, step in enumerate(series)]
            speeds = [step.get("speed_kmh", 0.0) for step in series]
            if times:
                ax.plot(times, speeds, label=agent.upper(), color=colors[agent], linewidth=1.8)
        ax.legend()
        fig.savefig(os.path.join(output_dir, f"speed_profile_seed_{seed}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"[Visualization] Speed profiles saved to: {output_dir}")


if __name__ == "__main__":
    plot_speed_profiles()
