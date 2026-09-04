import os
import glob
import json
import matplotlib.pyplot as plt

def plot_speed_profiles(telemetry_dir: str = "telemetry", output_dir: str = "results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    agents = ["human", "llm", "rl"]
    colors = {"human": "green", "llm": "blue", "rl": "red"}

    seeds_by_agent = {a: {} for a in agents}
    for agent in agents:
        for fpath in glob.glob(os.path.join(telemetry_dir, agent, "seed_*.json")):
            with open(fpath, "r") as f:
                d = json.load(f)
                seed = d["experiment_metadata"]["seed"]
                seeds_by_agent[agent][seed] = fpath

    all_seeds = set(seeds_by_agent["human"].keys()) | set(seeds_by_agent["llm"].keys()) | set(seeds_by_agent["rl"].keys())

    for seed in all_seeds:
        plt.figure(figsize=(10, 5))
        plt.title(f"Speed Profile vs. Time - Seed {seed}")
        plt.xlabel("Timestamp (seconds)")
        plt.ylabel("Speed (km/h)")
        plt.grid(True)
        plt.axhline(y=40.0, color="gray", linestyle="--", label="Target Speed (40 km/h)")

        plotted = False
        for agent in agents:
            if seed in seeds_by_agent[agent]:
                with open(seeds_by_agent[agent][seed], "r") as f:
                    data = json.load(f)
                series = data.get("time_series_telemetry", [])
                ts = [s["timestamp"] for s in series]
                speeds = [s["speed_kmh"] for s in series]
                if ts and speeds:
                    plt.plot(ts, speeds, label=agent.upper(), color=colors[agent], linewidth=1.8)
                    plotted = True

        if plotted:
            plt.legend()
            out_file = os.path.join(output_dir, f"speed_profile_seed_{seed}.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[Visualization] Speed profile plots saved to: {output_dir}")

if __name__ == "__main__":
    plot_speed_profiles()
