import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance_comparison(csv_path: str = "results/summary_metrics.csv", output_dir: str = "results/plots"):
    if not os.path.exists(csv_path):
        print(f"[Visualization] {csv_path} not found. Run analysis first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    metrics = [
        ("rmse_lane_deviation_m", "RMSE Lane Deviation (m)"),
        ("mae_speed_error_kmh", "MAE Speed Error (km/h)"),
        ("control_smoothness_steering", "Control Smoothness J_steering"),
        ("completion_time_sec", "Completion Time (s)"),
    ]

    for col, label in metrics:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=df, x="agent_type", y=col, ci="sd", palette="viridis")
            plt.title(f"Agent Comparison: {label}")
            plt.ylabel(label)
            plt.xlabel("Agent Type")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            out_file = os.path.join(output_dir, f"bar_{col}.png")
            plt.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close()

    print(f"[Visualization] Performance comparison plots saved to: {output_dir}")

if __name__ == "__main__":
    plot_performance_comparison()
