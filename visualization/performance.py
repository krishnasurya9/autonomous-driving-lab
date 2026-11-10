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
        ("completion_time_sec", "Successful Completion Time (s)"),
        ("final_route_progress", "Final Route Progress"),
        ("total_collisions", "Mean Collision Count"),
        ("off_road_events", "Mean Road Departure Count"),
    ]

    for col, label in metrics:
        if col in df.columns:
            plot_data = df
            if col == "completion_time_sec":
                plot_data = df[df["task_completed"].astype(str).str.lower().isin(["true", "1"])]
                if plot_data.empty:
                    print("[Visualization] No successful episodes yet; skipping completion-time plot.")
                    continue
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=plot_data,
                x="agent_type",
                y=col,
                errorbar="sd",
                hue="agent_type",
                palette="viridis",
                legend=False,
                ax=ax,
            )
            ax.set_title(f"Agent Comparison: {label}")
            ax.set_ylabel(label)
            ax.set_xlabel("Agent Type")
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            out_file = os.path.join(output_dir, f"bar_{col}.png")
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

    if "task_completed" in df.columns:
        completion = df.assign(task_completed=df["task_completed"].astype(str).str.lower().isin(["true", "1"]))
        completion = completion.groupby("agent_type", as_index=False)["task_completed"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=completion, x="agent_type", y="task_completed", hue="agent_type", palette="viridis", legend=False, ax=ax)
        ax.set(title="Task Completion Rate", xlabel="Agent Type", ylabel="Fraction completed", ylim=(0, 1))
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        fig.savefig(os.path.join(output_dir, "bar_task_completion_rate.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[Visualization] Performance comparison plots saved to: {output_dir}")

if __name__ == "__main__":
    plot_performance_comparison()
