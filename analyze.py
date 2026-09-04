import sys
import os

sys.path.insert(0, os.path.abspath("."))

from evaluation.compare_agents import compare_telemetry_results
from visualization.trajectories import plot_trajectory_overlay
from visualization.performance import plot_performance_comparison
from visualization.speed import plot_speed_profiles
from visualization.rl_training import plot_rl_learning_curve

def main():
    print("=" * 60)
    print("RUNNING STATISTICAL ANALYSIS & PLOT GENERATION PIPELINE")
    print("=" * 60)

    # 1. Statistical paired comparisons & metric CSV generation
    compare_telemetry_results(telemetry_dir="telemetry", output_results_dir="results")

    # 2. Generate visualization plots
    plot_performance_comparison(csv_path="results/summary_metrics.csv", output_dir="results/plots")
    plot_trajectory_overlay(telemetry_dir="telemetry", output_dir="results/plots")
    plot_speed_profiles(telemetry_dir="telemetry", output_dir="results/plots")
    plot_rl_learning_curve(output_dir="results/plots")

    print("=" * 60)
    print("ANALYSIS AND VISUALIZATION COMPLETE! Results saved to results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
