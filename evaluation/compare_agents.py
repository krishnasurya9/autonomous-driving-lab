import os
import sys
import json
import glob
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))

from evaluation.metrics import compute_episode_metrics
from evaluation.statistical_tests import run_paired_tests

def compare_telemetry_results(
    telemetry_dir: str = "telemetry",
    output_results_dir: str = "results",
) -> Dict[str, Any]:
    """Reads telemetry for Human, LLM, and RL, matches seeds, computes paired statistics."""
    os.makedirs(output_results_dir, exist_ok=True)

    agents = ["human", "llm", "rl"]
    results_by_agent: Dict[str, Dict[int, Dict[str, Any]]] = {a: {} for a in agents}

    for agent in agents:
        agent_dir = os.path.join(telemetry_dir, agent)
        if not os.path.exists(agent_dir):
            continue
        json_files = glob.glob(os.path.join(agent_dir, "seed_*.json"))
        for fpath in json_files:
            metrics = compute_episode_metrics(fpath)
            seed = metrics["seed"]
            results_by_agent[agent][seed] = metrics

    # Find common seeds across agents
    common_seeds = sorted(
        list(
            set(results_by_agent["human"].keys())
            & set(results_by_agent["llm"].keys())
            & set(results_by_agent["rl"].keys())
        )
    )

    if not common_seeds:
        # Fallback to seeds common between any available agents
        common_seeds = sorted(
            list(set(results_by_agent["llm"].keys()) & set(results_by_agent["rl"].keys()))
        )

    print(f"[Comparison] Found {len(common_seeds)} paired scenario seeds for analysis.")

    rows = []
    for seed in common_seeds:
        for agent in agents:
            if seed in results_by_agent[agent]:
                m = results_by_agent[agent][seed]
                rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_results_dir, "summary_metrics.csv"), index=False)

    # Perform paired tests for key metrics
    metric_keys = [
        "rmse_lane_deviation_m",
        "mae_speed_error_kmh",
        "control_smoothness_steering",
        "completion_time_sec",
    ]

    comparisons = [("rl", "llm"), ("rl", "human"), ("llm", "human")]
    stats_results = []

    for agent_a, agent_b in comparisons:
        for m_key in metric_keys:
            arr_a = [results_by_agent[agent_a][s][m_key] for s in common_seeds if s in results_by_agent[agent_a] and s in results_by_agent[agent_b]]
            arr_b = [results_by_agent[agent_b][s][m_key] for s in common_seeds if s in results_by_agent[agent_a] and s in results_by_agent[agent_b]]

            if arr_a and arr_b:
                test_res = run_paired_tests(arr_a, arr_b, metric_name=m_key)
                test_res["comparison"] = f"{agent_a.upper()} vs {agent_b.upper()}"
                stats_results.append(test_res)

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_results_dir, "paired_statistical_tests.csv"), index=False)

    report_path = os.path.join(output_results_dir, "comparison_report.json")
    summary_report = {
        "n_paired_seeds": len(common_seeds),
        "common_seeds": common_seeds,
        "statistical_tests": stats_results,
    }
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"[Comparison] Summary CSV and statistical test results saved to: {output_results_dir}")
    return summary_report

if __name__ == "__main__":
    compare_telemetry_results()
