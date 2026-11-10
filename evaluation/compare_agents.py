"""Build episode summaries and paired tests from analysis-ready telemetry."""

import glob
import json
import os
from typing import Any, Dict

import pandas as pd

from evaluation.metrics import compute_episode_metrics
from evaluation.statistical_tests import run_paired_tests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compare_telemetry_results(
    telemetry_dir: str = "telemetry_2d",
    output_results_dir: str = "results",
    seed_manifest_path: str = "experiments/seeds/test_seeds.json",
) -> Dict[str, Any]:
    """Compare available agents by shared seed; fallback LLM runs are excluded."""
    os.makedirs(output_results_dir, exist_ok=True)
    agents = ["human", "llm", "rl"]
    eligible: Dict[str, Dict[int, Dict[str, Any]]] = {agent: {} for agent in agents}
    all_rows = []
    fallback_llm_count = 0
    fallback_rl_count = 0
    manifest_path = seed_manifest_path if os.path.isabs(seed_manifest_path) else os.path.join(PROJECT_DIR, seed_manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as stream:
        allowed_seeds = {int(seed) for seed in json.load(stream)}

    for agent in agents:
        paths = glob.glob(os.path.join(telemetry_dir, f"{agent.upper()}_SCENARIO_*.json"))
        paths += glob.glob(os.path.join(telemetry_dir, agent, "seed_*.json"))
        for path in sorted(paths, key=os.path.getmtime):
            metrics = compute_episode_metrics(path)
            if int(metrics["seed"]) not in allowed_seeds:
                continue
            metrics["source_file"] = os.path.basename(path)
            if agent == "llm" and not metrics.get("llm_real_inference", False):
                metrics["agent_type"] = "llm_fallback"
                fallback_llm_count += 1
                all_rows.append(metrics)
                continue
            if agent == "rl" and not metrics.get("rl_real_policy", False):
                metrics["agent_type"] = "rl_fallback"
                fallback_rl_count += 1
                all_rows.append(metrics)
                continue
            eligible[agent][int(metrics["seed"])] = metrics
            all_rows.append(metrics)

    summary_columns = [
        "scenario_id", "seed", "agent_type", "task_completed", "completion_time_sec",
        "total_collisions", "off_road_events", "rmse_lane_deviation_m", "mae_speed_error_kmh",
        "control_smoothness_steering", "avg_speed_kmh", "avg_decision_latency_sec",
        "invalid_llm_outputs", "final_route_progress", "terminated_reason", "llm_real_inference",
        "llm_backends", "agent_policies", "rl_real_policy", "source_file",
    ]
    pd.DataFrame(all_rows, columns=summary_columns).to_csv(os.path.join(output_results_dir, "summary_metrics.csv"), index=False)
    metric_keys = [
        "rmse_lane_deviation_m",
        "mae_speed_error_kmh",
        "control_smoothness_steering",
        "completion_time_sec",
        "final_route_progress",
    ]
    pair_reports = []
    for agent_a, agent_b in [("rl", "llm"), ("rl", "human"), ("llm", "human")]:
        paired_seeds = sorted(set(eligible[agent_a]) & set(eligible[agent_b]))
        for key in metric_keys:
            values_a = [eligible[agent_a][seed][key] for seed in paired_seeds]
            values_b = [eligible[agent_b][seed][key] for seed in paired_seeds]
            result = run_paired_tests(values_a, values_b, metric_name=key)
            result["comparison"] = f"{agent_a.upper()} vs {agent_b.upper()}"
            result["paired_seeds"] = paired_seeds
            pair_reports.append(result)

    pd.DataFrame(pair_reports).to_csv(os.path.join(output_results_dir, "paired_statistical_tests.csv"), index=False)
    paired_counts = {
        f"{a.upper()}_vs_{b.upper()}": len(set(eligible[a]) & set(eligible[b]))
        for a, b in [("rl", "llm"), ("rl", "human"), ("llm", "human")]
    }
    summary_report = {
        "telemetry_directory": telemetry_dir,
        "seed_manifest": os.path.relpath(manifest_path, PROJECT_DIR),
        "eligible_seed_range": sorted(allowed_seeds),
        "valid_real_llm_seeds": sorted(eligible["llm"]),
        "fallback_llm_episodes_excluded_from_comparisons": fallback_llm_count,
        "fallback_rl_episodes_excluded_from_comparisons": fallback_rl_count,
        "eligible_seeds_by_agent": {agent: sorted(records) for agent, records in eligible.items()},
        "paired_seed_counts": paired_counts,
        "statistical_tests": pair_reports,
        "interpretation": "Only real local-model LLM episodes are included in LLM comparisons. Rule-fallback episodes remain visible in summary_metrics.csv as llm_fallback.",
    }
    with open(os.path.join(output_results_dir, "comparison_report.json"), "w", encoding="utf-8") as stream:
        json.dump(summary_report, stream, indent=2, allow_nan=False)

    print(f"[Comparison] Paired seed counts: {paired_counts}; excluded fallback LLM episodes: {fallback_llm_count}.")
    return summary_report


if __name__ == "__main__":
    compare_telemetry_results()
