import json
import numpy as np
from typing import Dict, Any, List

def compute_episode_metrics(telemetry_file_path: str) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics from a telemetry JSON file."""
    with open(telemetry_file_path, "r") as f:
        data = json.load(f)

    meta = data["experiment_metadata"]
    summary = data["summary_metrics"]
    series = data["time_series_telemetry"]

    if not series:
        return summary

    target_speed = meta.get("target_speed_kmh", 40.0)
    steer_actions = [s["steering"] for s in series]
    speeds = [s["speed_kmh"] for s in series]
    lane_devs = [s["lane_deviation_m"] for s in series]

    # RMSE Lane Deviation
    rmse_lane = float(np.sqrt(np.mean(np.square(lane_devs))))

    # MAE Speed Error
    mae_speed = float(np.mean([abs(v - target_speed) for v in speeds]))

    # Steering Smoothness J_steering = (1 / (T-1)) * sum(|u_{t+1} - u_t|)
    if len(steer_actions) > 1:
        steer_diffs = np.abs(np.diff(steer_actions))
        j_steering = float(np.mean(steer_diffs))
    else:
        j_steering = 0.0

    metrics = {
        "scenario_id": meta["scenario_id"],
        "seed": meta["seed"],
        "agent_type": meta["agent_type"],
        "task_completed": summary.get("task_completed", False),
        "completion_time_sec": summary.get("completion_time_sec", 0.0),
        "total_collisions": summary.get("total_collisions", 0),
        "off_road_events": summary.get("off_road_events", 0),
        "rmse_lane_deviation_m": round(rmse_lane, 4),
        "mae_speed_error_kmh": round(mae_speed, 4),
        "control_smoothness_steering": round(j_steering, 4),
        "avg_speed_kmh": round(float(np.mean(speeds)), 2),
        "avg_decision_latency_sec": summary.get("avg_decision_latency_sec", 0.0),
        "invalid_llm_outputs": summary.get("invalid_llm_outputs", 0),
    }

    return metrics
