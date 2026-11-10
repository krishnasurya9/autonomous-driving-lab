import json
import numpy as np
from typing import Dict, Any, List

def compute_episode_metrics(telemetry_file_path: str) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics from a telemetry JSON file."""
    with open(telemetry_file_path, "r") as f:
        data = json.load(f)

    meta = data.get("experiment_metadata", data)
    summary = data.get("summary_metrics", data)
    series = data.get("time_series_telemetry", data.get("steps", []))

    if not series:
        return summary

    target_speed = float(meta.get("target_speed_kmh", 40.0))
    steer_actions = np.asarray([s.get("steering", 0.0) for s in series], dtype=np.float64)
    speeds = np.asarray([s.get("speed_kmh", 0.0) for s in series], dtype=np.float64)
    lane_devs = np.asarray([s.get("lane_deviation_m", 0.0) for s in series], dtype=np.float64)
    if not (np.isfinite(steer_actions).all() and np.isfinite(speeds).all() and np.isfinite(lane_devs).all()):
        raise ValueError(f"Telemetry contains non-finite values: {telemetry_file_path}")

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
        "scenario_id": meta.get("scenario_id", f"SCENARIO_{meta.get('seed', 'unknown')}"),
        "seed": int(meta["seed"]),
        "agent_type": str(meta.get("agent_type", "unknown")).lower(),
        "task_completed": summary.get("task_completed", False),
        "completion_time_sec": summary.get("completion_time_sec", len(series) * float(meta.get("step_duration_sec", 0.05))),
        "total_collisions": summary.get("total_collisions", summary.get("collisions", 0)),
        "off_road_events": summary.get("off_road_events", 1 if summary.get("off_road") else 0),
        "rmse_lane_deviation_m": round(rmse_lane, 4),
        "mae_speed_error_kmh": round(mae_speed, 4),
        "control_smoothness_steering": round(j_steering, 4),
        "avg_speed_kmh": round(float(np.mean(speeds)), 2),
        "avg_decision_latency_sec": summary.get("avg_decision_latency_sec", 0.0),
        "invalid_llm_outputs": summary.get("invalid_llm_outputs", 0),
        "final_route_progress": float(summary.get("final_route_progress", series[-1].get("route_progress", 0.0))),
        "terminated_reason": summary.get("terminated_reason"),
        "llm_real_inference": bool(summary.get("llm_real_inference", False)),
        "llm_backends": summary.get("llm_backends", []),
        "agent_policies": summary.get("agent_policies", []),
    }
    if metrics["agent_type"].lower() == "rl":
        metrics["rl_real_policy"] = "SAC" in metrics["agent_policies"]
    else:
        metrics["rl_real_policy"] = None

    return metrics
