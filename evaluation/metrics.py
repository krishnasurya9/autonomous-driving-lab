"""Episode metrics for both the legacy and the current flat telemetry schema."""

import json
from typing import Any, Dict

import numpy as np


def compute_episode_metrics(telemetry_file_path: str) -> Dict[str, Any]:
    with open(telemetry_file_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    meta = data.get("experiment_metadata", data)
    summary = data.get("summary_metrics", data)
    series = data.get("time_series_telemetry", data.get("steps", [])) or []
    target_speed = float(meta.get("target_speed_kmh", 40.0))
    speeds = [float(step.get("speed_kmh", 0.0)) for step in series]
    lanes = [float(step.get("lane_deviation_m", 0.0)) for step in series]
    steering = [float(step.get("steering", 0.0)) for step in series]

    rmse_lane = float(np.sqrt(np.mean(np.square(lanes)))) if lanes else 0.0
    mae_speed = float(np.mean(np.abs(np.asarray(speeds) - target_speed))) if speeds else 0.0
    steer_smoothness = float(np.mean(np.abs(np.diff(steering)))) if len(steering) > 1 else 0.0
    steps = len(series)
    return {
        "scenario_id": meta.get("scenario_id", f"SCENARIO_{meta.get('seed', 'unknown')}"),
        "seed": meta.get("seed"),
        "agent_type": str(meta.get("agent_type", "unknown")).lower(),
        "agent_policies": summary.get("agent_policies", []),
        "task_completed": bool(summary.get("task_completed", False)),
        "completion_time_sec": float(summary.get("completion_time_sec", steps * 0.05)),
        "terminated_reason": summary.get("terminated_reason"),
        "total_collisions": int(summary.get("total_collisions", summary.get("collisions", 0))),
        "off_road_events": int(summary.get("off_road_events", 1 if summary.get("off_road") else 0)),
        "rmse_lane_deviation_m": round(rmse_lane, 4),
        "mae_speed_error_kmh": round(mae_speed, 4),
        "control_smoothness_steering": round(steer_smoothness, 4),
        "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "avg_decision_latency_sec": float(summary.get("avg_decision_latency_sec", 0.0)),
        "invalid_llm_outputs": int(summary.get("invalid_llm_outputs", 0)),
    }
