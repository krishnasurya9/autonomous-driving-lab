"""Write the common, analysis-ready episode telemetry schema."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class TelemetryLogger:
    def __init__(
        self,
        agent_type: str,
        scenario_id: str,
        seed: int,
        output_dir: str = "telemetry_2d",
        target_speed_kmh: float = 40.0,
        step_duration_sec: float = 0.05,
        environment_backend: str = "2d",
        observation_dim: int = 8,
        action_dim: int = 2,
        policy_checkpoint: Optional[str] = None,
    ):
        self.agent_type = agent_type.upper()
        self.scenario_id = scenario_id
        self.seed = int(seed)
        self.target_speed_kmh = float(target_speed_kmh)
        self.step_duration_sec = float(step_duration_sec)
        self.environment_backend = environment_backend
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.policy_checkpoint = policy_checkpoint
        self.output_dir = output_dir
        self.steps = []
        self.collisions = 0
        self.off_road = False
        self.llm_backends = set()
        self.agent_policies = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def record_collision(self):
        self.collisions += 1

    def record_off_road(self):
        self.off_road = True

    def log_step(
        self,
        step,
        pos_x,
        pos_y,
        speed_kmh,
        steering,
        throttle,
        brake,
        lane_deviation_m,
        heading_error_deg,
        **kwargs,
    ):
        row = {
            "step": int(step),
            "timestamp": float(step) * self.step_duration_sec,
            "pos_x": float(pos_x),
            "pos_y": float(pos_y),
            "speed_kmh": float(speed_kmh),
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake),
            "lane_deviation_m": float(lane_deviation_m),
            "heading_error_deg": float(heading_error_deg),
        }
        row.update(kwargs)
        if row.get("llm_backend"):
            self.llm_backends.add(str(row["llm_backend"]))
        if row.get("agent_policy"):
            self.agent_policies.add(str(row["agent_policy"]))
        self.steps.append(row)

    def save(
        self,
        task_completed: bool = False,
        terminated_reason: Optional[str] = None,
        final_route_progress: Optional[float] = None,
    ) -> str:
        timestamp = int(time.time() * 1000)
        filepath = os.path.join(
            self.output_dir,
            f"{self.agent_type}_SCENARIO_{self.seed}_{timestamp}.json",
        )
        latencies = [
            float(row["llm_latency_sec"])
            for row in self.steps
            if row.get("llm_latency_sec") is not None
        ]
        data: Dict[str, Any] = {
            "agent_type": self.agent_type,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "target_speed_kmh": self.target_speed_kmh,
            "telemetry_schema_version": 2,
            "environment_backend": self.environment_backend,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "policy_checkpoint": self.policy_checkpoint,
            "task_completed": bool(task_completed),
            "terminated_reason": terminated_reason,
            "collisions": self.collisions,
            "off_road_events": int(self.off_road),
            "episode_steps": len(self.steps),
            "completion_time_sec": len(self.steps) * self.step_duration_sec,
            "final_route_progress": final_route_progress,
            "agent_policies": sorted(self.agent_policies),
            "llm_backends": sorted(self.llm_backends),
            "llm_real_inference": any(
                backend not in {"pending", "rule_fallback", "TransformersBackend"}
                for backend in self.llm_backends
            ),
            "avg_decision_latency_sec": sum(latencies) / len(latencies) if latencies else 0.0,
            "invalid_llm_outputs": sum(bool(row.get("llm_invalid_output")) for row in self.steps),
            "steps": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        return filepath
