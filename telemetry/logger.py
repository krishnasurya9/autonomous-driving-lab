import os
import json
import time

class TelemetryLogger:
    def __init__(self, agent_type, scenario_id, seed, output_dir="telemetry_data"):
        self.agent_type = agent_type
        self.scenario_id = scenario_id
        self.seed = seed
        self.output_dir = output_dir
        self.steps = []
        self.collisions = 0
        self.off_road = False
        os.makedirs(self.output_dir, exist_ok=True)

    def record_collision(self):
        self.collisions += 1

    def record_off_road(self):
        self.off_road = True

    def log_step(self, step, pos_x, pos_y, speed_kmh, steering, throttle, brake, lane_deviation_m, heading_error_deg, **kwargs):
        self.steps.append({
            "step": step,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "speed_kmh": speed_kmh,
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "lane_deviation_m": lane_deviation_m,
            "heading_error_deg": heading_error_deg,
            **kwargs
        })

    def save(self, task_completed=False, terminated_reason=None):
        timestamp = int(time.time() * 1000)
        filename = f"{self.agent_type}_{self.scenario_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        llm_backends = sorted({
            step.get("llm_backend") for step in self.steps
            if step.get("llm_backend") not in (None, "", "pending")
        })
        llm_latencies = [
            step["llm_latency_sec"] for step in self.steps
            if isinstance(step.get("llm_latency_sec"), (int, float)) and step["llm_latency_sec"] > 0
        ]
        policies = sorted({step.get("agent_policy") for step in self.steps if step.get("agent_policy")})
        
        data = {
            "agent_type": self.agent_type,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "target_speed_kmh": 40.0,
            "task_completed": task_completed,
            "terminated_reason": terminated_reason,
            "collisions": self.collisions,
            "off_road_events": int(self.off_road),
            "episode_steps": len(self.steps),
            "completion_time_sec": len(self.steps) * 0.05,
            "agent_policies": policies,
            "llm_backends": llm_backends,
            "llm_real_inference": any(backend != "rule_fallback" for backend in llm_backends) if self.agent_type.lower() == "llm" else None,
            "avg_decision_latency_sec": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0,
            "invalid_llm_outputs": sum(bool(step.get("llm_invalid_output", False)) for step in self.steps),
            "steps": self.steps
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
        return filepath
