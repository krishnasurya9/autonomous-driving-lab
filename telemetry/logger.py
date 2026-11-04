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
        timestamp = int(time.time())
        filename = f"{self.agent_type}_{self.scenario_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "agent_type": self.agent_type,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "task_completed": task_completed,
            "terminated_reason": terminated_reason,
            "steps": self.steps
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
        return filepath
