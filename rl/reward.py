"""Configurable multi-objective driving reward shared by train and evaluation."""

from pathlib import Path

import numpy as np
import yaml


class DrivingReward:
    def __init__(self, target_speed_kmh: float = 40.0, config_path: str = "config.yaml"):
        self.target_speed_kmh = float(target_speed_kmh)
        defaults = {
            "w_progress": 1.0, "w_lane": 1.5, "w_speed": 1.0,
            "w_safety": 1.0, "w_task": 10.0, "crash_penalty": -10.0,
            "out_of_road_penalty": -10.0, "progress_bonus": 2.0,
            "steering_penalty": -0.1,
        }
        path = Path(config_path)
        if path.exists():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                defaults.update(loaded.get("reward_weights", {}))
            except (OSError, yaml.YAMLError, AttributeError):
                pass
        for key, value in defaults.items():
            setattr(self, key, float(value))

    def compute_reward(
        self,
        speed_kmh: float,
        lane_deviation_m: float,
        heading_error_deg: float,
        crashed: bool,
        out_of_road: bool,
        arrived_at_destination: bool,
        route_progress: float = 0.0,
        steering: float = 0.0,
    ) -> float:
        target = max(self.target_speed_kmh, 1.0)
        r_progress = speed_kmh / target + route_progress * self.progress_bonus
        r_lane = -(abs(lane_deviation_m) ** 1.5) - 0.1 * abs(np.radians(heading_error_deg))
        r_speed = 1.0 - min(abs(speed_kmh - target) / target, 1.0)
        r_speed += self.steering_penalty * steering**2
        r_safety = (self.crash_penalty if crashed else 0.0) + (self.out_of_road_penalty if out_of_road else 0.0)
        r_task = 5.0 if arrived_at_destination else 0.0
        return float(
            self.w_progress * r_progress + self.w_lane * r_lane
            + self.w_speed * r_speed + self.w_safety * r_safety + self.w_task * r_task
        )
