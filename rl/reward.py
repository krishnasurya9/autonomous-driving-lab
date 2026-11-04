import numpy as np
import yaml
import os

class DrivingReward:
    """
    Multi-objective reward function for RL driving agent:
    R = w1 * R_progress + w2 * R_lane + w3 * R_speed + w4 * R_safety + w5 * R_task
    """

    def __init__(
        self,
        target_speed_kmh: float = 40.0,
    ):
        self.target_speed_kmh = target_speed_kmh
        
        self.w_progress = 1.0
        self.w_lane = 1.5
        self.w_speed = 1.0
        self.w_safety = 1.0
        self.w_task = 10.0
        self.crash_penalty = -10.0
        self.out_of_road_penalty = -10.0
        self.progress_bonus = 2.0
        self.steering_penalty = -0.1

        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                rw = config.get("reward_weights", {})
                self.w_progress = rw.get("w_progress", self.w_progress)
                self.w_lane = rw.get("w_lane", self.w_lane)
                self.w_speed = rw.get("w_speed", self.w_speed)
                self.w_safety = rw.get("w_safety", self.w_safety)
                self.w_task = rw.get("w_task", self.w_task)
                self.crash_penalty = rw.get("crash_penalty", self.crash_penalty)
                self.out_of_road_penalty = rw.get("out_of_road_penalty", self.out_of_road_penalty)
                self.progress_bonus = rw.get("progress_bonus", self.progress_bonus)
                self.steering_penalty = rw.get("steering_penalty", self.steering_penalty)
        except Exception:
            pass

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
        # Progress reward (movement in km/h normalized)
        r_progress = (speed_kmh / self.target_speed_kmh) + (route_progress * self.progress_bonus)

        # Lane deviation penalty
        r_lane = -(abs(lane_deviation_m) ** 1.5) - 0.1 * abs(np.radians(heading_error_deg))

        # Speed compliance reward & smooth driving bonus
        speed_diff = abs(speed_kmh - self.target_speed_kmh)
        r_speed = 1.0 - min(speed_diff / self.target_speed_kmh, 1.0)
        r_speed += self.steering_penalty * (steering ** 2)

        # Safety penalties
        r_safety = 0.0
        if crashed:
            r_safety += self.crash_penalty
        if out_of_road:
            r_safety += self.out_of_road_penalty

        # Task completion reward
        r_task = 5.0 if arrived_at_destination else 0.0

        total_reward = (
            self.w_progress * r_progress
            + self.w_lane * r_lane
            + self.w_speed * r_speed
            + self.w_safety * r_safety
            + self.w_task * r_task
        )
        return float(total_reward)
