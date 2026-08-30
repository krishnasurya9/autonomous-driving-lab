import numpy as np

class DrivingReward:
    """
    Multi-objective reward function for RL driving agent:
    R = w1 * R_progress + w2 * R_lane + w3 * R_speed + w4 * R_safety + w5 * R_task
    """

    def __init__(
        self,
        w_progress: float = 1.0,
        w_lane: float = 1.5,
        w_speed: float = 1.0,
        w_safety: float = 5.0,
        w_task: float = 10.0,
        target_speed_kmh: float = 40.0,
    ):
        self.w_progress = w_progress
        self.w_lane = w_lane
        self.w_speed = w_speed
        self.w_safety = w_safety
        self.w_task = w_task
        self.target_speed_kmh = target_speed_kmh

    def compute_reward(
        self,
        speed_kmh: float,
        lane_deviation_m: float,
        heading_error_deg: float,
        crashed: bool,
        out_of_road: bool,
        arrived_at_destination: bool,
    ) -> float:
        # Progress reward (movement in km/h normalized)
        r_progress = speed_kmh / self.target_speed_kmh

        # Lane deviation penalty
        r_lane = -abs(lane_deviation_m) - 0.1 * abs(np.radians(heading_error_deg))

        # Speed compliance reward
        speed_diff = abs(speed_kmh - self.target_speed_kmh)
        r_speed = 1.0 - min(speed_diff / self.target_speed_kmh, 1.0)

        # Safety penalties
        r_safety = 0.0
        if crashed:
            r_safety -= 2.0
        if out_of_road:
            r_safety -= 2.0

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
