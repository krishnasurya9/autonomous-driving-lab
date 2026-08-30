import numpy as np
from controller.decision_api import (
    HighLevelDecision,
    MANEUVER_ACCELERATE,
    MANEUVER_MAINTAIN,
    MANEUVER_SLOW_DOWN,
    MANEUVER_BRAKE,
    MANEUVER_TURN_LEFT,
    MANEUVER_TURN_RIGHT,
    MANEUVER_PREPARE_LEFT,
    MANEUVER_PREPARE_RIGHT,
    MANEUVER_STOP,
)

class LowLevelController:
    """
    Translates HighLevelDecision + current state into vehicle control signals:
    steering in [-1, 1], throttle in [0, 1], brake in [0, 1].
    
    Uses deterministic proportional speed and lateral tracking control.
    """

    def __init__(self, kp_speed: float = 0.05, kp_steer: float = 0.8, kd_steer: float = 0.1):
        self.kp_speed = kp_speed
        self.kp_steer = kp_steer
        self.kd_steer = kd_steer
        self.prev_heading_error = 0.0

    def compute_action(
        self,
        decision: HighLevelDecision,
        current_speed_kmh: float,
        lane_deviation_m: float,
        heading_error_deg: float,
    ) -> np.ndarray:
        """
        Returns np.array([steering, throttle_brake]) as MetaDrive expects,
        where throttle_brake > 0 is throttle and < 0 is brake.
        """
        maneuver = decision.maneuver
        target_speed = decision.target_speed_kmh

        # Adjust target speed based on maneuver
        if maneuver == MANEUVER_ACCELERATE:
            target_speed += 10.0
        elif maneuver in (MANEUVER_SLOW_DOWN, MANEUVER_PREPARE_LEFT, MANEUVER_PREPARE_RIGHT):
            target_speed = max(10.0, target_speed - 10.0)
        elif maneuver in (MANEUVER_BRAKE, MANEUVER_STOP):
            target_speed = 0.0

        # Speed Control (Proportional)
        speed_error = target_speed - current_speed_kmh
        if maneuver == MANEUVER_STOP or target_speed <= 1.0:
            throttle_brake = -1.0 if current_speed_kmh > 1.0 else 0.0
        else:
            throttle_brake = np.clip(self.kp_speed * speed_error, -1.0, 1.0)

        # Lateral / Steering Control
        # Base lateral offset target from lane_target
        lateral_offset = 0.0
        if decision.lane_target == "LEFT" or maneuver in (MANEUVER_TURN_LEFT, MANEUVER_PREPARE_LEFT):
            lateral_offset = -0.5
        elif decision.lane_target == "RIGHT" or maneuver in (MANEUVER_TURN_RIGHT, MANEUVER_PREPARE_RIGHT):
            lateral_offset = 0.5

        effective_lane_dev = lane_deviation_m - lateral_offset
        heading_error_rad = np.radians(heading_error_deg)

        steer_p = -self.kp_steer * (effective_lane_dev * 0.3 + heading_error_rad)
        steer_d = -self.kd_steer * (heading_error_deg - self.prev_heading_error)
        self.prev_heading_error = heading_error_deg

        steering = np.clip(steer_p + steer_d, -1.0, 1.0)

        return np.array([float(steering), float(throttle_brake)], dtype=np.float32)
