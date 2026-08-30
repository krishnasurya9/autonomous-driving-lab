from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class DrivingState:
    """Standardized semantic driving state representation."""
    speed_kmh: float
    target_speed_kmh: float
    lane_deviation_m: float
    heading_error_deg: float
    road_curvature: str          # STRAIGHT, LOW, MODERATE, HIGH
    upcoming_maneuver: str       # STRAIGHT, LEFT, RIGHT, UNKNOWN
    distance_to_maneuver: float  # meters
    obstacle_present: bool
    obstacle_distance: float     # meters
    route_progress: float        # ratio [0.0, 1.0]
    distance_to_goal: float      # meters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_text(self) -> str:
        return (
            f"Current Driving State:\n"
            f"- Speed: {self.speed_kmh:.1f} km/h (Target: {self.target_speed_kmh:.1f} km/h)\n"
            f"- Lane Deviation: {self.lane_deviation_m:.2f} m\n"
            f"- Heading Error: {self.heading_error_deg:.1f} deg\n"
            f"- Road Curvature: {self.road_curvature}\n"
            f"- Upcoming Maneuver: {self.upcoming_maneuver} (in {self.distance_to_maneuver:.1f} m)\n"
            f"- Obstacle Present: {'YES' if self.obstacle_present else 'NONE'} "
            f"(Distance: {self.obstacle_distance:.1f} m)\n"
            f"- Route Progress: {self.route_progress * 100:.1f}% (Distance to Goal: {self.distance_to_goal:.1f} m)"
        )


class StateAdapter:
    """Extracts DrivingState from MetaDrive vehicle environment state."""

    def __init__(self, target_speed_kmh: float = 40.0):
        self.target_speed_kmh = target_speed_kmh

    def extract_state(self, env_or_obs: Any, vehicle: Optional[Any] = None) -> DrivingState:
        """Derives structured DrivingState from MetaDrive vehicle and environment objects."""
        if vehicle is None and hasattr(env_or_obs, "vehicle"):
            vehicle = env_or_obs.vehicle

        if vehicle is not None:
            speed_kmh = float(getattr(vehicle, "speed_kmh", getattr(vehicle, "speed", 0.0) * 3.6))
            
            # Distance to line / lane deviation
            try:
                lane = getattr(vehicle, "lane", None)
                if lane is not None:
                    long, lat = lane.local_coordinates(vehicle.position)
                    lane_deviation_m = float(lat)
                else:
                    lane_deviation_m = 0.0
            except Exception:
                lane_deviation_m = 0.0

            # Heading difference
            try:
                if hasattr(vehicle, "heading_diff") and hasattr(vehicle, "lane"):
                    heading_diff = vehicle.heading_diff(vehicle.lane)
                    heading_error_deg = float(np.degrees(heading_diff))
                else:
                    heading_error_deg = 0.0
            except Exception:
                heading_error_deg = 0.0

            # Route progress & goal distance
            try:
                navi = vehicle.navigation
                route_progress = float(getattr(navi, "route_completion", 0.0))
                distance_to_goal = float(getattr(navi, "total_length", 100.0) * (1.0 - route_progress))
            except Exception:
                route_progress = 0.0
                distance_to_goal = 100.0

            # Obstacles detection via lidar/surrounding vehicles
            try:
                surrounding = vehicle.lidar.get_surrounding_vehicles() if hasattr(vehicle, "lidar") else []
                obstacle_present = len(surrounding) > 0
                obstacle_distance = float(surrounding[0].distance) if obstacle_present else 100.0
            except Exception:
                obstacle_present = False
                obstacle_distance = 100.0

            # Curvature estimation
            road_curvature = "STRAIGHT"
            if abs(heading_error_deg) > 15:
                road_curvature = "HIGH"
            elif abs(heading_error_deg) > 5:
                road_curvature = "MODERATE"
            elif abs(heading_error_deg) > 2:
                road_curvature = "LOW"

            # Upcoming maneuver
            upcoming_maneuver = "STRAIGHT"
            distance_to_maneuver = 50.0
            if heading_error_deg < -5:
                upcoming_maneuver = "LEFT"
                distance_to_maneuver = 20.0
            elif heading_error_deg > 5:
                upcoming_maneuver = "RIGHT"
                distance_to_maneuver = 20.0

        else:
            # Fallback if vehicle object unavailable
            speed_kmh = 0.0
            lane_deviation_m = 0.0
            heading_error_deg = 0.0
            road_curvature = "STRAIGHT"
            upcoming_maneuver = "STRAIGHT"
            distance_to_maneuver = 50.0
            obstacle_present = False
            obstacle_distance = 100.0
            route_progress = 0.0
            distance_to_goal = 100.0

        return DrivingState(
            speed_kmh=speed_kmh,
            target_speed_kmh=self.target_speed_kmh,
            lane_deviation_m=lane_deviation_m,
            heading_error_deg=heading_error_deg,
            road_curvature=road_curvature,
            upcoming_maneuver=upcoming_maneuver,
            distance_to_maneuver=distance_to_maneuver,
            obstacle_present=obstacle_present,
            obstacle_distance=obstacle_distance,
            route_progress=route_progress,
            distance_to_goal=distance_to_goal,
        )
