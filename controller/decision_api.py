from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# Standardized maneuvers
MANEUVER_ACCELERATE = "ACCELERATE"
MANEUVER_MAINTAIN = "MAINTAIN"
MANEUVER_SLOW_DOWN = "SLOW_DOWN"
MANEUVER_BRAKE = "BRAKE"
MANEUVER_TURN_LEFT = "TURN_LEFT"
MANEUVER_TURN_RIGHT = "TURN_RIGHT"
MANEUVER_PREPARE_LEFT = "PREPARE_LEFT"
MANEUVER_PREPARE_RIGHT = "PREPARE_RIGHT"
MANEUVER_STOP = "STOP"

VALID_MANEUVERS = {
    MANEUVER_ACCELERATE,
    MANEUVER_MAINTAIN,
    MANEUVER_SLOW_DOWN,
    MANEUVER_BRAKE,
    MANEUVER_TURN_LEFT,
    MANEUVER_TURN_RIGHT,
    MANEUVER_PREPARE_LEFT,
    MANEUVER_PREPARE_RIGHT,
    MANEUVER_STOP,
}

@dataclass
class HighLevelDecision:
    """Standardized high-level decision structure across agents."""
    target_speed_kmh: float = 40.0
    maneuver: str = MANEUVER_MAINTAIN
    lane_target: str = "CENTER"  # CENTER, LEFT, RIGHT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HighLevelDecision":
        maneuver = str(d.get("maneuver", MANEUVER_MAINTAIN)).upper()
        if maneuver not in VALID_MANEUVERS:
            maneuver = MANEUVER_MAINTAIN
        
        try:
            target_speed = float(d.get("target_speed_kmh", 40.0))
        except (ValueError, TypeError):
            target_speed = 40.0

        lane_target = str(d.get("lane_target", "CENTER")).upper()
        return cls(
            target_speed_kmh=target_speed,
            maneuver=maneuver,
            lane_target=lane_target
        )
