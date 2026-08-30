"""Compatibility shim: the 2D environment draws its own HUD on the Tk canvas."""


class HUDManager:
    def __init__(self, agent_type: str, seed: int, target_speed_kmh: float = 40.0):
        self.agent_type = agent_type.upper()
        self.seed = seed
        self.target_speed_kmh = target_speed_kmh

    def update(self, *args, **kwargs):
        """Kept for older callers; HUD rendering now belongs to Driving2DEnv."""

    def cleanup(self):
        return None
