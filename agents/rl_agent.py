import os
from typing import Any, Dict, Tuple, Optional
import numpy as np

from agents.base_agent import BaseAgent
from environment.state_adapter import StateAdapter
from controller.decision_api import HighLevelDecision, MANEUVER_MAINTAIN
from controller.low_level_controller import LowLevelController

class RLAgent(BaseAgent):
    """
    Continuous Reinforcement Learning Agent executing a trained SAC policy.
    Outputs continuous (steering, throttle_brake) directly.
    """

    def __init__(self, checkpoint_path: str = "rl/checkpoints/sac_metadrive.zip", target_speed_kmh: float = 40.0):
        super().__init__(name="RL")
        self.checkpoint_path = checkpoint_path
        self.state_adapter = StateAdapter(target_speed_kmh=target_speed_kmh)
        self.fallback_controller = LowLevelController()
        self.fallback_decision = HighLevelDecision(target_speed_kmh=target_speed_kmh, maneuver=MANEUVER_MAINTAIN)
        self.model = None

        if os.path.exists(checkpoint_path):
            from stable_baselines3 import SAC
            print(f"[RLAgent] Loading SAC model from checkpoint: {checkpoint_path}")
            self.model = SAC.load(checkpoint_path)
        else:
            print(f"[RLAgent] Warning: Checkpoint {checkpoint_path} not found. Operating with fallback baseline policy (no torch load).")

    def get_action(self, env: Any, obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is not None:
            # Predict action from trained policy
            action, _states = self.model.predict(obs, deterministic=True)
            action = np.array(action, dtype=np.float32)
        else:
            # Fallback deterministic smooth policy (same controller as LLM agent)
            state = self.state_adapter.extract_state(env)
            action = self.fallback_controller.compute_action(
                decision=self.fallback_decision,
                current_speed_kmh=state.speed_kmh,
                lane_deviation_m=state.lane_deviation_m,
                heading_error_deg=state.heading_error_deg,
            )

        info = {"agent_type": "RL", "policy": "SAC"}
        return action, info
