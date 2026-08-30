from typing import Any, Dict, Tuple
import numpy as np
from agents.base_agent import BaseAgent

class HumanAgent(BaseAgent):
    """
    Playable Human Driver Agent.
    Captures live keyboard inputs (W/A/S/D or Arrow keys) in MetaDrive 3D mode.
    """

    def __init__(self):
        super().__init__(name="Human")

    def get_action(self, env: Any, obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        steering = 0.0
        throttle_brake = 0.0

        # Try MetaDrive engine controller
        if hasattr(env, "_env") and env._env is not None:
            raw_env = env._env
        else:
            raw_env = env

        if hasattr(raw_env, "engine") and raw_env.engine is not None and hasattr(raw_env.engine, "controller") and raw_env.engine.controller is not None:
            try:
                action = raw_env.engine.controller.get_action(raw_env.vehicle)
                steering = float(action[0])
                throttle_brake = float(action[1])
            except Exception:
                pass

        action_vec = np.array([steering, throttle_brake], dtype=np.float32)
        info = {"agent_type": "Human", "control_mode": "interactive_keyboard"}
        return action_vec, info
