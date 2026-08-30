from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for all driving agents (Human, LLM, RL)."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_action(self, env: Any, obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Computes continuous action np.array([steering, throttle_brake])
        and returns action + info dictionary containing any metadata/decision logs.
        """
        pass

    def reset(self):
        """Called at the beginning of a new episode."""
        pass
