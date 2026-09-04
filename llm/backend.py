from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_decision_raw(self, prompt_text: str) -> Tuple[str, float]:
        """
        Sends prompt to LLM server / model.
        Returns (raw_response_text, inference_latency_seconds).
        """
        pass
