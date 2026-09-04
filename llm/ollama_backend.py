import time
import requests
from typing import Tuple
from llm.backend import LLMBackend
from llm.prompts import SYSTEM_PROMPT

class OllamaBackend(LLMBackend):
    """Adapter for local Ollama REST API endpoint (http://localhost:11434/api/generate)."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        endpoint: str = "http://localhost:11434/api/generate",
        temperature: float = 0.1,
        timeout_sec: float = 5.0,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def generate_decision_raw(self, prompt_text: str) -> Tuple[str, float]:
        payload = {
            "model": self.model_name,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt_text}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        start = time.time()
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
            latency = time.time() - start
            if response.status_code == 200:
                data = response.json()
                raw_text = data.get("response", "")
                return raw_text, latency
            else:
                return "", latency
        except Exception:
            latency = time.time() - start
            return "", latency
