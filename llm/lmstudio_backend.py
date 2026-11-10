import time
import requests
from typing import Tuple
from llm.backend import LLMBackend
from llm.prompts import SYSTEM_PROMPT

class LMStudioBackend(LLMBackend):
    """Adapter for LM Studio's OpenAI-compatible REST API."""

    def __init__(
        self,
        model_name: str = "local-model",
        endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
        temperature: float = 0.1,
        timeout_sec: float = 60.0,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def generate_decision_raw(self, prompt_text: str) -> Tuple[str, float]:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "temperature": self.temperature,
            "max_tokens": 150,
        }

        start = time.time()
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout_sec)
            latency = time.time() - start
            if response.status_code == 200:
                raw_text = response.json()["choices"][0]["message"]["content"].strip()
                return raw_text, latency
            else:
                return "", latency
        except Exception:
            latency = time.time() - start
            return "", latency
