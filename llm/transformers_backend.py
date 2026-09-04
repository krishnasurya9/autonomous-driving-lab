import time
from typing import Tuple
from llm.backend import LLMBackend

class TransformersBackend(LLMBackend):
    """Fallback local rule-based / mock backend when local server is unavailable."""

    def __init__(self, model_name: str = "local-rule-engine"):
        super().__init__(model_name=model_name)

    def generate_decision_raw(self, prompt_text: str) -> Tuple[str, float]:
        start = time.time()
        # Rule-based fallback mimicking small instruction model outputs
        if "OBSTACLE: YES" in prompt_text.upper():
            json_out = '{"target_speed_kmh": 20.0, "maneuver": "SLOW_DOWN", "lane_target": "CENTER"}'
        elif "HIGH" in prompt_text.upper() or "CURVE" in prompt_text.upper():
            json_out = '{"target_speed_kmh": 30.0, "maneuver": "SLOW_DOWN", "lane_target": "CENTER"}'
        else:
            json_out = '{"target_speed_kmh": 40.0, "maneuver": "MAINTAIN", "lane_target": "CENTER"}'

        latency = time.time() - start
        return json_out, latency
