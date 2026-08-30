import time
import threading
from typing import Any, Dict, Tuple, Optional
import numpy as np

from agents.base_agent import BaseAgent
from environment.state_adapter import StateAdapter, DrivingState
from controller.decision_api import HighLevelDecision, MANEUVER_MAINTAIN
from controller.low_level_controller import LowLevelController
from llm.prompts import generate_user_prompt
from llm.output_parser import OutputParser
from llm.backend import LLMBackend
from llm.ollama_backend import OllamaBackend
from llm.lmstudio_backend import LMStudioBackend
from llm.transformers_backend import TransformersBackend

class LLMAgent(BaseAgent):
    """
    LLM Driving Agent.
    Operates hierarchically: high-level LLM planner updates decision at decision_frequency_hz,
    low-level controller executes continuous control actions every simulator frame.
    """

    def __init__(
        self,
        backend_type: str = "ollama",
        model_name: str = "qwen2.5:7b",
        decision_frequency_hz: float = 1.0,
        sim_step_dt: float = 0.02,  # 50 Hz simulator default
        target_speed_kmh: float = 40.0,
    ):
        super().__init__(name="LLM")
        self.decision_frequency_hz = decision_frequency_hz
        self.decision_interval_steps = max(1, int((1.0 / decision_frequency_hz) / sim_step_dt))
        
        # Instantiate backend adapter
        if backend_type.lower() == "ollama":
            self.backend: LLMBackend = OllamaBackend(model_name=model_name)
        elif backend_type.lower() == "lmstudio":
            self.backend = LMStudioBackend(model_name=model_name)
        else:
            self.backend = TransformersBackend(model_name=model_name)

        self.state_adapter = StateAdapter(target_speed_kmh=target_speed_kmh)
        self.controller = LowLevelController()
        
        self.current_decision = HighLevelDecision(target_speed_kmh=target_speed_kmh, maneuver=MANEUVER_MAINTAIN)
        self.step_counter = 0
        self.last_latency = 0.0
        self.last_invalid = False
        self._decision_lock = threading.Lock()
        self._llm_thread: Optional[threading.Thread] = None

    def reset(self):
        self.step_counter = 0
        self.current_decision = HighLevelDecision(target_speed_kmh=self.state_adapter.target_speed_kmh, maneuver=MANEUVER_MAINTAIN)
        self.last_latency = 0.0
        self.last_invalid = False
        self._llm_thread = None

    def _request_decision_async(self, prompt_text: str) -> None:
        if self._llm_thread is not None and self._llm_thread.is_alive():
            return

        def _worker() -> None:
            raw_response, latency = self.backend.generate_decision_raw(prompt_text)
            if not raw_response:
                fb_backend = TransformersBackend()
                raw_response, latency = fb_backend.generate_decision_raw(prompt_text)
            decision, invalid = OutputParser.parse_response(raw_response, fallback_speed=self.state_adapter.target_speed_kmh)
            with self._decision_lock:
                self.current_decision = decision
                self.last_latency = latency
                self.last_invalid = invalid

        self._llm_thread = threading.Thread(target=_worker, daemon=True)
        self._llm_thread.start()

    def get_action(self, env: Any, obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        state: DrivingState = self.state_adapter.extract_state(env)

        # Trigger LLM high-level decision at decision interval (non-blocking for render loop)
        if self.step_counter % self.decision_interval_steps == 0:
            prompt_text = generate_user_prompt(state.to_prompt_text())
            self._request_decision_async(prompt_text)

        with self._decision_lock:
            active_decision = self.current_decision
            active_latency = self.last_latency
            active_invalid = self.last_invalid

        # Compute continuous low-level action
        action = self.controller.compute_action(
            decision=active_decision,
            current_speed_kmh=state.speed_kmh,
            lane_deviation_m=state.lane_deviation_m,
            heading_error_deg=state.heading_error_deg,
        )

        info = {
            "agent_type": "LLM",
            "decision": active_decision.to_dict(),
            "llm_latency_sec": active_latency,
            "llm_invalid_output": active_invalid,
        }

        self.step_counter += 1
        return action, info
