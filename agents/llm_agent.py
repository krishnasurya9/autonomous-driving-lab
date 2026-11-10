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
        endpoint: Optional[str] = None,
        temperature: float = 0.1,
    ):
        super().__init__(name="LLM")
        self.decision_frequency_hz = decision_frequency_hz
        self.decision_interval_steps = max(1, int((1.0 / decision_frequency_hz) / sim_step_dt))
        
        # Instantiate backend adapter
        if backend_type.lower() == "ollama":
            self.backend: LLMBackend = OllamaBackend(
                model_name=model_name,
                endpoint=endpoint or "http://localhost:11434/api/generate",
                temperature=temperature,
            )
        elif backend_type.lower() == "lmstudio":
            self.backend = LMStudioBackend(
                model_name=model_name,
                endpoint=endpoint or "http://127.0.0.1:1234/api/v1/chat",
                temperature=temperature,
            )
        else:
            self.backend = TransformersBackend(model_name=model_name)

        self.state_adapter = StateAdapter(target_speed_kmh=target_speed_kmh)
        self.controller = LowLevelController()
        
        self.current_decision = HighLevelDecision(target_speed_kmh=target_speed_kmh, maneuver=MANEUVER_MAINTAIN)
        self.step_counter = 0
        self.last_latency = 0.0
        self.last_invalid = False
        self.last_backend = "pending"
        self._decision_lock = threading.Lock()
        self._llm_thread: Optional[threading.Thread] = None

    def reset(self):
        self.step_counter = 0
        self.current_decision = HighLevelDecision(target_speed_kmh=self.state_adapter.target_speed_kmh, maneuver=MANEUVER_MAINTAIN)
        self.last_latency = 0.0
        self.last_invalid = False
        self.last_backend = "pending"
        self._llm_thread = None

    def wait_for_pending_decision(self) -> None:
        """Wait briefly for the last asynchronous planner call before saving telemetry."""
        if self._llm_thread is not None and self._llm_thread.is_alive():
            timeout = float(getattr(self.backend, "timeout_sec", 5.0)) + 1.0
            self._llm_thread.join(timeout=timeout)

    def _request_decision_async(self, prompt_text: str) -> None:
        if self._llm_thread is not None and self._llm_thread.is_alive():
            return

        def _worker() -> None:
            raw_response, latency = self.backend.generate_decision_raw(prompt_text)
            backend_used = "rule_fallback" if isinstance(self.backend, TransformersBackend) else self.backend.__class__.__name__
            if not raw_response:
                fb_backend = TransformersBackend()
                raw_response, fallback_latency = fb_backend.generate_decision_raw(prompt_text)
                latency += fallback_latency
                backend_used = "rule_fallback"
            decision, invalid = OutputParser.parse_response(raw_response, fallback_speed=self.state_adapter.target_speed_kmh)
            with self._decision_lock:
                self.current_decision = decision
                self.last_latency = latency
                self.last_invalid = invalid
                self.last_backend = backend_used

        self._llm_thread = threading.Thread(target=_worker, daemon=True)
        self._llm_thread.start()

    def get_action(self, env: Any, obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        state: DrivingState = self.state_adapter.extract_state(env)

        # In headless evaluation, simulation steps are not paced to real time.
        # Wait at the next planner tick for the previous async request so the
        # LLM gets its configured decision interval and telemetry records it.
        if self.step_counter % self.decision_interval_steps == 0:
            if self.step_counter and not getattr(env, "render_enabled", False):
                if self._llm_thread is not None and self._llm_thread.is_alive():
                    timeout = float(getattr(self.backend, "timeout_sec", 5.0)) + 1.0
                    self._llm_thread.join(timeout=timeout)
            prompt_text = generate_user_prompt(state.to_prompt_text())
            self._request_decision_async(prompt_text)

        with self._decision_lock:
            active_decision = self.current_decision
            active_latency = self.last_latency
            active_invalid = self.last_invalid
            active_backend = self.last_backend

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
            # Record a latency once at a planner tick, not once per simulator frame.
            "llm_latency_sec": active_latency if self.step_counter % self.decision_interval_steps == 0 and active_backend != "pending" else None,
            "llm_invalid_output": active_invalid,
            "llm_backend": active_backend,
        }

        self.step_counter += 1
        return action, info
