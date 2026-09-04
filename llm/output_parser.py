import json
import re
from typing import Tuple, Dict, Any
from controller.decision_api import HighLevelDecision, MANEUVER_MAINTAIN

class OutputParser:
    """Parses and validates LLM raw response text into HighLevelDecision with fallbacks."""

    @staticmethod
    def parse_response(raw_text: str, fallback_speed: float = 40.0) -> Tuple[HighLevelDecision, bool]:
        """
        Parses raw text from LLM.
        Returns (HighLevelDecision, is_invalid_flag).
        """
        if not raw_text or not isinstance(raw_text, str):
            return HighLevelDecision(target_speed_kmh=fallback_speed, maneuver=MANEUVER_MAINTAIN), True

        # Try extract JSON substring if wrapped in markdown ```json ... ```
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_text.strip()

        try:
            data = json.loads(json_str)
            decision = HighLevelDecision.from_dict(data)
            return decision, False
        except Exception:
            # Fallback on parsing error
            fallback_decision = HighLevelDecision(
                target_speed_kmh=fallback_speed,
                maneuver=MANEUVER_MAINTAIN,
                lane_target="CENTER"
            )
            return fallback_decision, True
