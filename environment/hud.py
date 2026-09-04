import os
from typing import Dict, Any, Optional

class HUDManager:
    """Manages real-time Heads-Up Display (HUD) overlays on MetaDrive 3D visual window."""

    def __init__(self, agent_type: str, seed: int, target_speed_kmh: float = 40.0):
        self.agent_type = agent_type.upper()
        self.seed = seed
        self.target_speed_kmh = target_speed_kmh
        self.onscreen_text = None
        self.warning_text = None
        self._initialized = False
        self._last_text = ""
        self._last_warning = ""
        import time
        self._last_update_time = time.time()

    def _init_onscreen_text(self):
        if self._initialized:
            return
        try:
            from direct.gui.OnscreenText import OnscreenText
            from panda3d.core import TextNode
            self.onscreen_text = OnscreenText(
                text="",
                pos=(-1.3, 0.9),
                scale=0.045,
                fg=(1, 1, 1, 1),
                bg=(0, 0, 0, 0.6),
                align=TextNode.ALeft,
                mayChange=True,
            )
            self.warning_text = OnscreenText(
                text="",
                pos=(0.0, 0.5),
                scale=0.1,
                fg=(1, 0, 0, 1),
                align=TextNode.ACenter,
                mayChange=True,
                shadow=(0, 0, 0, 1),
            )
            self._initialized = True
        except Exception:
            self._initialized = False

    def update(
        self,
        speed_kmh: float,
        lane_deviation_m: float,
        heading_error_deg: float,
        route_progress: float,
        collisions: int = 0,
        maneuver: Optional[str] = None,
        camera_name: str = "CHASE",
        off_road: bool = False,
    ):
        import time
        if time.time() - self._last_update_time < 0.1 and not off_road:
            return
        self._last_update_time = time.time()

        hud_lines = [
            f"=== AUTONOMOUS DRIVING LAB ===",
            f"MODE: {self.agent_type:<10} | SEED: {self.seed}",
            f"----------------------------------",
            f"Speed:       {speed_kmh:5.1f} / {self.target_speed_kmh:.0f} km/h",
            f"Lane Dev:    {lane_deviation_m:5.2f} m",
            f"Heading Err: {heading_error_deg:5.1f} deg",
            f"Progress:    {route_progress * 100:5.1f}%",
            f"Collisions:  {collisions}",
        ]

        if maneuver:
            hud_lines.append(f"Maneuver:    {maneuver}")

        if self.agent_type == "HUMAN":
            hud_lines.append("Controls:    W/A/S/D or Arrow Keys | Space: Brake")
        
        hud_lines.append(f"Camera:      {camera_name} (Press 'C' to change)")
        hud_lines.append("Press [ESC] to Exit Episode")

        if off_road:
            hud_lines.append("")
            hud_lines.append("!!! OFF ROAD — EPISODE ENDED !!!")

        hud_text = "\n".join(hud_lines)

        self._init_onscreen_text()
        if self.onscreen_text is not None:
            if hud_text != self._last_text:
                try:
                    self.onscreen_text.setText(hud_text)
                    self._last_text = hud_text
                except Exception:
                    pass
            
            if self.warning_text is not None:
                warn_msg = ""
                if off_road:
                    warn_msg = "CRITICAL: OFF ROAD!"
                elif collisions > 0:
                    warn_msg = "WARNING: COLLISION DETECTED!"
                
                # Make it flash
                if warn_msg and int(time.time() * 5) % 2 == 0:
                    warn_msg = ""

                if warn_msg != self._last_warning:
                    try:
                        self.warning_text.setText(warn_msg)
                        self._last_warning = warn_msg
                    except Exception:
                        pass
        else:
            # Fallback stdout print
            pass

    def cleanup(self):
        if self.onscreen_text is not None:
            try:
                self.onscreen_text.destroy()
                if self.warning_text is not None:
                    self.warning_text.destroy()
            except Exception:
                pass
            self.onscreen_text = None
            self.warning_text = None
        self._initialized = False
