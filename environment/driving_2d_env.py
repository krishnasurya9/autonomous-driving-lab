"""Lightweight top-down driving simulator with a Gymnasium-compatible API."""

from __future__ import annotations

import math
import time
import tkinter as tk
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.state_adapter import DrivingState
from rl.reward import DrivingReward


class Driving2DEnv(gym.Env):
    """Seeded single-lane road driving task for humans and continuous-control agents.

    The environment exposes an 8-value observation and 2-value continuous action,
    shared by the controller, human interface, and SAC training pipeline.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}
    dt = 0.05

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.start_seed = int(self.config.get("seed", 1000))
        self.target_speed_kmh = float(self.config.get("target_speed_kmh", 40.0))
        self.lane_width_m = float(self.config.get("lane_width_m", 3.5))
        self.route_length_m = float(self.config.get("route_length_m", 500.0))
        self.horizon = int(self.config.get("horizon", 1000))
        self.render_enabled = bool(self.config.get("render", False))

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_fn = DrivingReward(target_speed_kmh=self.target_speed_kmh)

        self._root = None
        self._canvas = None
        self._keys = set()
        self._window_closed = False
        self._seed = self.start_seed
        self._phase_a = 0.0
        self._phase_b = 0.0
        self._obstacle_s = 180.0
        self._obstacle_offset = 0.0
        self.progress_m = 0.0
        self.speed_kmh = 0.0
        self.lane_deviation_m = 0.0
        self.heading_error_deg = 0.0
        self.last_steering = 0.0
        self.step_count = 0
        self.collisions = 0
        self.out_of_road = False
        self.terminated_reason = None

    def _road_x(self, s: float) -> float:
        return 4.0 * math.sin(s * 0.010 + self._phase_a) + 2.0 * math.sin(s * 0.025 + self._phase_b)

    def _road_heading(self, s: float) -> float:
        return math.atan2(self._road_x(s + 0.5) - self._road_x(s - 0.5), 1.0)

    def _observation(self) -> np.ndarray:
        heading = math.radians(self.heading_error_deg)
        curvature = float(np.clip((self._road_heading(self.progress_m + 8.0) - self._road_heading(self.progress_m - 8.0)) * 2.0, -1.0, 1.0))
        obstacle_dist = max(0.0, self._obstacle_s - self.progress_m)
        if self._obstacle_s < self.progress_m:
            obstacle_dist = 100.0
        return np.asarray([
            self.speed_kmh / 80.0,
            self.target_speed_kmh / 80.0,
            np.clip(self.lane_deviation_m / 3.0, -2.0, 2.0),
            np.clip(self.heading_error_deg / 45.0, -2.0, 2.0),
            min(obstacle_dist, 100.0) / 100.0,
            np.clip(self.progress_m / self.route_length_m, 0.0, 1.0),
            curvature,
            np.clip(self._obstacle_offset / self.lane_width_m, -1.0, 1.0)
            if obstacle_dist < 100.0 else 0.0,
        ], dtype=np.float32)

    def get_driving_state(self) -> DrivingState:
        dist = self._obstacle_s - self.progress_m
        obstacle_present = 0.0 <= dist <= 100.0
        heading_abs = abs(self.heading_error_deg)
        curvature_name = "HIGH" if heading_abs > 15 else "MODERATE" if heading_abs > 5 else "LOW" if heading_abs > 2 else "STRAIGHT"
        maneuver = "LEFT" if self.heading_error_deg < -5 else "RIGHT" if self.heading_error_deg > 5 else "STRAIGHT"
        return DrivingState(
            speed_kmh=self.speed_kmh,
            target_speed_kmh=self.target_speed_kmh,
            lane_deviation_m=self.lane_deviation_m,
            heading_error_deg=self.heading_error_deg,
            road_curvature=curvature_name,
            upcoming_maneuver=maneuver,
            distance_to_maneuver=20.0 if maneuver != "STRAIGHT" else 50.0,
            obstacle_present=obstacle_present,
            obstacle_distance=max(0.0, dist) if obstacle_present else 100.0,
            route_progress=float(np.clip(self.progress_m / self.route_length_m, 0.0, 1.0)),
            distance_to_goal=max(0.0, self.route_length_m - self.progress_m),
            obstacle_lateral_offset_m=self._obstacle_offset if obstacle_present else 0.0,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else int(seed)
        rng = np.random.default_rng(self._seed)
        self._phase_a = float(rng.uniform(-math.pi, math.pi))
        self._phase_b = float(rng.uniform(-math.pi, math.pi))
        self._obstacle_s = float(rng.uniform(150.0, 240.0))
        self._obstacle_offset = float(rng.choice([-1.0, 1.0]) * self.lane_width_m * 0.2)
        self.progress_m = 0.0
        self.speed_kmh = 0.0
        self.lane_deviation_m = 0.0
        self.heading_error_deg = 0.0
        self.last_steering = 0.0
        self.step_count = 0
        self.collisions = 0
        self.out_of_road = False
        self.terminated_reason = None
        self._window_closed = False
        if self.render_enabled:
            self._ensure_window()
            self.render()
        return self._observation(), {"seed": self._seed}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        steering = float(np.clip(action[0] if len(action) else 0.0, -1.0, 1.0))
        longitudinal = float(np.clip(action[1] if len(action) > 1 else 0.0, -1.0, 1.0))

        old_progress = self.progress_m
        acceleration = (5.0 if longitudinal >= 0.0 else 9.0) * longitudinal
        self.speed_kmh = float(np.clip(self.speed_kmh + acceleration * 3.6 * self.dt, 0.0, 100.0))
        speed_mps = self.speed_kmh / 3.6
        self.heading_error_deg += steering * 65.0 * self.dt
        self.heading_error_deg = float(np.clip(self.heading_error_deg, -80.0, 80.0))
        self.progress_m += max(0.0, speed_mps * math.cos(math.radians(self.heading_error_deg))) * self.dt
        self.lane_deviation_m += speed_mps * math.sin(math.radians(self.heading_error_deg)) * self.dt
        self.heading_error_deg -= math.degrees(self._road_heading(self.progress_m) - self._road_heading(old_progress))
        self.last_steering = steering
        self.step_count += 1

        obstacle_hit = (
            abs(self.progress_m - self._obstacle_s) < 2.0
            and abs(self.lane_deviation_m - self._obstacle_offset) < 1.15
        )
        self.collisions = int(obstacle_hit)
        self.out_of_road = abs(self.lane_deviation_m) > self.lane_width_m * 0.5 + 0.6
        arrived = self.progress_m >= self.route_length_m
        terminated = bool(obstacle_hit or self.out_of_road or arrived)
        truncated = self.step_count >= self.horizon and not terminated
        if obstacle_hit:
            self.terminated_reason = "collision"
        elif self.out_of_road:
            self.terminated_reason = "off_road"
        elif arrived:
            self.terminated_reason = "arrived_dest"
        elif truncated:
            self.terminated_reason = "time_limit"

        state = self.get_driving_state()
        reward = self.reward_fn.compute_reward(
            speed_kmh=state.speed_kmh,
            lane_deviation_m=state.lane_deviation_m,
            heading_error_deg=state.heading_error_deg,
            crashed=obstacle_hit,
            out_of_road=self.out_of_road,
            arrived_at_destination=arrived,
            route_progress=state.route_progress,
            steering=steering,
        )
        info = {
            "crash": obstacle_hit,
            "out_of_road": self.out_of_road,
            "arrive_dest": arrived,
            "terminated_reason": self.terminated_reason,
            "route_progress": state.route_progress,
            "obstacle_distance": state.obstacle_distance,
            "obstacle_lateral_offset_m": state.obstacle_lateral_offset_m,
            "collisions": self.collisions,
            "seed": self._seed,
        }
        if self.render_enabled:
            self.render()
        return self._observation(), reward, terminated, truncated, info

    def get_human_action(self) -> np.ndarray:
        if self._window_closed:
            return np.zeros(2, dtype=np.float32)
        self._ensure_window()
        self._pump_events()
        steer = float("d" in self._keys or "right" in self._keys) - float("a" in self._keys or "left" in self._keys)
        steer *= 0.15
        throttle = float("w" in self._keys or "up" in self._keys)
        brake = float("s" in self._keys or "down" in self._keys or "space" in self._keys)
        return np.asarray([steer, throttle - brake], dtype=np.float32)

    def _ensure_window(self):
        if self._window_closed:
            return
        if self._root is not None:
            return
        self._root = tk.Tk()
        self._root.title(f"Autonomous Driving Lab · 2D · Seed {self._seed}")
        self._root.geometry("1000x720")
        self._root.configure(bg="#111827")
        self._root.protocol("WM_DELETE_WINDOW", self._close_window)
        self._canvas = tk.Canvas(self._root, width=1000, height=720, bg="#263746", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._root.focus_force()
        self._root.bind("<KeyPress>", self._key_down)
        self._root.bind("<KeyRelease>", self._key_up)

    def _key_down(self, event):
        key = event.keysym.lower()
        self._keys.add(key)
        if key == "escape":
            self._close_window()

    def _key_up(self, event):
        self._keys.discard(event.keysym.lower())

    def _close_window(self):
        self._window_closed = True
        if self._root is not None:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
            self._root = None
            self._canvas = None

    def _pump_events(self):
        if self._root is not None:
            try:
                self._root.update_idletasks()
                self._root.update()
            except tk.TclError:
                self._window_closed = True

    def render(self):
        if not self.render_enabled and self._root is None:
            return None
        self._ensure_window()
        self._pump_events()
        if self._canvas is None:
            return None

        canvas = self._canvas
        canvas.delete("all")
        width = max(700, canvas.winfo_width())
        height = max(500, canvas.winfo_height())
        scale = 10.0
        car_y = int(height * 0.70)
        center_now = self._road_x(self.progress_m)
        road_points = []
        for s in np.linspace(self.progress_m - 35.0, self.progress_m + 90.0, 70):
            x = width * 0.5 + (self._road_x(float(s)) - center_now) * scale
            y = car_y - (float(s) - self.progress_m) * scale
            road_points.extend((x, y))
        canvas.create_line(*road_points, fill="#68717e", width=int(self.lane_width_m * scale * 2.5), smooth=True)
        canvas.create_line(*road_points, fill="#e5e7eb", width=2, dash=(12, 14), smooth=True)

        obstacle_delta = self._obstacle_s - self.progress_m
        if -35.0 <= obstacle_delta <= 90.0:
            ox = width * 0.5 + (self._obstacle_offset - self.lane_deviation_m) * scale
            oy = car_y - obstacle_delta * scale
            canvas.create_oval(ox - 10, oy - 10, ox + 10, oy + 10, fill="#f97316", outline="#ffedd5", width=2)
            canvas.create_text(ox + 28, oy, text="obstacle", fill="white", anchor="w", font=("Segoe UI", 9))

        car_x = width * 0.5
        points = [(-14, 20), (-12, -18), (0, -27), (12, -18), (14, 20)]
        angle = math.radians(self.heading_error_deg)
        rotated = []
        for px, py in points:
            rotated.extend((car_x + px * math.cos(angle) - py * math.sin(angle), car_y + px * math.sin(angle) + py * math.cos(angle)))
        canvas.create_polygon(*rotated, fill="#38bdf8", outline="#e0f2fe", width=2)
        canvas.create_text(20, 18, anchor="nw", fill="white", font=("Segoe UI", 12, "bold"),
                           text=f"2D DRIVE  ·  {self._seed}\nSpeed  {self.speed_kmh:4.0f} / {self.target_speed_kmh:.0f} km/h\nLane   {self.lane_deviation_m:+.2f} m\nRoute  {100 * self.progress_m / self.route_length_m:4.1f}%")
        canvas.create_text(width - 18, height - 18, anchor="se", fill="#d1d5db", font=("Segoe UI", 10),
                           text="W/↑ accelerate   S/↓/Space brake   Tap A/← or D/→ to steer   Esc quit")
        self._pump_events()
        return None

    def show_episode_result(self, reason: Optional[str], route_progress: float) -> None:
        """Keep a rendered human episode's result visible until the user closes it."""
        if not self.render_enabled or self._root is None or self._canvas is None:
            return
        message = f"Episode ended: {reason or 'incomplete'}\nRoute: {100.0 * route_progress:.1f}%\nPress Esc or close this window to continue."
        self._canvas.create_rectangle(190, 255, 810, 425, fill="#111827", outline="#e5e7eb", width=2)
        self._canvas.create_text(500, 340, text=message, fill="white", font=("Segoe UI", 18, "bold"), justify="center")
        while not self._window_closed:
            self._pump_events()
            if not self._window_closed:
                time.sleep(0.05)

    def close(self):
        self._close_window()
