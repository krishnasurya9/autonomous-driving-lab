import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np

# Patch NumPy 2.x removed deprecated aliases for legacy MetaDrive compatibility
# np.float must be np.float64 (not plain float) so MetaDrive's isinstance checks pass
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "bool"):
    np.bool = np.bool_

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from environment.state_adapter import StateAdapter
from rl.reward import DrivingReward

class CustomMetaDriveEnv(gym.Env):
    """
    Gymnasium Wrapper around MetaDrive for RL training & unified agent evaluation.
    Provides compact normalized state observations and custom multi-objective rewards.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.start_seed = self.config.get("seed", 1000)
        self.target_speed_kmh = self.config.get("target_speed_kmh", 40.0)

        # 7-dimensional normalized state vector
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(7,), dtype=np.float32
        )

        # Action space: [steering, throttle_brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        self.state_adapter = StateAdapter(target_speed_kmh=self.target_speed_kmh)
        self.reward_fn = DrivingReward(target_speed_kmh=self.target_speed_kmh)
        self._env = None

    def _lazy_init_env(self, seed: int):
        if self._env is None:
            # Patch panda3d-gltf >= 1.3.0 which removed patch_loader (MetaDrive calls it on engine init)
            import gltf as _gltf
            if not hasattr(_gltf, "patch_loader"):
                _gltf.patch_loader = lambda loader=None: None

            from metadrive.envs.metadrive_env import MetaDriveEnv
            env_config = {
                "start_seed": seed,
                "environment_num": 1,
                "use_render": self.config.get("render", False),
                "manual_control": self.config.get("manual_control", False),
                "traffic_density": self.config.get("traffic_density", 0.1),
            }
            self._env = MetaDriveEnv(env_config)

    def _get_obs_vector(self) -> np.ndarray:
        state = self.state_adapter.extract_state(self._env)
        
        # Curvature code numeric
        curv_map = {"STRAIGHT": 0.0, "LOW": 0.33, "MODERATE": 0.66, "HIGH": 1.0}
        c_code = curv_map.get(state.road_curvature, 0.0)

        obs_vec = np.array([
            state.speed_kmh / 80.0,                           # normalized speed
            state.target_speed_kmh / 80.0,                    # normalized target speed
            np.clip(state.lane_deviation_m / 3.0, -2.0, 2.0), # normalized lane dev
            np.clip(state.heading_error_deg / 45.0, -2.0, 2.0),# normalized heading error
            state.obstacle_distance / 100.0,                  # normalized obstacle dist
            state.route_progress,                              # progress [0, 1]
            c_code,                                           # road curvature
        ], dtype=np.float32)

        return obs_vec

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        target_seed = seed if seed is not None else self.start_seed
        self._lazy_init_env(target_seed)
        try:
            res = self._env.reset(force_seed=target_seed)
        except TypeError:
            res = self._env.reset()

        if isinstance(res, tuple):
            info = res[1] if len(res) > 1 else {}
        else:
            info = {}

        obs_vec = self._get_obs_vector()
        return obs_vec, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        step_res = self._env.step(action)
        if len(step_res) == 5:
            obs, r, terminated, truncated, info = step_res
        elif len(step_res) == 4:
            obs, r, done, info = step_res
            terminated, truncated = done, False
        else:
            r, terminated, truncated, info = 0.0, False, False, {}
        
        state = self.state_adapter.extract_state(self._env)
        crashed = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        arrived = info.get("arrive_dest", False)

        reward = self.reward_fn.compute_reward(
            speed_kmh=state.speed_kmh,
            lane_deviation_m=state.lane_deviation_m,
            heading_error_deg=state.heading_error_deg,
            crashed=crashed,
            out_of_road=out_of_road,
            arrived_at_destination=arrived,
        )

        obs_vec = self._get_obs_vector()
        return obs_vec, reward, terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
