"""Repeatable smoke verification for the active 2D Gymnasium environment."""

import numpy as np

from environment.driving_2d_env import Driving2DEnv
from environment.state_adapter import StateAdapter


def _rollout(env, seed):
    observations, rewards, flags = [], [], []
    obs, _ = env.reset(seed=seed)
    observations.append(obs.copy())
    for step in range(env.horizon):
        action = np.asarray([0.0, 0.6], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())
        rewards.append(reward)
        flags.append((terminated, truncated, info.get("terminated_reason")))
        assert env.observation_space.contains(obs), f"invalid observation at step {step}"
        assert np.isfinite(reward), f"non-finite reward at step {step}"
        if terminated or truncated:
            break
    return observations, rewards, flags, info


def main():
    env = Driving2DEnv({"seed": 2037, "render": False, "horizon": 100})
    try:
        first = _rollout(env, 2037)
        second = _rollout(env, 2037)
        assert len(first[0][0]) == 8
        assert all(np.array_equal(a, b) for a, b in zip(first[0], second[0])), "seeded observations differ"
        assert first[1] == second[1], "seeded rewards differ"
        assert first[2] == second[2], "seeded termination sequence differs"
        state = StateAdapter(target_speed_kmh=40.0).extract_state(env)
        assert np.isfinite([
            state.speed_kmh,
            state.lane_deviation_m,
            state.heading_error_deg,
            state.route_progress,
            state.obstacle_lateral_offset_m,
        ]).all()
        assert first[2][-1][1] is True or first[2][-1][0] is True, "episode did not finish at its horizon"
        env.reset(seed=2037)
        env.progress_m = env._obstacle_s - 25.0
        obstacle_observation = env._observation()
        assert np.isclose(
            obstacle_observation[-1], env._obstacle_offset / env.lane_width_m
        ), "signed obstacle lane offset was not encoded in observation"
        obstacle_state = env.get_driving_state()
        assert np.isclose(obstacle_state.obstacle_lateral_offset_m, env._obstacle_offset)
        print(
            "2D feasibility PASS: deterministic reset/rollout, 8-D observations, "
            f"finite state/reward, {len(first[1])} steps, reason={first[3].get('terminated_reason')}"
        )
    finally:
        env.close()

    keyboard_env = Driving2DEnv({"render": False})
    try:
        keyboard_env._ensure_window = lambda: None
        keyboard_env._pump_events = lambda: None
        keyboard_env._keys = {"w", "a"}
        assert np.allclose(keyboard_env.get_human_action(), [-0.15, 1.0])
        keyboard_env._keys = {"right", "space"}
        assert np.allclose(keyboard_env.get_human_action(), [0.15, -1.0])
        keyboard_env._keys = {"down"}
        assert np.allclose(keyboard_env.get_human_action(), [0.0, -1.0])
        print("Human keyboard mapping PASS: steering and throttle/brake keys map correctly")
    finally:
        keyboard_env.close()


if __name__ == "__main__":
    main()
