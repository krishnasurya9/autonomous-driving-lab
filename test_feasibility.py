"""Small manual smoke check for the 2D environment; run after setup."""

from environment.driving_2d_env import Driving2DEnv


def main():
    env = Driving2DEnv({"seed": 2037, "render": False, "horizon": 100})
    obs, info = env.reset(seed=2037)
    print(f"2D environment reset: observation shape={obs.shape}, seed={info['seed']}")
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step([0.0, 0.6])
        if terminated or truncated:
            break
    state = env.get_driving_state()
    print(
        f"steps={step + 1}, speed={state.speed_kmh:.1f} km/h, "
        f"lane={state.lane_deviation_m:.2f} m, reason={info.get('terminated_reason')}"
    )
    env.close()


if __name__ == "__main__":
    main()
