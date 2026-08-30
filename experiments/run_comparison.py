"""Run a shared-seed episode for a human, LLM planner, or SAC agent."""

import argparse
import json
from typing import List

from agents.human_agent import HumanAgent
from environment.driving_2d_env import Driving2DEnv
from environment.state_adapter import StateAdapter
from telemetry.logger import TelemetryLogger


def run_paired_experiment_on_seed(
    seed: int,
    agent_type: str,
    max_steps: int = 1000,
    render: bool = True,
    save_video: bool = False,
):
    agent_type = agent_type.lower()
    print(f"--- Seed {seed} · {agent_type.upper()} · 2D (render={render}) ---")

    # Load SAC before constructing the simulator to avoid the previous Windows DLL-order issue.
    if agent_type == "human":
        agent = HumanAgent()
    elif agent_type == "llm":
        from agents.llm_agent import LLMAgent
        agent = LLMAgent(backend_type="ollama", model_name="qwen2.5:7b")
    elif agent_type == "rl":
        from agents.rl_agent import RLAgent
        agent = RLAgent(checkpoint_path="rl/checkpoints/sac_2d.zip")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    env = Driving2DEnv({"seed": seed, "render": render, "target_speed_kmh": 40.0, "horizon": max_steps})
    obs, _ = env.reset(seed=seed)
    agent.reset()
    state_adapter = StateAdapter(target_speed_kmh=40.0)
    logger = TelemetryLogger(
        agent_type=agent_type.upper(),
        scenario_id=f"SCENARIO_{seed}",
        seed=seed,
        output_dir="telemetry_2d",
    )

    if save_video:
        print("2D video export is not configured yet; continuing without a recording.")

    env_info = {}
    terminated = truncated = False
    for step in range(max_steps):
        action, action_info = agent.get_action(env, obs)
        if env._window_closed:
            env_info = {"terminated_reason": "user_quit"}
            break
        obs, reward, terminated, truncated, env_info = env.step(action)
        state = state_adapter.extract_state(env)
        if env_info.get("crash"):
            logger.record_collision()
        if env_info.get("out_of_road"):
            logger.record_off_road()

        logger.log_step(
            step=step,
            pos_x=float(env._road_x(env.progress_m) + env.lane_deviation_m),
            pos_y=float(env.progress_m),
            speed_kmh=state.speed_kmh,
            steering=float(action[0]),
            throttle=float(max(float(action[1]), 0.0)),
            brake=float(max(-float(action[1]), 0.0)),
            lane_deviation_m=state.lane_deviation_m,
            heading_error_deg=state.heading_error_deg,
            route_progress=state.route_progress,
            agent_policy=action_info.get("policy", action_info.get("control_mode")),
            llm_decision=action_info.get("decision"),
            llm_backend=action_info.get("llm_backend"),
            llm_latency_sec=action_info.get("llm_latency_sec"),
            llm_invalid_output=action_info.get("llm_invalid_output", False),
        )
        if terminated or truncated:
            break

    completed = bool(env_info.get("arrive_dest", False))
    filepath = logger.save(
        task_completed=completed,
        terminated_reason=env_info.get("terminated_reason"),
    )
    env.close()
    print(f"    Episode saved: {filepath} ({env.terminated_reason or 'stopped'})")
    return filepath


def run_paired_suite(seeds: List[int], agents: List[str] = None):
    # Headless human episodes have no meaningful keyboard input; evaluate those interactively.
    agents = agents or ["llm", "rl"]
    print(f"Starting 2D agent comparison for {len(seeds)} shared seeds: {', '.join(agents)}")
    for seed in seeds:
        print(f"\n=== Scenario seed {seed} ===")
        for agent in agents:
            run_paired_experiment_on_seed(seed=seed, agent_type=agent, render=False, save_video=False)
    print("\nComparison episodes complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_file", default="experiments/seeds/test_seeds.json")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    with open(args.seeds_file, "r", encoding="utf-8") as stream:
        run_paired_suite(json.load(stream)[:args.count])
