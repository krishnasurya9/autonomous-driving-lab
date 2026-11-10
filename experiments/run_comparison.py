"""Run individual or seed-matched episodes in the configured driving simulator."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import yaml

from agents.human_agent import HumanAgent
from environment.driving_2d_env import Driving2DEnv
from environment.state_adapter import StateAdapter
from telemetry.logger import TelemetryLogger


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or os.path.join(PROJECT_DIR, "config.yaml")
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _make_agent(agent_type: str, config: Dict[str, Any], sim_step_dt: float, observation_dim: int):
    normalized = agent_type.lower()
    target_speed = float(config.get("environment", {}).get("target_speed_kmh", 40.0))
    if normalized == "human":
        return HumanAgent()
    if normalized == "llm":
        from agents.llm_agent import LLMAgent

        llm_config = config.get("llm", {})
        return LLMAgent(
            backend_type=llm_config.get("backend", "ollama"),
            model_name=llm_config.get("model", "qwen2.5:7b"),
            decision_frequency_hz=float(llm_config.get("decision_frequency_hz", 1.0)),
            sim_step_dt=sim_step_dt,
            target_speed_kmh=target_speed,
            endpoint=llm_config.get("ollama_url") if llm_config.get("backend", "ollama").lower() == "ollama" else llm_config.get("lmstudio_url"),
            temperature=float(llm_config.get("temperature", 0.1)),
        )
    if normalized == "rl":
        from agents.rl_agent import RLAgent

        checkpoint = config.get("rl", {}).get("checkpoint", "rl/checkpoints/sac_2d.zip")
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(PROJECT_DIR, checkpoint)
        return RLAgent(
            checkpoint_path=checkpoint,
            target_speed_kmh=target_speed,
            expected_observation_dim=observation_dim,
        )
    raise ValueError(f"Unknown agent type {agent_type!r}; expected human, llm, or rl.")


def run_paired_experiment_on_seed(
    seed: int,
    agent_type: str,
    max_steps: Optional[int] = None,
    render: bool = True,
    save_video: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Run one agent on one deterministic 2D scenario and save its telemetry."""
    cfg = config or load_config()
    env_cfg = dict(cfg.get("environment", {}))
    if env_cfg.get("backend", "2d").lower() != "2d":
        raise ValueError("This runner currently supports environment.backend: 2d only.")
    env_cfg.update(seed=int(seed), render=bool(render))
    if max_steps is not None:
        env_cfg["horizon"] = int(max_steps)
    if agent_type.lower() == "human" and not render:
        raise ValueError("Human runs require rendering so keyboard controls are available.")
    if save_video:
        raise NotImplementedError("Video export is not configured for the 2D simulator.")

    env = Driving2DEnv(env_cfg)
    agent = _make_agent(agent_type, cfg, env.dt, env.observation_space.shape[0])
    adapter = StateAdapter(target_speed_kmh=env.target_speed_kmh)
    obs, _reset_info = env.reset(seed=int(seed))
    agent.reset()
    logger = TelemetryLogger(
        agent_type=agent_type.upper(),
        scenario_id=f"SCENARIO_{seed}",
        seed=int(seed),
        target_speed_kmh=env.target_speed_kmh,
        step_duration_sec=env.dt,
        environment_backend=env_cfg.get("backend", "2d"),
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        policy_checkpoint=getattr(agent, "checkpoint_path", None),
        output_dir=os.path.join(PROJECT_DIR, "telemetry_2d"),
    )

    terminated = False
    truncated = False
    info: Dict[str, Any] = {}
    step = 0
    try:
        limit = int(env.horizon if max_steps is None else max_steps)
        while step < limit and not (terminated or truncated):
            if agent_type.lower() == "human" and env._window_closed:
                info = {**info, "terminated_reason": "user_exit"}
                break
            frame_started = time.perf_counter()
            action, agent_info = agent.get_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            state = adapter.extract_state(env)
            if info.get("crash"):
                logger.record_collision()
            if info.get("out_of_road"):
                logger.record_off_road()
            logger.log_step(
                step=step,
                pos_x=float(env.progress_m),
                pos_y=float(state.lane_deviation_m),
                speed_kmh=float(state.speed_kmh),
                steering=float(action[0]),
                throttle=max(float(action[1]), 0.0),
                brake=max(-float(action[1]), 0.0),
                lane_deviation_m=float(state.lane_deviation_m),
                heading_error_deg=float(state.heading_error_deg),
                route_progress=float(state.route_progress),
                obstacle_distance=float(state.obstacle_distance),
                obstacle_lateral_offset_m=float(state.obstacle_lateral_offset_m),
                reward=float(reward),
                agent_policy=agent_info.get("policy"),
                llm_decision=agent_info.get("decision"),
                llm_backend=agent_info.get("llm_backend"),
                llm_latency_sec=agent_info.get("llm_latency_sec"),
                llm_invalid_output=bool(agent_info.get("llm_invalid_output", False)),
            )
            if render:
                env.render()
                remaining = env.dt - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
            step += 1

        if hasattr(agent, "wait_for_pending_decision"):
            agent.wait_for_pending_decision()
        filepath = logger.save(
            task_completed=bool(info.get("arrive_dest", False)),
            terminated_reason=info.get("terminated_reason"),
            final_route_progress=float(info.get("route_progress", env.progress_m / env.route_length_m)),
        )
        print(
            f"{agent_type.upper()} seed {seed}: {step} steps, "
            f"{100.0 * env.progress_m / env.route_length_m:.1f}% route, "
            f"{info.get('terminated_reason', 'incomplete')}; telemetry: {filepath}"
        )
        if agent_type.lower() == "human" and render and (terminated or truncated):
            env.show_episode_result(
                reason=info.get("terminated_reason"),
                route_progress=float(info.get("route_progress", env.progress_m / env.route_length_m)),
            )
        return filepath
    finally:
        env.close()


def run_paired_suite(seeds: List[int], agents: Optional[List[str]] = None) -> List[str]:
    """Run selected agents sequentially on each supplied seed."""
    agent_list = agents or ["llm", "rl"]
    outputs = []
    for seed in seeds:
        print(f"\n=== Scenario seed {seed} ===")
        for agent in agent_list:
            outputs.append(run_paired_experiment_on_seed(seed=seed, agent_type=agent, render=False))
    print(f"\nCompleted {len(outputs)} episodes across {len(seeds)} seeds.")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a seed-matched 2D driving experiment.")
    parser.add_argument("--seeds-file", default="experiments/seeds/test_seeds.json")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--agents", default="llm,rl")
    args = parser.parse_args()
    seed_file = args.seeds_file if os.path.isabs(args.seeds_file) else os.path.join(PROJECT_DIR, args.seeds_file)
    with open(seed_file, "r", encoding="utf-8") as stream:
        seeds = json.load(stream)
    if args.count is not None:
        seeds = seeds[: args.count]
    run_paired_suite(seeds, [item.strip().lower() for item in args.agents.split(",") if item.strip()])
