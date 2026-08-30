import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "int"):
    np.int = np.int_
if not hasattr(np, "bool"):
    np.bool = np.bool_

import sys
import json
import argparse
from typing import List

sys.path.insert(0, os.path.abspath("."))

from environment.metadrive_env import CustomMetaDriveEnv
from environment.state_adapter import StateAdapter
from environment.hud import HUDManager
from telemetry.logger import TelemetryLogger
from telemetry.video_recorder import VideoRecorder
from agents.human_agent import HumanAgent
# LLMAgent and RLAgent are imported lazily below to avoid crashing when
# their heavy dependencies (torch, transformers) are not needed.

def run_paired_experiment_on_seed(seed: int, agent_type: str, max_steps: int = 1000, render: bool = True, save_video: bool = False):
    print(f"--- Running Seed {seed} for Agent: {agent_type.upper()} (render={render}, save_video={save_video}) ---")

    is_human = (agent_type.lower() == "human")
    agent_type_lower = agent_type.lower()

    # Load RL/torch before any MetaDrive gltf imports to avoid Windows DLL conflicts.
    if is_human:
        agent = HumanAgent()
    elif agent_type_lower == "llm":
        from agents.llm_agent import LLMAgent
        agent = LLMAgent(backend_type="ollama", model_name="qwen2.5:7b")
    elif agent_type_lower == "rl":
        from agents.rl_agent import RLAgent
        agent = RLAgent(checkpoint_path="rl/checkpoints/sac_metadrive.zip")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    env_config = {
        "seed": seed,
        "render": render,
        "manual_control": is_human and render,
        "traffic_density": 0.1,
        "target_speed_kmh": 40.0,
    }
    env = CustomMetaDriveEnv(env_config)
    obs_vec, _ = env.reset(seed=seed)

    agent.reset()
    state_adapter = StateAdapter(target_speed_kmh=40.0)
    logger = TelemetryLogger(
        agent_type=agent_type.upper(),
        scenario_id=f"SCENARIO_{seed}",
        seed=seed,
        output_dir="telemetry",
    )

    hud = HUDManager(agent_type=agent_type, seed=seed, target_speed_kmh=40.0) if render else None

    video_recorder = None
    if save_video:
        video_recorder = VideoRecorder(agent_type=agent_type, seed=seed, output_dir="recordings")

    terminated = False
    truncated = False
    step = 0

    while step < max_steps and not (terminated or truncated):
        action, info = agent.get_action(env, obs_vec)
        obs_vec, reward, terminated, truncated, env_info = env.step(action)

        state = state_adapter.extract_state(env)
        vehicle_pos = env._env.vehicle.position if (hasattr(env, "_env") and hasattr(env._env, "vehicle")) else (0.0, 0.0)

        crashed = env_info.get("crash", False)
        out_of_road = env_info.get("out_of_road", False)
        if crashed:
            logger.record_collision()
        if out_of_road:
            logger.record_off_road()

        logger.log_step(
            step=step,
            pos_x=float(vehicle_pos[0]),
            pos_y=float(vehicle_pos[1]),
            speed_kmh=state.speed_kmh,
            steering=float(action[0]),
            throttle=float(action[1] if action[1] > 0 else 0.0),
            brake=float(-action[1] if action[1] < 0 else 0.0),
            lane_deviation_m=state.lane_deviation_m,
            heading_error_deg=state.heading_error_deg,
            llm_decision=info.get("decision"),
            llm_latency_sec=info.get("llm_latency_sec"),
            llm_invalid_output=info.get("llm_invalid_output", False),
        )

        if hud is not None:
            maneuver_text = None
            if "decision" in info and isinstance(info["decision"], dict):
                maneuver_text = info["decision"].get("maneuver")
            hud.update(
                speed_kmh=state.speed_kmh,
                lane_deviation_m=state.lane_deviation_m,
                heading_error_deg=state.heading_error_deg,
                route_progress=state.route_progress,
                collisions=logger.collisions,
                maneuver=maneuver_text,
            )

        if video_recorder is not None:
            try:
                frame = env._env.render(mode="topdown")
                if frame is not None:
                    video_recorder.add_frame(frame)
            except Exception:
                pass

        step += 1

    if hud is not None:
        hud.cleanup()

    task_completed = env_info.get("arrive_dest", False) or (not terminated and not truncated)
    filepath = logger.save(task_completed=task_completed)
    if video_recorder is not None:
        video_recorder.close()

    env.close()
    print(f"    Completed Seed {seed} for {agent_type.upper()}. Telemetry: {filepath}")

def run_paired_suite(seeds: List[int], agents: List[str] = ["human", "llm", "rl"]):
    print("=" * 60)
    print(f"STARTING PAIRED EXPERIMENT SUITE OVER {len(seeds)} SEEDS")
    print("=" * 60)

    for seed in seeds:
        print(f"\n==================== SCENARIO SEED {seed} ====================")
        for agent in agents:
            run_paired_experiment_on_seed(seed=seed, agent_type=agent, render=False, save_video=False)

    print("\n" + "=" * 60)
    print("PAIRED EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_file", type=str, default="experiments/seeds/test_seeds.json")
    args = parser.parse_args()

    if os.path.exists(args.seeds_file):
        with open(args.seeds_file, "r") as f:
            test_seeds = json.load(f)
    else:
        test_seeds = [2000, 2001, 2002]

    run_paired_suite(seeds=test_seeds[:3])
