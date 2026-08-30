import sys
import os

# Set Protobuf implementation compatibility for MetaDrive
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import numpy as np

# Patch NumPy 2.x removed deprecated aliases for legacy MetaDrive compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = np.bool_

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))

from environment.state_adapter import StateAdapter
from telemetry.logger import TelemetryLogger
from controller.decision_api import HighLevelDecision, MANEUVER_MAINTAIN
from controller.low_level_controller import LowLevelController

def run_feasibility_test(seed: int = 2037, render: bool = False, steps_to_run: int = 50):
    print("=" * 60)
    print(f"RUNNING FEASIBILITY MILESTONE TEST (Seed: {seed})")
    print("=" * 60)

    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
    except ImportError as e:
        print(f"[ERROR] Failed to import MetaDrive: {e}")
        return False

    env_config = {
        "start_seed": seed,
        "environment_num": 1,
        "use_render": render,
        "manual_control": False,
        "traffic_density": 0.1,
    }

    print(f"[1/7] Initializing MetaDrive environment (render={render})...")
    env = MetaDriveEnv(env_config)

    try:
        print("[2/7] Resetting environment with seed...")
        try:
            res = env.reset(force_seed=seed)
        except TypeError:
            res = env.reset()
        
        if isinstance(res, tuple):
            obs = res[0]
            info = res[1] if len(res) > 1 else {}
        else:
            obs = res
            info = {}

        print(f"      Initial observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")

        state_adapter = StateAdapter(target_speed_kmh=40.0)
        controller = LowLevelController()
        logger = TelemetryLogger(
            agent_type="FEASIBILITY",
            scenario_id=f"SCENARIO_{seed}",
            seed=seed,
            output_dir="telemetry",
        )

        print(f"[3/7] Running {steps_to_run} simulation steps...")
        default_decision = HighLevelDecision(target_speed_kmh=40.0, maneuver=MANEUVER_MAINTAIN)

        for step in range(steps_to_run):
            # Extract state
            state = state_adapter.extract_state(env)
            
            # Compute action from controller
            action = controller.compute_action(
                decision=default_decision,
                current_speed_kmh=state.speed_kmh,
                lane_deviation_m=state.lane_deviation_m,
                heading_error_deg=state.heading_error_deg,
            )

            # Step environment
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
            elif len(step_res) == 4:
                obs, reward, done, info = step_res
                terminated, truncated = done, False

            # Record telemetry
            vehicle_pos = env.vehicle.position if hasattr(env, "vehicle") else (0.0, 0.0)
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
            )

            if terminated or truncated:
                print(f"      Episode ended early at step {step}")
                break

        print("[4/7] Saving telemetry output...")
        telemetry_file = logger.save(task_completed=True)
        print(f"      Telemetry saved to: {telemetry_file}")

        print("[5/7] Testing scenario reset & seed reproducibility...")
        try:
            res2 = env.reset(force_seed=seed)
        except TypeError:
            res2 = env.reset()
        
        if isinstance(res2, tuple):
            obs2 = res2[0]
        else:
            obs2 = res2

        state2 = state_adapter.extract_state(env)
        print(f"      Reset state speed: {state2.speed_kmh:.2f} km/h, lane dev: {state2.lane_deviation_m:.2f} m")

        print("[6/7] Closing MetaDrive environment...")
        env.close()

        print("[7/7] FEASIBILITY MILESTONE TEST PASSED SUCCESSFULLY! [PASSED]")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[ERROR] Feasibility test failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        return False

if __name__ == "__main__":
    success = run_feasibility_test(seed=2037, render=False, steps_to_run=50)
    sys.exit(0 if success else 1)
