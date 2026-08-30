# 2D Autonomous Driving Lab — Implementation Plan

## Goal

Provide a lightweight top-down driving task for the existing Human, local LLM, and SAC agents. Keep one common environment API, state representation, reward function, and telemetry format so the agents can be iterated on without MetaDrive or Panda3D.

## Current implementation

- `environment/driving_2d_env.py` implements a deterministic seeded road, acceleration/braking, steering, lane departure, one obstacle, route completion, a seven-value observation, and continuous `[steering, throttle]` control.
- `environment/metadrive_env.py` preserves the former import name as a compatibility alias; it no longer loads MetaDrive.
- The Tk window renders the track and reads keyboard controls. Human, LLM, and SAC agent classes remain in use.
- SAC training saves the new policy as `rl/checkpoints/sac_2d.zip`. Train a fresh 2D model; an old MetaDrive model is not a validated replacement.
- New episode logs go to `telemetry_2d/`, separate from historical MetaDrive data. Metrics and trajectory/speed plots read this directory.
- `setup_environment.ps1` installs `requirements.txt` into `A:\envs\metadrive_env` when the user runs it.

## Verification completed (2026-10-08)

- Installed project dependencies in `A:\envs\metadrive_env`; `pip check` reports no broken requirements. Added `tensorboard` to `requirements.txt` after SAC training exposed the missing package.
- The environment smoke check, SB3 environment checker, Python compilation, agent/statistics imports, and analysis pipeline completed.
- Added deterministic cycling through the configured training seeds and evaluation on the validation split; verified the seed sequence before training.
- Trained SAC for 5,000 steps on CPU. The evaluator produced one five-episode validation point (mean reward 886.61, standard deviation 472.52); this is an initial pilot with high variation, not evidence of a competent or general policy.
- Fixed headless LLM timing so an async decision can complete at the 1 Hz planner boundary; telemetry records the selected fallback and request latency once per decision.
- Verified an 80-step equal-horizon RL/LLM smoke pair on seed 2003. Both reached the time limit without collision. The LLM record identifies `rule_fallback` and a 4.11-second failed-request/fallback latency; Ollama was unavailable, so this is not real model inference.
- The analysis pipeline matched one seed for RL vs LLM and correctly returned “Insufficient paired samples.” No human run was collected.
- Ollama was not listening on `localhost:11434`, so no real model inference was verified.

## Next steps

1. Extend SAC training and evaluate across more checkpoints; inspect the wide validation variation before using the checkpoint for comparisons.
2. Start Ollama with the configured model, verify one real LLM decision, and confirm backend, latency, and parser status in telemetry.
3. Launch and document an interactive human run; then collect all agents on matching held-out test seeds.
4. Check log completeness and equal episode conditions before running the paired statistical analysis.
5. Expand scenario variety and difficulty only after the basic protocol is reproducible; add video export only if the study needs it.
6. Present the 2D environment as a prototype. Do not combine its results with historical 3D MetaDrive results or claim statistical differences before adequate matched data exists.

## Boundaries

The current road is procedurally generated from a deterministic formula. The vehicle dynamics are intentionally simplified and do not model traffic, tire forces, or realistic collision physics. This implementation supports agent and control development; it does not preserve the experimental meaning of previous MetaDrive results.
