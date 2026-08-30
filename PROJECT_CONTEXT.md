# Project Handoff — 2D Migration (2026-10-08)

## User direction

Move the autonomous-driving project from a heavy 3D MetaDrive setup to a simpler 2D experience. Use the separate Python environment at `A:\envs\metadrive_env`, inspect what is broken there, and prepare the project for continued development.

## Current architecture

```text
app.py / run.py / evaluate.py / train.py
                  │
                  ▼
        Driving2DEnv (Gymnasium)
         ├─ Human keyboard input
         ├─ LLM planner + low-level controller
         └─ SAC policy / fallback controller
                  │
                  ▼
          telemetry_2d/*.json
                  │
          metrics and plots
```

`environment/metadrive_env.py` remains as a compatibility import for older callers, but aliases to `Driving2DEnv`; no MetaDrive/Panda3D dependency is used by the 2D runtime.

## Environment diagnosis

- Interpreter exists at `A:\envs\metadrive_env\Scripts\python.exe` and reports Python 3.13.2.
- Initially, every project dependency was missing (Gymnasium, Stable-Baselines3, Torch, NumPy, PyYAML, requests, pandas, matplotlib, SciPy, seaborn, and tqdm); `tkinter` was available.
- Installed the packages from `requirements.txt` into the requested interpreter. `pip check` now reports “No broken requirements found.”
- `setup_environment.ps1` reproduces the install in that interpreter.

## Changes made

- Added a deterministic seeded top-down road, vehicle state/dynamics, obstacle and off-road termination, arrival/horizon signals, a Tk renderer, and keyboard controls.
- Kept the original observation and action shapes so the agents and SAC interfaces remain aligned.
- Updated the human agent, 2D runner, batch runner, SAC training path, launcher labels, and run instructions.
- Changed training output to `rl/checkpoints/sac_2d.zip`; a fresh 2D model is needed. The prior MetaDrive policy is not considered validated for this environment.
- New run telemetry is isolated at `telemetry_2d/`; metrics, trajectory/speed charts, and the real SAC evaluation plot now use 2D data. Legacy `telemetry/` data stays separate.
- Fixed merge conflict markers in project code/config/docs that made several modules unparsable, and made `telemetry/` source files trackable.
- Replaced the feasibility script's direct MetaDrive launch with a 2D environment smoke check. It passed: 7-value observations, 100 steps, non-zero speed and lane state, time-limit termination.
- `compileall` passed for the project sources.

## Limits and remaining work

- The current simulator is deliberately simple: one generated road, a single obstacle, simplified vehicle dynamics, and no traffic or realistic crash physics.
- 2D video export is not implemented; passing `--video` prints a clear notice.
- Ollama availability/model presence was not checked. An unavailable server triggers the project's rule-based fallback, which must not be reported as an LLM result.
- No fresh SAC training, live GUI episode, or full agent comparison was run. Historical 3D logs and result CSVs are not valid 2D results.

## Continue here

1. Launch `app.py` or `run.py --agent human --seed 2037` and confirm keyboard/window behavior.
2. Train a fresh SAC policy and inspect its real evaluation history.
3. Verify one real Ollama decision and fallback status before collecting matched LLM/SAC seeds.
4. Collect interactive human runs separately and only compare seeds with complete data for each included agent.
