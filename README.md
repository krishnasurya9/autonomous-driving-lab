# Autonomous Driving Lab — 2D

A small top-down driving simulator for comparing a human driver, a local LLM planner, and a SAC policy. The simulator is now a deterministic, seeded 2D Gymnasium environment; it does not import or require MetaDrive, Panda3D, or pygame.

## Current setup

- Project folder: `A:\MSC\sem 3\Rl\project`
- Requested Python environment: `A:\envs\metadrive_env`
- Environment interpreter: `A:\envs\metadrive_env\Scripts\python.exe` (Python 3.13.2)
- The project's dependencies have been installed into the supplied environment; `pip check` reports no broken requirements. `setup_environment.ps1` reproduces that setup later.

From PowerShell, in the project folder:

```powershell
.\setup_environment.ps1
$env:PYTHON = 'A:\envs\metadrive_env\Scripts\python.exe'
& $env:PYTHON app.py
```

The launcher honors `$env:PYTHON`; otherwise it uses the requested environment path when present, then the Python interpreter running the launcher.

## Driving

```powershell
& $env:PYTHON run.py --agent human --seed 2037
& $env:PYTHON run.py --agent llm --seed 2037
& $env:PYTHON run.py --agent rl --seed 2037
& $env:PYTHON test_feasibility.py
```

Human controls: **W/↑** accelerate, **S/↓/Space** brake, **A/←** and **D/→** steer, **Esc** closes the episode. The orange marker is a seeded obstacle. RL starts with a controller fallback until a 2D SAC checkpoint exists.

## Train and compare

```powershell
& $env:PYTHON train.py --algorithm sac --timesteps 50000
& $env:PYTHON evaluate.py --seeds test --count 5 --agents llm,rl
& $env:PYTHON analyze.py
```

Training writes `rl/checkpoints/sac_2d.zip`. Human episodes are interactive and should be run separately; headless comparisons default to LLM and RL. The LLM requires Ollama at `http://localhost:11434` with the configured model available. If no model responds, the agent uses its rule-based fallback; do not label those runs as LLM inference.

The 2D environment preserves the existing 7-value observation and 2-value action shape. It models speed, steering, lane position, road curvature, one seeded obstacle, off-road termination, and route completion. It is a lightweight prototype for agent/controller work, not a replacement for MetaDrive's traffic or 3D physics. Old MetaDrive checkpoints and telemetry are not valid 2D experiment results. New episodes are stored separately under `telemetry_2d/`.

## Main files

- `environment/driving_2d_env.py` — 2D road, vehicle dynamics, keyboard window, reward/termination signals
- `environment/metadrive_env.py` — compatibility import for scripts using the old class name
- `agents/` and `controller/` — human, LLM, SAC, and low-level control
- `experiments/run_comparison.py` — run episodes and write telemetry
- `rl/train_sac.py` — SAC training and evaluation callbacks
- `evaluation/` and `visualization/` — metrics and plots
- `setup_environment.ps1` / `requirements.txt` — setup for the supplied Python environment

## Known next steps

1. Run a short human episode and short RL training run, then inspect the new 2D telemetry and policy behavior.
2. Start Ollama and verify one real model decision before treating any LLM episode as a model result.
3. Add 2D video export if recordings are needed; the existing `--video` option currently reports that export is not configured.
4. Collect the full seed-matched comparison only after the interactive, RL, and real LLM paths have been checked. Historical CSVs and MetaDrive logs must not be mixed with 2D results.
