# 🏎️ Autonomous Driving Lab (Human vs. Small-LLM vs. RL in MetaDrive)

---

## 🚀 Version 2 — What's New (Branch: `ver2`)

> This branch contains all updates and optimizations made on top of the original `main` branch.
> Merge this into `main` once reviewed.

### ✅ Bug Fixes
| Fix | File |
|-----|------|
| Patched NumPy 2.x array-shape crash in manual controller | `metadrive/engine/core/manual_controller.py` |
| Fixed `panda3d-simplepbr` ImportError (downgraded to `0.9.0`) | `requirements.txt` |
| Resolved MetaDrive Gymnasium registration crash (`env_specs` KeyError) | `metadrive/register.py` |
| Fixed `TelemetryLogger` ModuleNotFoundError (created from scratch) | `telemetry/logger.py` |
| Patched telemetry schema mismatch in `compute_episode_metrics` | `evaluation/metrics.py` |
| Fixed telemetry glob pattern to match actual file names | `evaluation/compare_agents.py` |

### 🎮 Gameplay Improvements
| Feature | File |
|---------|------|
| **Game Over popup** — "Try Again?" dialog on crash or off-road | `experiments/run_comparison.py` |
| **Seamless episode restart** — engine stays open, level resets | `experiments/run_comparison.py` |
| **Crash = instant termination** — collisions now end the episode | `experiments/run_comparison.py` |
| **Flashing HUD warnings** — red `CRITICAL: OFF ROAD!` / `WARNING: COLLISION!` overlay | `environment/hud.py` |
| **10Hz HUD rate-limiter** — eliminates extreme rendering choppiness | `environment/hud.py` |

### 🧠 Reinforcement Learning Optimizations
| Feature | File |
|---------|------|
| Configurable reward weights via `config.yaml` | `rl/reward.py`, `config.yaml` |
| `-10` crash penalty + `-10` off-road penalty | `config.yaml` |
| `+2.0` route progress bonus, `-0.1` steering smoothness penalty | `config.yaml` |
| SAC training uses `SubprocVecEnv` for eval (fixes MetaDrive singleton crash) | `rl/train_sac.py` |
| SAC model trained to 15,000 timesteps, checkpoint saved | `rl/checkpoints/sac_metadrive.zip` |

### 📊 Analysis & Benchmarking
| Feature | File |
|---------|------|
| Statistical paired tests (t-test + Wilcoxon) generated | `results/paired_statistical_tests.csv` |
| Performance comparison bar charts generated | `results/plots/` |
| `analyze.py` fixed — handles all telemetry schemas | `analyze.py` |

---


An interactive, live 3D driving simulator and controlled research benchmark for comparing **Human Drivers**, **Small Local LLM Planners**, and **Reinforcement Learning (SAC)** continuous-control agents in procedurally generated MetaDrive scenarios.

---

## 🚀 Launch the Driving Lab Interface

The primary user experience is a live 3D driving game launcher. Run:

```bash
python app.py
```

This launches the GUI Lab Interface:

```text
╔══════════════════════════════════════════════════════════════════╗
║                    🏎️ AUTONOMOUS DRIVING LAB                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Scenario Seed:  [ 2037 ]                                        ║
║                                                                  ║
║  [ 👤 PLAY HUMAN DRIVER (Live 3D WASD) ]                        ║
║  [ 🤖 WATCH RL AGENT (SAC Policy Live 3D) ]                      ║
║  [ 🧠 WATCH LLM AGENT (High-Level Planner Live 3D) ]             ║
║  [ 👁️ PREVIEW SCENARIO ROAD GEOMETRY ]                          ║
║                                                                  ║
║  [ 📊 RUN 20-SEED BATCH BENCHMARK ]                              ║
║  [ 📈 ANALYZE STATISTICS & PLOTS ]                               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🎮 Driving Modes & Controls

### 👤 1. Playable Human Mode (`PLAY HUMAN`)
- **WASD / Arrow Keys**: Interactive real-time steering, acceleration, and braking.
- **Spacebar**: Emergency stop / Handbrake.
- **`C` Key**: Toggle Camera view (Third-person chase camera, top-down view).
- **`ESC` Key**: Exit driving session.

### 🤖 2. Watch RL Agent (`WATCH RL`)
- Watch the trained continuous-control **SAC policy** navigate the procedural road geometry live on screen in 3D.

### 🧠 3. Watch LLM Agent (`WATCH LLM`)
- Watch the hierarchical **LLM High-Level Planner** (~1 Hz) and deterministic **Low-Level Controller** (50 Hz) guide the vehicle live on screen in 3D.

### 📺 4. Live On-Screen HUD Overlay
Every 3D driving mode displays a real-time HUD with telemetry:
- Current Speed vs. Target Speed (km/h)
- Lane Deviation (m) & Heading Error (deg)
- High-Level Maneuver / LLM Decision
- Route Completion Progress (%)
- Collision Count

---

## 💻 CLI & Batch Benchmark Reference

```bash
# 1. Verify environment & seed reproducibility
python test_feasibility.py

# 2. Single-Agent Live 3D Launch
python run.py --agent human --seed 2037
python run.py --agent llm --seed 2037
python run.py --agent rl --seed 2037

# 3. Train RL (SAC) policy on training seeds (1000..1099)
python train.py --algorithm sac --timesteps 10000

# 4. Run 20-seed matched evaluation benchmark
python evaluate.py --seeds test --count 20

# 5. Generate statistical summary tables and visualization plots
python analyze.py
```

---

## ✅ Current Status (What Works Now)

| Feature | Status |
|---------|--------|
| GUI launcher (`app.py`) | ✅ Working (Dark Theme + Error Handling) |
| Human driving (WASD / arrows) | ✅ Working (Seamless Restart Loop) |
| LLM agent (async planner, live 3D) | ✅ Working |
| RL agent live view | ✅ Working (Optimized SAC Policy) |
| Crash / lag fixes (Windows) | ✅ Fixed (10Hz HUD Limiter & Subprocess Training) |
| Telemetry logging | ✅ Working |
| Batch evaluation & analysis | ✅ Working |
| Trained SAC checkpoint | ✅ Trained and saved in `rl/checkpoints/` |
| Strong off-road penalties in RL | ✅ Fully implemented via `config.yaml` |
| Auto-stop when car leaves road | ✅ Implemented with GUI popup & Flashing HUD Warning |
| Environment Graphical Upgrades | ✅ Configured (HD Graphics, Daytime Lighting, Complex Map) |

---

## 🔧 Optional / Additional Setup

### 1. Windows Installation Notes
To install MetaDrive successfully on Windows without DLL crashes:
```bash
pip install metadrive-simulator==0.2.6.0 gymnasium==0.26.2
pip install panda3d-simplepbr==0.9.0 # (Fixes latest shader crash)
```

### 2. LLM Setup (Ollama)
To use the LLM agent, you need Ollama running locally:
```bash
ollama run qwen2.5:7b
```

### 3. Demo Video Recording
To generate top-down video recordings of an agent, use the `--video` flag:
```bash
python run.py --agent rl --seed 2037 --video
```
Videos will be saved to the `recordings/` folder as MP4.



## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `rl/reward.py` | RL reward & penalty function |
| `rl/train_sac.py` | SAC training loop & checkpoint saving |
| `environment/metadrive_env.py` | Simulator wrapper, episode termination logic |
| `experiments/run_comparison.py` | Live 3D game loop for all agents |
| `agents/rl_agent.py` | Loads trained SAC policy or fallback controller |
| `config.yaml` | Hyperparameters, reward weights, seeds |
