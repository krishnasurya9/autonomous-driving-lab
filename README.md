# 🏎️ Autonomous Driving Lab (Human vs. Small-LLM vs. RL in MetaDrive)

An interactive, live 3D driving simulator and controlled research benchmark for comparing **Human Drivers**, **Small Local LLM Planners**, and **Reinforcement Learning (SAC)** continuous-control agents in procedurally generated MetaDrive scenarios.

---

## 🚀 Launch the Driving Lab Interface

The primary user experience is a live 3D driving game launcher. Run:

```bash
A:\envs\main_env\Scripts\python.exe app.py
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
A:\envs\main_env\Scripts\python.exe test_feasibility.py

# 2. Single-Agent Live 3D Launch
A:\envs\main_env\Scripts\python.exe run.py --agent human --seed 2037
A:\envs\main_env\Scripts\python.exe run.py --agent llm --seed 2037
A:\envs\main_env\Scripts\python.exe run.py --agent rl --seed 2037

# 3. Train RL (SAC) policy on training seeds (1000..1099)
A:\envs\main_env\Scripts\python.exe train.py --algorithm sac --timesteps 10000

# 4. Run 20-seed matched evaluation benchmark
A:\envs\main_env\Scripts\python.exe evaluate.py --seeds test --count 20

# 5. Generate statistical summary tables and visualization plots
A:\envs\main_env\Scripts\python.exe analyze.py
```
