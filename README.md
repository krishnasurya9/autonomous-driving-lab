# 🏎️ Autonomous Driving Lab (Human vs. Small-LLM vs. RL in MetaDrive)

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
| GUI launcher (`app.py`) | ✅ Working |
| Human driving (WASD / arrows) | ✅ Working |
| LLM agent (async planner, live 3D) | ✅ Working |
| RL agent live view | ✅ Working (uses fallback policy until trained) |
| Crash / lag fixes (Windows) | ✅ Fixed |
| Telemetry logging | ✅ Working |
| Batch evaluation & analysis | need to test  |
| Trained SAC checkpoint | ❌ Not trained yet (`rl/checkpoints/sac_metadrive.zip` missing) |
| Strong off-road penalties in RL | ⚠️ Basic only (see below) |
| Auto-stop when car leaves road | ❌ Not implemented yet |

---

## 🔧 TODO / Next Steps

These are the main tasks still needed to complete the project:

### 1. Improve RL Rewards & Penalties (`rl/reward.py`)

The reward function exists but needs tuning so the agent learns to stay on the road. Planned changes:

- **Stronger off-road penalty** — increase weight when `out_of_road=True` (currently `-2.0` via `w_safety`)
- **Progressive lane deviation penalty** — larger penalty as `lane_deviation_m` grows (not just linear)
- **Collision penalty** — bigger negative reward on crash to discourage unsafe driving
- **Route progress reward** — reward moving toward destination, not just speed
- **Smooth driving bonus** — small reward for low steering jitter and stable speed
- **Configurable weights** — expose reward weights in `config.yaml` for easy tuning

**File to edit:** `rl/reward.py`

### 2. Auto-Stop Episode When Car Goes Off Road

Right now the episode continues even after the car leaves the road. Need to add:

- **Terminate episode** when `out_of_road=True` in `environment/metadrive_env.py` (`step()`)
- **Optional early stop** in `experiments/run_comparison.py` for live 3D modes (Human / RL / LLM)
- **HUD message** showing "OFF ROAD — EPISODE ENDED" before closing
- **Telemetry flag** in logs: `terminated_reason: "off_road"`

**Files to edit:** `environment/metadrive_env.py`, `experiments/run_comparison.py`, `environment/hud.py`

### 3. Train the RL (SAC) Policy

The RL agent currently uses a simple fallback controller, which is why the car often drives off-road.

```bash
# Train SAC policy (saves checkpoint automatically)
python train.py --algorithm sac --timesteps 50000
```

After training, checkpoint should be saved to:
`rl/checkpoints/sac_metadrive.zip`

Then re-run:
```bash
python run.py --agent rl --seed 2037
```

**File to edit:** `rl/train_sac.py` (ensure auto-save of best checkpoint during training)

### 4. Auto-Save RL Checkpoints During Training

- Save checkpoint every N training steps
- Keep **best** checkpoint based on evaluation reward (lane keeping + route completion)
- Save training curves to `results/plots/rl_training.png`

**File to edit:** `rl/train_sac.py`

### 5. Evaluation & Comparison After Training

Once RL is trained, run the full benchmark:

```bash
python evaluate.py --seeds test --count 20
python analyze.py
```

Compare Human vs LLM vs RL on:
- Lane deviation (RMSE)
- Speed error (MAE)
- Collision count
- Route completion rate
- Off-road events

### 6. Optional Improvements

- [ ] Add `requirements.txt` install instructions for MetaDrive on Windows
- [ ] Add Ollama setup note for LLM mode (`ollama pull qwen2.5:7b`)
- [ ] Add off-road / collision sound or visual warning in HUD
- [ ] Record demo videos with `python run.py --agent rl --seed 2037 --video`

---

## 📁 Key Files Reference

| File | Purpose |
|------|---------|
| `rl/reward.py` | RL reward & penalty function |
| `rl/train_sac.py` | SAC training loop & checkpoint saving |
| `environment/metadrive_env.py` | Simulator wrapper, episode termination logic |
| `experiments/run_comparison.py` | Live 3D game loop for all agents |
| `agents/rl_agent.py` | Loads trained SAC policy or fallback controller |
| `config.yaml` | Hyperparameters, reward weights, seeds |
