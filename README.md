# Autonomous Driving Lab (Human vs Small-LLM vs SAC in MetaDrive)

Interactive MetaDrive lab and **intended** research benchmark comparing:

1. **Human** keyboard driving  
2. **Small local LLM** high-level planner + low-level controller  
3. **SAC** continuous-control RL  

This README describes the **actual current local tree** (2026-09-18), not the finished experiment.

Full A-to-Z handoff: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)  
Original design (partly unimplemented): [`docs/implementation_plan.md`](docs/implementation_plan.md)  
Upstream snapshot you started from: https://github.com/krishnasurya9/autonomous-driving-lab (`main` @ `ad60d3c6`)

---

## 1. Project Overview

MetaDrive is the shared simulator. All three agents are supposed to run on **the same procedural seeds** with shared telemetry and paired statistics (Human vs LLM vs RL).

**Research objective:** quantify lane keeping, speed tracking, safety (collision / off-road), and route progress for a small on-device LLM planner versus a human and a SAC policy.

**Status of that objective:** software scaffolding exists. A **valid 20-seed comparison dataset does not**. Existing CSVs show zero speed and zero lane deviation because telemetry is extracted from the Gym wrapper instead of the inner MetaDrive vehicle.

---

## 2. Current Architecture

```text
run.py / app.py / evaluate.py / train.py
        │
        ▼
CustomMetaDriveEnv  →  MetaDrive (inner)
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
Human  LLM planner  SAC
        │            │
        ▼            ▼
 Low-level controller   continuous [steer, throttle_brake]
        │
        ▼
 Telemetry JSON  →  metrics / paired tests / plots
```

**Architectural changes after the GitHub baseline (not a redesign):**

- Stronger configurable reward + off-road `terminated=True` on the wrapper  
- Human live loop: crash stop, off-road stop, optional Tk retry  
- SAC `EvalCallback` + `SubprocVecEnv` eval during training  
- Telemetry/metrics adapted to a **flat** JSON schema (differs from the implementation plan)

LLM remains hierarchical (~1 Hz JSON → 50 Hz controller). SAC remains end-to-end.

---

## 3. Repository Structure

| Path | What it does now |
|------|------------------|
| `app.py` | Tk launcher (batch button runs **5** seeds, not 20) |
| `run.py` | One agent, one seed, optional `--video` |
| `train.py` | SAC training entry (default **10k** steps) |
| `evaluate.py` | Seed-looped Human+LLM+RL (default **5** seeds) |
| `analyze.py` | Stats CSVs + plots |
| `test_feasibility.py` | Raw MetaDrive smoke test |
| `config.yaml` | Env / SAC / LLM / **reward_weights** |
| `environment/` | Wrapper, `StateAdapter`, HUD |
| `agents/` | `human_agent.py`, `llm_agent.py`, `rl_agent.py` |
| `llm/` | Ollama, LM Studio, **rule-based** “transformers” fallback |
| `controller/` | Maneuver JSON + PD/P controller |
| `rl/` | Reward, SAC train; local `checkpoints/` (gitignored) |
| `experiments/` | `run_comparison.py` + seed JSON |
| `evaluation/` | Metrics, t-test/Wilcoxon, compare |
| `visualization/` | Plots (**trajectory/speed still look in the wrong folders**) |
| `telemetry/` | Logger + JSON (**gitignored**, including `.py`) |
| `results/` | Generated CSVs/plots (gitignored) |
| `docs/implementation_plan.md` | Original plan |

---

## 4. What Was Already Implemented

When this repo was handed over (GitHub `main`), it already included:

- MetaDrive Gym wrapper, 7-D obs, 2-D actions, NumPy 2 / gltf shims  
- Human / LLM / RL agent classes and GUI  
- LLM backends, prompts, JSON parser, async planner  
- Low-level controller  
- SAC training script **without** a trained zip on GitHub  
- Evaluation, paired tests, visualization **scripts**  
- Seed files: train 1000–1019, val 1100–1109, test 2000–2019  
- README TODOs: richer reward, off-road stop, train SAC, checkpoints, 20-seed eval  

GitHub README then: trained checkpoint **missing**; off-road auto-stop **not implemented**; batch eval “need to test”.

---

## 5. What Was Added / Modified After That Baseline

| Item | Status |
|------|--------|
| `config.yaml` `reward_weights`; crash/off-road −10; lane `|dev|^1.5`; progress bonus; steering penalty | Added |
| `rl/reward.py` reads YAML; extra reward args | Modified |
| Off-road termination + `terminated_reason` in wrapper `step()` | Added |
| Human crash terminate, 1s pause, Game Over retry | Added |
| HUD 10 Hz limiter + flashing off-road/collision text | Added |
| SAC `EvalCallback`, `SubprocVecEnv` eval, copy best → `sac_metadrive.zip`, `rl_training.png` | Added |
| Local SAC run to **15,000** steps + checkpoints on disk | Produced locally (not on GitHub) |
| Metrics/compare tolerate flat `steps` schema and `AGENT_SCENARIO_*.json` names | Modified |
| `telemetry/logger.py` created locally | Added, **not in git** |
| README Version 2 changelog | Added, then corrected by this update |

Site-packages patches (MetaDrive `manual_controller.py`, `register.py`) are **not** in this repository.

---

## 6. Current Agent Status

| Agent | Status | Notes |
|-------|--------|--------|
| Human | 🟡 Implemented, 3D not re-verified this audit | WASD via MetaDrive controller. Many local `HUMAN_*.json` exist but kinematics are mostly zeros (logging bug). |
| LLM | ⚠️ Loop runs; **real model not verified** | Intended `qwen2.5:7b` / Ollama. This audit: `localhost:11434` timeout. Agent stays on `MAINTAIN` (default + rule fallback). `TransformersBackend` is **not** HuggingFace. |
| SAC | 🟡 Checkpoint on disk; live load fragile on Windows | `rl/checkpoints/sac_metadrive.zip` present. `SAC.load` works if torch loads **before** MetaDrive. After MetaDrive: `WinError 1114` on `c10.dll`. |

---

## 7. Environment and Controller

**MetaDrive** wrapped by `CustomMetaDriveEnv`.

- **Obs:** speed, target speed, lane dev, heading error, obstacle distance, route progress, curvature code (normalized).  
- **Action:** `[steering, throttle_brake] ∈ [-1, 1]`.  
- **Controller:** used by LLM (and RL fallback): P speed, P/D steering from lane/heading and maneuver.  
- **Reward:** `w_progress R_progress + w_lane R_lane + w_speed R_speed + w_safety R_safety + w_task R_task` with YAML weights.  
- **Termination:** MetaDrive crash/arrive/horizon **plus** wrapper forces terminate on `out_of_road`. Live loop also stops on crash.  
- **Off-road / collision logging:** flags exist in the loop; **JSON save omits them**.

Verified this audit: wrapper reset/step, 7-D obs. Off-road terminate **not** triggered in 80 random steps.

---

## 8. Training

| Item | Current |
|------|---------|
| Algorithm | SAC `MlpPolicy` (Stable-Baselines3) |
| Config | lr 3e-4, buffer 50k, batch 256, gamma 0.99 |
| `config.yaml` `total_timesteps` | 50000 (**not wired** into `train.py`) |
| What was actually trained locally | **15,000** steps (checkpoints at 5k/10k/15k) |
| Train maps | Starts at seed **1000** only (file lists 20 seeds; loop unused) |
| Checkpoints | `rl/checkpoints/sac_metadrive.zip`, `best_model.zip`, `sac_metadrive_checkpoint_*_steps.zip` (gitignored) |
| Eval rewards (`evaluations.npz` means) | 355.9 @ 5k, 1488.9 @ 10k, 1293.0 @ 15k |

These rewards are **not** the paper metrics (RMSE/MAE). Policy competence in 3D is **unverified** this audit.

```bash
python train.py --algorithm sac --timesteps 15000
```

---

## 9. Evaluation

```bash
python evaluate.py --seeds test --count 20   # planned
python evaluate.py --seeds test --count 5    # what CLI/GUI actually default to
python analyze.py
```

**Paired design:** for each seed, run human then llm then rl. Implemented in `run_paired_suite`. Human in that loop is **not** a recorded replay; it is a live (or unattended) `HumanAgent`.

**Existing local telemetry:** seeds 2000–2002 plus several 2037 runs. `summary_metrics.csv` showed RMSE lane **0**, speed MAE **40**, avg speed **0**. **Do not cite as results.**

---

## 10. Statistical Analysis

**Implemented:** paired t-test, Wilcoxon signed-rank, Cohen’s d; comparisons RL vs LLM, RL vs Human, LLM vs Human; metrics RMSE lane, MAE speed, steering smoothness, completion time; α = 0.05; CSVs under `results/`.

**Not implemented:** assumption checks, CIs, tests on collisions/off-road/success rate, n=20, automatic exclusion of invalid episodes.

**Current output:** n≈3–4; many t statistics NaN because series are identical zeros. **Not a valid analysis.**

`visualization/rl_training.py` still draws a **synthetic** random curve. Prefer `results/plots/rl_training.png` from `train_sac.py` if present.

---

## 11. How to Run

Use **Windows Python 3.12** (`C:\Program Files\Python312\python.exe`). The repo `.venv` does **not** contain project packages.

```bat
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m pip install -r requirements.txt
python -m pip install metadrive-simulator==0.2.6.0 gymnasium==0.26.2 panda3d-simplepbr==0.9.0

python test_feasibility.py

python app.py
python run.py --agent human --seed 2037
python run.py --agent llm --seed 2037
python run.py --agent rl --seed 2037
python run.py --agent rl --seed 2037 --video

python train.py --algorithm sac --timesteps 15000
python evaluate.py --seeds test --count 5
python analyze.py
```

LLM (optional, **not running** at last audit):

```bat
ollama run qwen2.5:7b
```

Human controls (when MetaDrive manual control is on): WASD / arrows, Space brake, ESC exit. HUD mentions `C` for camera; **this repo has no camera handler**.

---

## 12. Current Status

Legend: ✅ tested this audit · 🟡 implemented, not fully tested · 🟠 partial · ❌ missing · ⚠️ has an issue · 📌 planned

| Component | Status | Verified | Notes |
|-----------|--------|----------|-------|
| MetaDrive env | 🟡 | Feasibility PASS; wrapper step PASS | Needs protobuf env var |
| Human agent | 🟡 | Prior telemetry; not live this audit | Logging zeros |
| LLM agent | ⚠️ | Fallback MAINTAIN only | Ollama down |
| SAC agent | ⚠️ | Zip load PASS if torch-first | DLL fail after MetaDrive |
| Controller | ✅ | Unit PASS | |
| Observation | 🟡 | 7-D obs PASS in wrapper | Telemetry path wrong |
| Actions | ✅ | Controller + SAC dummy predict | |
| Reward | ✅ | Unit PASS, YAML loaded | Policy effect untested |
| Off-road detect/terminate | 🟠 | Code present | Not hit in 80 random steps |
| Collision handling | 🟠 | Live loop sets terminate | Not persisted in JSON |
| Route progress | 🟠 | In obs/reward | Not in saved summaries |
| Episode termination | 🟡 | Feasibility ended step 43 | |
| Seeds / pairing | 🟠 | Files + loop exist | n=20 not done; train uses 1 seed |
| Reproducibility | 🟠 | Reset called | Weak check (speed/lane 0 at reset print) |
| Logging | ⚠️ | Files write | Zeros + gitignored logger |
| Checkpoints | 🟡 | Files on disk | Gitignored; not on GitHub |
| RL training | 🟡 | Artifacts 15k | Not 50k |
| Evaluation pipeline | ⚠️ | Script runs | Invalid numbers |
| Metrics | 🟠 | Formulas exist | Inputs broken |
| Visualization | 🟠 | Bar plots from CSV | Trajectories/speed wrong glob; RL curve script synthetic |
| LLM planning | ⚠️ | | Not actually queried |
| LLM → controller | 🟡 | | Works with default/fallback JSON |
| Statistics | ⚠️ | Code runs | Unusable sample |
| Human vs LLM vs RL | ❌ | | No valid paired dataset |
| Experiment management | 🟠 | | GUI count 5; no scenario_manager |

---

## 13. Known Issues

- Telemetry `extract_state(CustomMetaDriveEnv)` never sees `.vehicle` → **zero speed/lane** in almost all agent logs.  
- `TelemetryLogger.save()` drops collisions / off-road / `terminated_reason`.  
- `.gitignore` ignores **all** of `telemetry/`, so `logger.py` is not on GitHub.  
- Ollama not reachable; LLM results are not LLM results.  
- Torch DLL vs MetaDrive on Windows.  
- Protobuf descriptor error without `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.  
- `app.py` “20-seed” → `--count 5`. Preview = human drive.  
- `requirements.txt` unpinned (README previously mentioned `panda3d-simplepbr==0.9.0`; file does not pin it).  
- Local continuation commits **not pushed**. GitHub has no `ver2`.  
- `git` may be missing from PATH. `.git/config` origin URL must not be shared (embedded credential).  
- Camera toggle / “HD graphics” / “fully working batch analysis” were overstated in an earlier README revision.

---

## 14. Remaining Work

1. Fix state extraction for telemetry (`env._env`).  
2. Fix logger schema; gitignore JSON only.  
3. Fix visualization globs; stop publishing the synthetic RL curve.  
4. Import/load SAC before MetaDrive.  
5. Run a real Ollama episode and store raw JSON.  
6. Collect 20-seed LLM vs RL (human separately), then analyze.  
7. Optionally train more than 15k and/or more map seeds.  
8. Pin dependencies; document protobuf.  
9. Push `ver2` **without** embedding a PAT in the remote URL.

---

## 15. Reproducibility

| Item | Value |
|------|--------|
| Python | 3.12.3 (`C:\Program Files\Python312\python.exe`) |
| numpy | 2.4.4 (this machine) |
| torch | 2.11.0+cpu (this machine) |
| Env var | `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` |
| Target speed | 40 km/h |
| Test seeds | 2000–2019 |
| Train seeds file | 1000–1019 (training uses 1000) |
| SAC checkpoint | `rl/checkpoints/sac_metadrive.zip` (~3.07 MB), 15k steps |
| GitHub remote `main` | `ad60d3c6` — **does not include** local training or this README |
| Local branch | `ver2` @ `c122de80` plus uncommitted doc updates |

---

## CLI cheat sheet (unchanged entry points)

```bash
python test_feasibility.py
python run.py --agent human --seed 2037
python run.py --agent llm --seed 2037
python run.py --agent rl --seed 2037
python train.py --algorithm sac --timesteps 10000
python evaluate.py --seeds test --count 20
python analyze.py
```

Treat evaluate `--count 20` as **aspirational** until logging is fixed.
