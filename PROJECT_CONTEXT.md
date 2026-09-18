# PROJECT_CONTEXT.md

Handoff document for **Autonomous Driving Lab**. This describes the **actual current local codebase**, not an idealized end state.

If another developer or AI agent continues this work, start here. Do **not** rebuild the project from scratch. The GitHub repository already contained a substantial lab; later work continued from that baseline.

**Audit date:** 2026-09-18  
**Local workspace:** `C:\Users\HP\.gemini\antigravity\scratch\autonomous-driving-lab-main`  
**Upstream (what the owner handed over):** https://github.com/krishnasurya9/autonomous-driving-lab (`main` only, latest remote commit `ad60d3c6`)

---

## 1. Project Identity

- **Project name:** Autonomous Driving Lab (Human vs Small Local LLM vs SAC in MetaDrive)
- **Objective:** Compare three agent types on matched MetaDrive procedural scenarios using shared telemetry and metrics.
- **Research question:** How does a small local LLM high-level planner (plus a deterministic low-level controller) compare with a human driver and a continuous SAC policy on the same seed-matched driving tasks?
- **Overall concept:** MetaDrive is a shared experimental laboratory. Human uses keyboard control. LLM outputs JSON maneuvers at ~1 Hz. SAC outputs continuous `[steering, throttle_brake]` every step. Evaluation is intended to be paired by seed.

---

## 2. Starting Codebase

This is what existed on GitHub `main` when the repository was handed over (README TODOs + source tree). It was **not** an empty project.

Already present and relevant:

| Area | What already existed |
|------|----------------------|
| Environment | `CustomMetaDriveEnv` Gymnasium wrapper, NumPy 2.x alias patches, `panda3d-gltf.patch_loader` shim, 7-D observation, 2-D action |
| State | `DrivingState` + `StateAdapter` (lane lat, heading_diff, navigation, lidar heuristic) |
| Human | `HumanAgent` reading MetaDrive `engine.controller` |
| LLM | Ollama / LM Studio / rule-based “Transformers” backends, async planner, JSON parser, prompts |
| Controller | `HighLevelDecision` + PD/P low-level controller |
| RL | SAC training script (checkpoint every 5k steps, save `sac_metadrive.zip`), `DrivingReward` with crash/off-road **−2.0** inside `r_safety`, `w_safety=5.0` |
| RL agent | Load SAC zip **or** fallback to `LowLevelController` / `MAINTAIN` |
| Eval | `evaluate.py`, `run_comparison.py`, metrics, paired t-test + Wilcoxon + Cohen’s d |
| Viz | trajectory / performance / speed / synthetic RL curve scripts |
| GUI | `app.py` Tk launcher |
| Seeds | train `1000–1019` (20, not 100), val `1100–1109` (10, not 25), test `2000–2019` (20) |
| Docs | README with working GUI/human/LLM/RL-fallback claims; TODOs for rewards, off-road stop, training, checkpoints, 20-seed eval |
| Missing on GitHub | `telemetry/` package (gitignored), checkpoints, results; no `scenario_manager.py`, `environment_config.py`, `evaluate_rl.py`, `run_human.py` / `run_llm.py` / `run_rl.py` |

GitHub README status at handoff:

- GUI, human, LLM live 3D, RL live view (fallback until trained): claimed working
- Batch evaluation: “need to test”
- Trained SAC checkpoint: **missing**
- Strong off-road penalties: basic only
- Auto-stop when off-road: **not implemented**

---

## 3. Tasks Given To You

Taken from GitHub `README.md` section **TODO / Next Steps** (handoff tasks), plus the original `docs/implementation_plan.md` research design.

### Tasks Given After Initial Codebase

1. **Task:** Improve RL rewards and penalties (`rl/reward.py`).  
   **Why:** Agent was not learning to stay on the road; crash/off-road were only −2.0.  
   **Expected:** Stronger off-road/collision penalties, progressive lane penalty, route-progress reward, smoothness term, weights in `config.yaml`.  
   **README:** TODO item 1.

2. **Task:** Auto-stop episode when the car goes off-road.  
   **Why:** Episodes continued after leaving the road.  
   **Expected:** Terminate in `metadrive_env.py` `step()`, optional live-3D stop, HUD “OFF ROAD”, telemetry `terminated_reason: "off_road"`.  
   **README:** TODO item 2.

3. **Task:** Train the SAC policy.  
   **Why:** RL live view used the fallback controller and often left the road.  
   **Expected:** `python train.py --algorithm sac --timesteps 50000` → `rl/checkpoints/sac_metadrive.zip`, then `run.py --agent rl`.  
   **README:** TODO item 3.

4. **Task:** Auto-save RL checkpoints during training.  
   **Why:** Need periodic + best checkpoints and a training curve.  
   **Expected:** Save every N steps, keep best by eval reward, plot `results/plots/rl_training.png`.  
   **README:** TODO item 4.

5. **Task:** Evaluation and comparison after training.  
   **Why:** Research comparison requires matched Human vs LLM vs RL metrics.  
   **Expected:** `evaluate.py --seeds test --count 20` then `analyze.py`; RMSE lane, MAE speed, collisions, route completion, off-road.  
   **README:** TODO item 5.

6. **Task (optional):** Windows MetaDrive install notes, Ollama setup, HUD off-road/collision warning, `--video` demos.  
   **README:** TODO item 6.

This later **status/handoff** request is documentation-only: inspect, test lightly, update README, write this file. It is **not** a rebuild.

---

## 4. Work You Completed

“You” here means the **continuation work on the local tree after GitHub `main`**, visible as diffs vs https://github.com/krishnasurya9/autonomous-driving-lab/blob/main plus local artifacts. Local git later recorded that work as:

- `b7f67bad` — `Update Autonomous Driving Lab with fixes and optimizations` (local `main`)
- `c122de80` — `ver2: Add Version 2 changelog to README with all fixes and optimizations` (current `ver2`)

These commits are **local only**. GitHub still has only `main` @ `ad60d3c6` and no `ver2` branch.

### 4.1 `rl/reward.py`

- **Change:** Load `reward_weights` from `config.yaml`. Crash/off-road penalties −10. Progressive lane `|dev|^1.5`. Route progress bonus. Steering-squared smoothness term. `compute_reward(..., route_progress, steering)`.
- **Feature:** Configurable multi-objective reward (TODO 1).
- **Why:** README asked for stronger safety and progress shaping.
- **How:** YAML overrides defaults; reward used by `CustomMetaDriveEnv.step`.
- **Testing (this audit):** Unit check: crash/off-road rewards lower than on-road. **PASS**.
- **Limitations:** Weights not used by `train.py` CLI; progress is `speed/target + route_progress * bonus` (not delta-progress). Steering penalty is applied inside `r_speed`.
- **Status:** Implemented. Lightly unit-tested. Not shown to produce a competent on-road policy.

### 4.2 `config.yaml`

- **Change:** Added `reward_weights` block (`w_safety` 1.0 vs original constructor default 5.0; crash/oor −10; progress_bonus 2.0; steering_penalty −0.1).
- **Status:** Implemented. `rl.total_timesteps: 50000` is still unused by `train.py`.

### 4.3 `environment/metadrive_env.py`

- **Change:** If `out_of_road`, set `terminated=True` and `info["terminated_reason"]="off_road"`. Pass `route_progress` and `steering` into reward.
- **Feature:** Off-road episode termination for the Gym wrapper (TODO 2).
- **Testing (this audit):** Env reset/step works (7-D obs). 80 random steps did **not** hit off-road (vehicle barely moved). Termination path not visually confirmed this audit.
- **Status:** Implemented. Not fully verified.

### 4.4 `experiments/run_comparison.py`

- **Change:** Human live loop: crash → terminate; off-road recorded; 1s pause; Tk “Game Over / Try Again?”; engine reused via `while True`.
- **Feature:** Live human restart + crash stop (TODO 2 + gameplay).
- **Testing:** Not re-run in GUI this audit. Prior telemetry files exist for human seeds.
- **Limitations:** `terminated_reason` is **not** passed to `logger.save()`. `extract_state(env)` uses the wrapper, so logged speed/lane are often **zeros** (pre-existing design bug; see §10).
- **Status:** Implemented. Human 3D previously used (many `HUMAN_SCENARIO_2037_*.json`). Restart popup not verified this audit.

### 4.5 `environment/hud.py`

- **Change:** 10 Hz update limiter; flashing `CRITICAL: OFF ROAD!` / `WARNING: COLLISION DETECTED!`; “OFF ROAD — EPISODE ENDED” line.
- **Testing:** Not verified in a live 3D window this audit.
- **Limitations:** Camera label is hardcoded `"CHASE"`; no `C` key handler in this repo.
- **Status:** Implemented, untested this audit.

### 4.6 `rl/train_sac.py`

- **Change:** `Monitor` train env; `SubprocVecEnv` eval env; `EvalCallback` best model; copy `best_model.zip` → `sac_metadrive.zip`; plot `results/plots/rl_training.png` from `evaluations.npz`.
- **Feature:** TODO 3–4 (partial: trained 15k not 50k).
- **Evidence:** Local files `sac_metadrive_checkpoint_{5000,10000,15000}_steps.zip`, `best_model.zip`, `sac_metadrive.zip` (~3.07 MB), `evaluations.npz` timesteps `[5000, 10000, 15000]`, mean eval rewards `[355.88, 1488.87, 1292.98]`, TensorBoard `SAC_1/2/3`.
- **Testing (this audit):** `SAC.load` + dummy `predict` **PASS** when torch is imported **before** MetaDrive. After MetaDrive import, `RLAgent` load failed with `WinError 1114` on `c10.dll` (known Windows order issue).
- **Limitations:** Training still uses **one** start seed (`train_seeds[0]==1000`), not the full 1000–1099 curriculum. Default CLI timesteps 10k; artifacts show 15k.
- **Status:** Implemented and trained to 15k. Policy quality unverified in live 3D this audit.

### 4.7 `evaluation/metrics.py` and `evaluation/compare_agents.py`

- **Change:** Tolerate flat telemetry schema (`experiment_metadata` missing; `steps` vs `time_series_telemetry`). Glob `telemetry/{AGENT}_SCENARIO_*.json` instead of `telemetry/{agent}/seed_*.json`.
- **Why:** `telemetry/logger.py` writes a flat schema and flat filenames (and the whole `telemetry/` dir is gitignored).
- **Testing:** `compute_episode_metrics` on `LLM_SCENARIO_2000_*.json` **PASS** (returns zeros). `compare_telemetry_results` **PASS**, n_paired=4 after this audit’s extra 2037 files.
- **Limitations:** Collisions/off-road often 0 because `save()` does not persist those fields. Visualization scripts still use the **old** glob path, so trajectory/speed plots from `analyze.py` are empty of real overlays.
- **Status:** Partial. Pipeline runs; numbers are not scientifically usable.

### 4.8 `telemetry/logger.py` (created; **gitignored**)

- **Change:** Minimal logger: `steps`, `collisions` in memory, `off_road` flag, save `{AGENT}_{SCENARIO}_{unix}.json`.
- **Why:** Original GitHub tree had no `telemetry/` package (ignored by `.gitignore`). Runtime import would fail without this file.
- **Testing:** In-memory record collision/off-road **PASS**. Feasibility save **PASS**.
- **Limitations:** `save()` omits collisions, off_road, summary_metrics, timestamps. Entire folder ignored by git, so clones from GitHub **will not include this module**.
- **Status:** Implemented locally. Not in GitHub. Schema incomplete.

### 4.9 `telemetry/video_recorder.py`

- **Already used from original `run_comparison`.** Local `recordings/rl_seed2037_1788096261.mp4` exists. Depends on OpenCV. Not re-tested this audit.

### 4.10 `README.md` (pre-this-handoff)

- **Change:** “Version 2” changelog claiming fixes, 15k training, stats CSV, HUD, game-over popup. Overstated completeness (e.g. “Batch evaluation & analysis ✅ Working”, “HD Graphics”).
- **This handoff:** README rewritten to match actual status (see repo `README.md`).

### 4.11 What was **not** added (still original)

Agents, LLM backends, controller math, `app.py`, `run.py`, `train.py` CLI, `evaluate.py`, `analyze.py`, seed JSON contents, `docs/implementation_plan.md` structure, `visualization/rl_training.py` synthetic curve (still present and still synthetic).

Patches named in the Version 2 README inside **installed MetaDrive** (`manual_controller.py`, `register.py`) are **site-packages**, not this repo.

---

## 5. Current Architecture

```text
app.py / run.py / evaluate.py / train.py
        │
        ▼
experiments/run_comparison.py          rl/train_sac.py
        │                                    │
        ▼                                    ▼
CustomMetaDriveEnv  ──step/reset──►  MetaDriveEnv (inner)
        │  obs 7-D, reward DrivingReward
        ▼
HumanAgent | LLMAgent | RLAgent
        │
        ├─ Human: MetaDrive keyboard controller
        ├─ LLM: prompt(DrivingState) → backend → JSON → HighLevelDecision
        │         → LowLevelController → [steer, throttle_brake]
        └─ RL: SAC.predict(obs) or fallback controller
        │
        ▼
TelemetryLogger → telemetry/*.json
        ▼
evaluation/compare_agents.py → results/*.csv
visualization/* → results/plots/   (trajectory/speed globs still wrong)
```

**Communication notes:**

- Live 3D and batch both go through `run_paired_experiment_on_seed`.
- LLM decisions are async; first frames use default `MAINTAIN`.
- SAC must be loaded before MetaDrive/Panda3D on this Windows machine.

---

## 6. Repository Map

| Path | Purpose now |
|------|-------------|
| `README.md` | Current-state documentation (updated this audit) |
| `PROJECT_CONTEXT.md` | This handoff |
| `docs/implementation_plan.md` | Original full design (partly unimplemented) |
| `config.yaml` | Env/LLM/RL + reward weights |
| `requirements.txt` | Unpinned deps (no `panda3d-simplepbr==0.9.0` pin despite README) |
| `app.py` | Tk launcher; batch button runs **5** seeds not 20; preview = human drive |
| `run.py` | Single agent + optional `--video` |
| `train.py` | `train_sac(total_timesteps)` default 10000 |
| `evaluate.py` | `run_paired_suite` default `--count 5` |
| `analyze.py` | compare + four plotters |
| `test_feasibility.py` | Raw MetaDrive 50-step smoke test |
| `environment/` | Wrapper, state, HUD |
| `agents/` | human / llm / rl |
| `llm/` | backends, prompts, parser |
| `controller/` | decision API + low-level control |
| `rl/` | reward, SAC train; `checkpoints/` and `tensorboard_logs/` local only |
| `experiments/seeds/` | train/val/test JSON |
| `experiments/run_comparison.py` | Live/batch loop |
| `evaluation/` | metrics, stats, compare |
| `visualization/` | plots; trajectory/speed expect **wrong** paths |
| `telemetry/` | **gitignored**; logger + JSON logs |
| `results/` | **gitignored**; CSVs/plots |
| `recordings/` | **gitignored**; one RL mp4 |
| `.venv/` | Empty of scientific deps; use `C:\Program Files\Python312\python.exe` |

---

## 7. Current Functionality

What can realistically be run on this machine (Python 3.12 at `C:\Program Files\Python312\python.exe`, protobuf workaround `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`):

- Feasibility smoke test: **yes** (passed this audit; episode ended at step 43).
- Custom env reset/step: **yes**.
- Reward / parser / controller / paired stats unit logic: **yes**.
- SAC zip load + dummy action: **yes if torch before MetaDrive**.
- LLM **real** Ollama `qwen2.5:7b`: **no** — `localhost:11434` timed out this audit. Agent still returns `MAINTAIN` via default/rule fallback.
- Human 3D: code present; previously exercised (telemetry). Not re-opened this audit.
- Full 20-seed Human+LLM+RL experiment: **not ready** (state logging bug, n=3 leftover metrics invalid, LLM not actually queried, human batch would block on keyboard).
- Analysis scripts: **run**, but interpret with extreme caution.

---

## 8. Verified Functionality

Verified **during this 2026-09-18 audit** unless noted.

| Item | Evidence |
|------|----------|
| Imports (numpy 2.4.4, torch 2.11 cpu, SB3, gymnasium) | PASS |
| Reward penalties from config | crash/oor −10; numeric check PASS |
| JSON output parser | valid + malformed PASS |
| Low-level controller bounds | maintain throttle 1; STOP brake −1 PASS |
| Paired tests function | n=4 toy PASS |
| Metrics on existing LLM json | PASS (zeros) |
| SAC checkpoint load + predict | PASS (~3.07 MB zip, 7-D obs, 2-D act) |
| `evaluations.npz` | timesteps 5k/10k/15k PASS |
| MetaDrive import without env var | FAIL protobuf descriptors |
| MetaDrive import with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` | PASS |
| `test_feasibility.py` | PASSED; early end step 43; telemetry wrote heading/speed after a few steps |
| CustomMetaDriveEnv 80 random steps | PASS; no off-road event |
| RLAgent after MetaDrive import | FAIL WinError 1114 `c10.dll` |
| Ollama | FAIL connect timeout |
| LLMAgent without Ollama | PASS returns MAINTAIN; not a real LLM call |
| `compare_telemetry_results` | PASS n_paired 4 seeds `[2000,2001,2002,2037]` |
| Telemetry logger in-memory | PASS |

**Previously executed (not this audit):** SAC training to 15k (checkpoints + TB logs), some human 2037 sessions, 3-seed batch-ish telemetry for 2000–2002, one RL recording mp4, analyze plots.

---

## 9. Unverified Functionality

- Live 3D human WASD, camera `C`, ESC, Game Over dialog
- Live 3D RL with trained weights (DLL order)
- Live 3D LLM with a real local model
- Off-road HUD flash and termination in a window
- `--video` recording path
- `app.py` buttons
- 20-seed evaluation
- SubprocVecEnv training eval on this machine right now
- Seed-matched reproducibility (feasibility reset only printed speed 0 / lane 0)
- HuggingFace transformers backend (file is **rule-based**, not HF)

---

## 10. Known Bugs / Problems

1. **Telemetry state extraction uses the Gym wrapper.** `run_comparison.py` calls `state_adapter.extract_state(env)` where `env` is `CustomMetaDriveEnv` (no `.vehicle`). Adapter falls back to zeros. Batch Human/LLM/RL JSON for 2000–2002 and many 2037 files show `speed_kmh=0`, `lane_deviation_m=0`. **Existing summary_metrics are not valid driving data.** Feasibility works because it uses raw `MetaDriveEnv`.

2. **`TelemetryLogger.save()` drops collisions / off-road / completion time / reason.** Metrics then report 0 collisions and 0 off-road.

3. **`.gitignore` ignores entire `telemetry/`**, including `logger.py` and `video_recorder.py`. GitHub clones cannot import telemetry.

4. **Visualization glob mismatch.** `visualization/trajectories.py` and `speed.py` look for `telemetry/{agent}/seed_*.json`. Logger writes `telemetry/HUMAN_SCENARIO_*.json`. Trajectory/speed plots do not use real logs.

5. **`visualization/rl_training.py` is synthetic** (`np.random` fake curve). Real curve is `results/plots/rl_training.png` from `train_sac.py` if `evaluations.npz` exists.

6. **LLM is not actually invoked** unless Ollama/LM Studio is up. Empty Ollama response falls through to rule backend. Prompt text says `Obstacle Present: NONE` so the rule `"OBSTACLE: YES"` never matches.

7. **Windows torch vs MetaDrive DLL order.** Load SAC before creating MetaDrive.

8. **Protobuf.** Must set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (already in several entry files; raw `import metadrive` without it fails).

9. **Human batch evaluation is interactive.** `evaluate.py` includes `"human"` by default; headless 20-seed human runs are not automated.

10. **`app.py` “20-seed” button uses `--count 5`.** Preview is another human launch.

11. **Training seed coverage.** One map seed 1000, 15k steps, not 50k, not 100 train maps.

12. **Paired stats on current CSVs** show identical RMSE/MAE (all zero) → NaN t-tests. n=3 or 4 is underpowered vs planned 20.

13. **Git `origin` URL contains a GitHub PAT in `.git/config`.** Do not commit or paste that URL. Rotate the token. `git` executable is not on PATH in this audit shell.

14. **Local README Version 2 overclaimed** HD graphics / fully working batch analysis. Corrected in the new README.

15. **Camera `C`** is documented but not implemented in project code (may be MetaDrive default; unverified).

---

## 11. Experimental Setup

### Agents

1. Human — keyboard via MetaDrive manual controller when `manual_control=True` and render on.
2. Small local LLM — intended `qwen2.5:7b` via Ollama; currently fallback rule JSON.
3. SAC — `MlpPolicy`, obs 7-D, act 2-D continuous.

### Environment

MetaDrive `MetaDriveEnv`: `environment_num=1`, `traffic_density=0.1`, `target_speed_kmh=40`, horizon in loops = 1000 steps.

### Seeds

| Split | File | Actual IDs | Plan |
|-------|------|------------|------|
| Train | `experiments/seeds/train_seeds.json` | 1000–1019 (20) | 1000–1099 |
| Val | `validation_seeds.json` | 1100–1109 (10) | 1100–1124 |
| Test | `test_seeds.json` | 2000–2019 (20) | 2000–2019 |

Training **does not iterate** those train seeds; it starts at 1000 only.  
Evaluation **can** loop test seeds in software; paired Human+LLM+RL on the same seed is implemented as nested loops. Default count is 5. GUI uses 5.

### Metrics (intended vs calculated)

| Metric | How calculated now |
|--------|--------------------|
| Lane deviation RMSE | `sqrt(mean(lane_deviation_m^2))` over steps |
| Speed MAE | `mean(\|speed - 40\|)` |
| Steering smoothness | `mean(\|Δsteering\|)` |
| Completion time | `summary.completion_time_sec` or `len(steps)*0.02` |
| Collisions | summary field (usually missing → 0) |
| Off-road events | summary field (usually missing → 0) |
| Task completed | `arrive_dest` or `not terminated` (can mark True if episode just hits max steps without flags) |
| LLM latency / invalid | from summary; logger does not aggregate them |

---

## 12. Current Results

**Do not treat these as research findings.**

From `results/summary_metrics.csv` (pre-overwrite style; values were all lane RMSE 0, speed MAE 40, avg speed 0 for seeds 2000–2002):

- Human episodes ~20 s (1000×0.02) marked `task_completed=True`
- LLM ~1.3–2.5 s `task_completed=False`
- RL ~1.8–3.2 s `task_completed=False`

This is consistent with **zeroed telemetry** and early termination, not with real lane-keeping scores.

Paired tests: only completion_time Human vs RL/LLM looked “significant”; RMSE/MAE differences were 0 with NaN t statistics.

Training eval rewards in `evaluations.npz` (mean): 355.9 @ 5k, 1488.9 @ 10k, 1293.0 @ 15k. These are **simulator eval rewards**, not the paper metrics, and were not independently reproduced this audit.

This audit’s `compare_telemetry_results` rerun expanded common seeds to include 2037; CSVs were regenerated and remain non-interpretable.

---

## 13. Remaining Work (priority)

### P0 — must fix before any real experiment

1. Log state from `env._env` (or expose `.vehicle` on the wrapper).
2. Persist collisions, off-road, `terminated_reason`, timestamps, route progress in telemetry JSON.
3. Stop gitignoring Python under `telemetry/` (ignore `*.json` only).
4. Fix visualization globs to match logger filenames.
5. Confirm Ollama (or LM Studio) is running before claiming LLM results; log backend used.

### P1 — RL / evaluation

6. Load SAC before MetaDrive in `run_comparison.py` (comment exists; still failed this audit when env was created first).
7. Train longer / more seeds if the policy is still off-road; verify live 3D.
8. Headless eval for LLM+RL only; collect human separately; then pair by seed.
9. Run 20 test seeds **after** logging is fixed.
10. Use real `evaluations.npz` in `plot_rl_learning_curve`; remove synthetic curve from the published analysis path.

### P2 — methodology

11. Expand train/val seed lists to match the plan, or document the reduced design.
12. Stats: Shapiro/QQ or Wilcoxon-primary; CI; include collisions/off-road/completion rate; n=20.
13. Pin `requirements.txt` (MetaDrive 0.2.6.0, gymnasium 0.26.2, panda3d-simplepbr 0.9.0, protobuf).
14. Implement or drop camera toggle / scenario preview / HF backend.

---

## 14. Exact Commands

Use system Python 3.12, not `.venv` (that venv has only pip).

```bat
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"C:\Program Files\Python312\python.exe" -m pip install -r requirements.txt
"C:\Program Files\Python312\python.exe" -m pip install metadrive-simulator==0.2.6.0 gymnasium==0.26.2 panda3d-simplepbr==0.9.0

"C:\Program Files\Python312\python.exe" test_feasibility.py

"C:\Program Files\Python312\python.exe" app.py
"C:\Program Files\Python312\python.exe" run.py --agent human --seed 2037
"C:\Program Files\Python312\python.exe" run.py --agent llm --seed 2037
"C:\Program Files\Python312\python.exe" run.py --agent rl --seed 2037
"C:\Program Files\Python312\python.exe" run.py --agent rl --seed 2037 --video

"C:\Program Files\Python312\python.exe" train.py --algorithm sac --timesteps 15000

"C:\Program Files\Python312\python.exe" evaluate.py --seeds test --count 5
"C:\Program Files\Python312\python.exe" analyze.py
```

Ollama (not running at audit time):

```bat
ollama run qwen2.5:7b
```

---

## 15. Important Technical Details

- Observation (7): `speed/80`, `target/80`, `clip(lane/3)`, `clip(heading/45)`, `obstacle_dist/100`, `route_progress`, curvature code.
- Action: `[steering, throttle_brake] ∈ [-1,1]^2`.
- SAC: lr 3e-4, buffer 50k, batch 256, gamma 0.99, MlpPolicy.
- LLM decision interval: `int((1/1.0)/0.02)=50` steps.
- `CustomMetaDriveEnv._get_obs_vector` correctly uses `self._env`; telemetry does not.
- Checkpoints: `rl/checkpoints/sac_metadrive.zip` (copy of best or final). Gitignored.
- Do not leak `.git/config` remote with embedded PAT.

---

## 16. Important Decisions

1. Keep hierarchical LLM + controller; SAC is end-to-end continuous (original architecture, unchanged).
2. Strengthen reward and terminate off-road rather than redesign the stack.
3. Use `SubprocVecEnv` for SB3 eval to avoid MetaDrive singleton crash during training (local continuation).
4. Recreate a **minimal** telemetry logger instead of the nested schema in the implementation plan — caused analysis/visualization mismatches.
5. Train 15k steps as a practical run, not the README’s 50k target.
6. Default evaluate count 5 for speed, while UI still says 20.

---

## 17. Assumptions and Limitations

- Windows + Python 3.12 + NumPy 2.x + CPU torch.
- “LLM agent working” historically meant the **loop runs**, not that qwen2.5 produced the actions.
- Human comparison requires a real driver and the same seeds; current batch human files are not valid kinematics.
- No claim that SAC beats human or LLM.
- Procedural maps with `environment_num=1` per env instance.

---

## 18. Recommended Next Steps

1. Fix `extract_state` to use `env._env` / `env.vehicle`.
2. Fix logger schema + `.gitignore`.
3. Align visualization paths.
4. Re-run one seed each: feasibility-style logging check (speed/lane non-zero).
5. Start MetaDrive **after** `SAC.load` for RL.
6. Start Ollama and log raw LLM JSON for one episode.
7. Only then collect 20-seed LLM vs RL (human separately).

Do not retrain or expand stats until (1)–(4) are done.

---

## HANDOFF SUMMARY

### Starting Point

GitHub `main` already had the full lab: MetaDrive wrapper, three agents, controller, SAC training (untrained), evaluation/stats/viz, GUI, seed files, and README TODOs for reward, off-road stop, training, checkpoints, and 20-seed comparison.

### Tasks Given

README TODOs 1–6: richer reward, off-road terminate, train SAC, checkpointing, evaluate/analyze, optional Windows/Ollama/HUD/video notes.

### Work Completed

Reward YAML + stronger penalties; off-road terminate in wrapper; human crash stop + retry popup; HUD limiter/warnings; SAC EvalCallback + 15k local checkpoints; metrics/compare globs for flat telemetry; local `telemetry/logger.py`; README v2 changelog (then corrected by this audit).

### Current Working Features

Feasibility test; custom env step; unit-level reward/controller/parser/stats; analysis script execution; SAC file load (torch-first); LLM **fallback** loop; local checkpoints on disk.

### Partially Working Features

Off-road termination (code in, not demonstrated this audit); RL live/eval (DLL order); LLM (no server); human 3D (not retested); telemetry (writes files, wrong zeros); statistics (runs on bad data).

### Not Completed

Valid 20-seed Human vs LLM vs RL dataset; real LLM inference in results; 50k training / full train-seed curriculum; nested telemetry schema from the plan; scenario_manager; HF backend; camera/preview as documented; push to GitHub.

### Known Problems

Zeroed batch telemetry; gitignored logger; viz path mismatch; Ollama down; protobuf env var; torch DLL after MetaDrive; PAT in git remote; `git` not on PATH; overstated prior README.

### Current Experimental Readiness

**Not ready** for a paper-style comparison. Ready for **engineering fixes** listed in P0.

### Immediate Next Step

Patch `experiments/run_comparison.py` (and any other logger callers) so `StateAdapter` reads the inner MetaDrive vehicle, then confirm one episode JSON has non-zero speed and lane deviation.

### Commands

See §14. Interpreter: `"C:\Program Files\Python312\python.exe"`.

### Important Files

Next edits: `experiments/run_comparison.py`, `environment/state_adapter.py`, `telemetry/logger.py`, `.gitignore`, `visualization/trajectories.py`, `visualization/speed.py`, `agents/rl_agent.py` / import order in `run_comparison.py`.

### Git (this audit)

- **Current branch:** `ver2` (HEAD file)
- **Local `main`:** `b7f67bad`
- **Latest commit on `ver2`:** `c122de80` — `ver2: Add Version 2 changelog to README with all fixes and optimizations`
- **Working tree after this handoff:** `README.md` and `PROJECT_CONTEXT.md` updated on disk; **not committed** by this audit
- **Push:** GitHub has only `main` @ `ad60d3c6`. No `ver2` remote branch. Continuation work and this documentation are **not** on GitHub
- **`git` CLI:** not recognized in PowerShell (`git` not in PATH). Status inferred from `.git/HEAD`, reflog, and GitHub API
- **Do not push** the remote URL as currently stored; it embeds a credential
