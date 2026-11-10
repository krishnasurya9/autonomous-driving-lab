# Project context and handoff

This file describes the active project as of 2026-11-07. The current faculty demo target is the lightweight 2D simulator; the older MetaDrive wrapper is retained as legacy code and is not used by the launcher, trainer, or evaluation runner.

## Research objective

Compare a continuous-control SAC policy, a local LLM planner plus low-level controller, and interactive human driving on the same deterministic road seeds. Evaluate route completion, collisions, road departures, lane tracking, speed error, control smoothness, and planner latency. A fallback rule engine is not a real LLM result.

## Active system

- `environment/driving_2d_env.py`: seeded 2D road, obstacle, 20 Hz dynamics, 8-value observation and continuous `[steering, throttle_brake]` action.
- `agents/human_agent.py`: keyboard controls from the rendered simulator.
- `agents/llm_agent.py`: 1 Hz high-level JSON planner and low-level controller; backend/fallback status is stored in telemetry.
- `agents/rl_agent.py`: SAC checkpoint loader with an observation-shape check.
- `experiments/run_comparison.py`: one-agent episodes and seeded suites; rendered episodes are paced at 20 Hz.
- `rl/train_sac.py`: SAC training cycles through manifest seeds and evaluates on the entire validation manifest.
- `telemetry/logger.py`, `evaluation/`, `visualization/`: flat episode JSON, metrics, paired tests, and plots restricted to the current final test manifest.
- `app.py`: Tk launcher for human, SAC, LLM, batch evaluation, and analysis.

The 8-value observation is normalized speed, normalized target speed, lane deviation, heading error, obstacle distance, route progress, road curvature, and signed obstacle lateral offset. The last field was added because the 7-value policy could see an obstacle but not tell which side it occupied. Old 7-value checkpoints are rejected by the active RL agent.

## Seed protocol

| Split | Range | Count | Purpose |
|---|---:|---:|---|
| Train | 1000–1099 | 100 | SAC learning |
| Validation | 1100–1124 | 25 | Checkpoint selection |
| Development audit | 2000–2039 | 40 | Used for earlier 7-value policy audits |
| Held-out comparison | 2040–2059 | 20 | 8-value SAC evaluated once; retain for matching baseline runs |

Seed manifests and their usage rules are in `experiments/seeds/`.

## Verification completed

- Python 3.13.2 environment imports the required Gymnasium, Stable-Baselines3, Torch, YAML, Pandas, Matplotlib, and SciPy packages.
- The 2D feasibility check passed deterministic reset and rollout, finite rewards, expected observation shape, and horizon truncation.
- Seed manifests are unique and disjoint (100 train, 25 validation, 40 development, 20 held-out comparison).
- Controller, output parser, and rule-fallback integration passed smoke checks.
- Telemetry JSON parses into metrics; `evaluate.py` runs a single SAC episode; `analyze.py` produces reports and plots.
- The old 7-value SAC checkpoint loaded, but on seeds 2020–2039 it completed 0/20 routes, timed out 19/20, collided once, and had mean route progress 39.8%. It is incompatible with the updated state and must not be used with the 8-value environment.
- A 1,000-step isolated training pilot verified callbacks and checkpoint saving.
- The 50,000-step 8-value SAC candidate completed all 20 held-out routes, with zero collisions and road departures. Its mean speed was about 86 km/h against the 40 km/h target, indicating a substantial tracking weakness.
- The analysis pipeline ran on the held-out manifest and correctly reported 20 eligible SAC episodes, zero eligible real-LLM or human episodes, and no paired comparisons.

## Current work and remaining requirements

The active state includes obstacle lateral offset, and the low-level controller/fallback planner can select the opposite passing side. The 8-value SAC policy is trained at `rl/checkpoints/candidate_8d/`; `config.yaml` points there. The user chose to keep 2040–2059 and stop retraining. Preserve both the checkpoint and its recorded SAC results; no more model training or tuning is part of the current demo preparation.

LM Studio is running at `127.0.0.1:1234`. The configured Nemotron 3 Nano 4B model returned a response through LM Studio's native chat endpoint, and a 40-step development-seed run recorded `LMStudioBackend` with `llm_real_inference: true`. This confirms connectivity only; it is not held-out evaluation data. Previous human episodes ended early because of `off_road`, not because of the configured time limit. Keyboard steering is now less sensitive, and the final episode result remains visible until the user closes the window. A hands-on keyboard confirmation is still needed.

Before making a complete faculty comparison:

1. Confirm LM Studio is running and the configured model appears at `http://127.0.0.1:1234/v1/models`; verify each collected episode reports `llm_real_inference: true`.
2. Collect interactive human episodes on seeds 2040–2059. The human runner saves them in `telemetry_2d/`.
3. Run real LLM episodes on the same seeds with `evaluate.py --seeds test --count 20 --agents llm`.
4. Confirm all three agents have real, complete telemetry for matching seeds, then run `analyze.py`.
5. Do not report a winner, significance, or a complete three-agent experiment if the paired count is below the intended sample or any LLM runs used fallback. The current SAC speed result must also be disclosed.

## Commands

```powershell
python test_feasibility.py
python train.py
python run.py --agent human --seed 2040
python run.py --agent llm --seed 2040
python run.py --agent rl --seed 2040
python evaluate.py --seeds test --count 20 --agents llm,rl
python analyze.py
```
