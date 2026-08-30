# Implementation Plan - Autonomous Driving Comparison (Human vs Small LLM vs RL in MetaDrive)

This plan outlines the complete end-to-end technical implementation for evaluating and comparing **Human**, **Small Local LLM**, and **Reinforcement Learning (SAC)** autonomous driving agents in procedurally generated **MetaDrive** simulation environments.

The simulator (MetaDrive) is used as a neutral experimental laboratory. All three agents will be benchmarked on identical, seed-matched procedural driving tasks under consistent evaluation metrics and telemetry logging.

---

## Architecture Overview

```text
                         METADRIVE SIMULATOR
                                  │
                                  ▼
                         ENVIRONMENT STATE
                                  │
                                  ▼
                       STATE ABSTRACTION LAYER
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
       HUMAN AGENT            LLM AGENT              RL AGENT
   (Keyboard / Direct)     (High-level Planner)     (Continuous SAC)
            │                     │                     │
            │               Decision API                │
            │            (JSON Maneuvers)               │
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  ▼
                         LOW-LEVEL CONTROLLER
                     (PD / Longitudinal Control)
                                  │
                                  ▼
                     STEERING / THROTTLE / BRAKE
                                  │
                                  ▼
                              TELEMETRY
                                  │
                                  ▼
                       EVALUATION & STATISTICS
```

---

## Proposed Directory & File Structure

```text
a:\MSC\sem 3\Rl\project\
├── README.md                          # Project documentation and CLI usage instructions
├── requirements.txt                   # Dependency list (metadrive-simulator, stable-baselines3, etc.)
├── config.yaml                        # Central configuration file
├── run.py                             # Main CLI runner entrypoint
├── train.py                           # RL policy training launcher
├── evaluate.py                        # Batch evaluation runner over seed splits
├── analyze.py                         # Statistical testing and metrics summary tool
│
├── environment/
│   ├── __init__.py
│   ├── metadrive_env.py               # Custom MetaDrive wrapper with standardized Gym interface
│   ├── state_adapter.py               # Extracts structured DrivingState dataclass from MetaDrive
│   ├── scenario_manager.py            # Manages procedural seed loading and environment resetting
│   └── environment_config.py          # Config structures for road blocks, traffic, and target speeds
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                  # Abstract base class for all driving agents
│   ├── human_agent.py                 # Human manual control agent (MetaDrive keyboard integration)
│   ├── rl_agent.py                    # Trained SAC agent inference wrapper
│   └── llm_agent.py                   # LLM driving planner with decision frequency gating
│
├── llm/
│   ├── __init__.py
│   ├── backend.py                     # Abstract LLM backend interface
│   ├── ollama_backend.py              # Ollama local REST API client adapter
│   ├── lmstudio_backend.py            # LM Studio OpenAI-compatible REST API adapter
│   ├── transformers_backend.py        # Local HuggingFace Transformers fallback backend
│   ├── prompts.py                     # Structured prompt templates for DrivingState -> JSON decision
│   └── output_parser.py               # JSON output parser with schema validation & fallback handling
│
├── controller/
│   ├── __init__.py
│   ├── decision_api.py                # Standardized High-Level Decision object (maneuver, target speed, lane)
│   └── low_level_controller.py        # Controller mapping high-level decisions to (steering, throttle, brake)
│
├── rl/
│   ├── __init__.py
│   ├── train_sac.py                   # SAC training script with checkpoint saving and TensorBoard logging
│   ├── evaluate_rl.py                 # Standalone RL evaluation script
│   ├── reward.py                      # Multi-objective reward function (progress, lane, speed, safety, task)
│   └── checkpoints/                   # Directory to store trained SAC model checkpoints
│
├── experiments/
│   ├── __init__.py
│   ├── seeds/
│   │   ├── train_seeds.json           # Seeds 1000–1099 (RL training set)
│   │   ├── validation_seeds.json      # Seeds 1100–1124 (Development & hyperparameter tuning)
│   │   └── test_seeds.json            # Seeds 2000–2019 (Held-out evaluation set)
│   ├── scenarios/                     # Scenario definition manifests (speed follow, curves, turns, stops)
│   ├── run_human.py                   # Script for interactive human benchmark runs
│   ├── run_llm.py                     # Script for LLM agent benchmark runs
│   ├── run_rl.py                      # Script for RL agent benchmark runs
│   └── run_comparison.py              # Automated paired scenario test suite
│
├── telemetry/
│   ├── logger.py                      # Telemetry recorder saving JSON time series and episode summaries
│   ├── human/                         # Telemetry output directory for human runs
│   ├── llm/                           # Telemetry output directory for LLM runs
│   └── rl/                            # Telemetry output directory for RL runs
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Metric computations (RMSE lane dev, MAE speed, smoothness, success)
│   ├── statistical_tests.py           # Paired t-test and Wilcoxon signed-rank test routines
│   └── compare_agents.py              # Cross-agent paired statistical comparison tool
│
├── visualization/
│   ├── __init__.py
│   ├── trajectories.py                # Overlay 2D trajectory plotting script
│   ├── performance.py                 # Bar & box plots for agent benchmarks
│   ├── speed.py                       # Speed profile vs. time plots
│   └── rl_training.py                 # Reward convergence curve generator
│
└── results/                           # Evaluation outputs, plots, and statistical summary reports
```

---

## Detailed Step-by-Step Implementation Strategy

### Step 1: Environment & Dependency Setup
1. Create `requirements.txt` with required dependencies: `metadrive-simulator`, `stable-baselines3`, `gymnasium`, `torch`, `requests`, `pyyaml`, `pandas`, `matplotlib`, `scipy`, `seaborn`.
2. Install `metadrive-simulator` and `stable-baselines3`.
3. Create `config.yaml` specifying central defaults for environment, RL training, LLM backend (Ollama default with `qwen2.5:7b` or `gemma2:2b`/`llama3.2:3b`), decision frequency (1 Hz), and target speed (40 km/h).

### Step 2: Minimal Feasibility Milestone Test
Build a minimal test script `test_feasibility.py` to verify:
- MetaDrive engine initializes without GUI errors (or with offscreen/onscreen rendering).
- Environment resets with a specified seed.
- Observation vector is retrieved.
- Dummy vehicle actions `[steering, throttle]` step the simulation.
- Telemetry logs basic step information.
- Environment resets to the same seed reproducibly.

### Step 3: State Abstraction, Seed Management, & Low-Level Controller
1. **`environment/state_adapter.py`**:
   - Define `DrivingState` dataclass (`speed_kmh`, `target_speed_kmh`, `lane_deviation_m`, `heading_error_deg`, `road_curvature`, `upcoming_maneuver`, `distance_to_maneuver`, `obstacle_present`, `obstacle_distance`, `route_progress`, `distance_to_goal`).
   - Extract raw signals from MetaDrive (`vehicle.speed_kmh`, `vehicle.lane.dist_to_line`, `vehicle.heading_diff`, navigation routing info).
2. **`experiments/seeds/`**:
   - Create JSON manifests for train (`1000..1099`), validation (`1100..1124`), and test (`2000..2019`).
3. **`controller/decision_api.py` & `controller/low_level_controller.py`**:
   - High-level decision JSON structure: `{"target_speed_kmh": float, "maneuver": str, "lane_target": str}`.
   - Maneuvers: `ACCELERATE`, `MAINTAIN`, `SLOW_DOWN`, `BRAKE`, `TURN_LEFT`, `TURN_RIGHT`, `PREPARE_LEFT`, `PREPARE_RIGHT`, `STOP`.
   - Low-level controller (PID / proportional control) mapping target speed and maneuver to exact continuous control `(steering, throttle, brake)`.

### Step 4: Agent Implementations
1. **`agents/human_agent.py`**:
   - Wraps MetaDrive's manual keyboard control interface (`W/A/S/D` or Arrow keys).
   - Logs identical telemetry format per step.
2. **`agents/rl_agent.py` & `rl/`**:
   - Observation space: Normalized compact state vector (speed, target speed, lane deviation, heading error, road curvature, distance to maneuver, route progress).
   - Continuous Action Space: `steering ∈ [-1, 1]`, `throttle/brake ∈ [-1, 1]` mapped to MetaDrive control.
   - Multi-objective reward function in `rl/reward.py`:
     $$R = w_1 R_{\text{progress}} + w_2 R_{\text{lane}} + w_3 R_{\text{speed}} + w_4 R_{\text{safety}} + w_5 R_{\text{task}}$$
   - Train continuous SAC policy using Stable-Baselines3 on training seeds (`1000..1099`). Save best checkpoint to `rl/checkpoints/sac_metadrive.zip`.
3. **`agents/llm_agent.py` & `llm/`**:
   - `LLMBackend` base class with `generate_decision(driving_state)` method.
   - Implement `OllamaBackend` (connecting to local Ollama http://localhost:11434), `LMStudioBackend` (http://localhost:1234/v1), and `TransformersBackend` fallbacks.
   - Prompts in `llm/prompts.py` instructing strict JSON output matching schema.
   - `output_parser.py` validates output, handles invalid JSON gracefully with fallback (`MAINTAIN` / gentle brake), and records `llm_invalid_output` flags in telemetry.
   - Operates at configurable low decision frequency (~1 Hz), while low-level controller operates at simulator step frequency (~20–50 Hz).

### Step 5: Evaluation Framework & Paired Testing
1. **Telemetry Logging (`telemetry/logger.py`)**:
   - Saves experiment metadata, summary metrics, and time series step telemetry to JSON files.
2. **Paired Test Execution (`experiments/run_comparison.py`)**:
   - Freeze trained RL agent.
   - Loop over test seeds (`2000..2019`).
   - For each seed:
     1. Reset MetaDrive with seed -> Run Human (or recorded input) -> Save `telemetry/human/seed_XXXX.json`.
     2. Reset MetaDrive with seed -> Run LLM Agent -> Save `telemetry/llm/seed_XXXX.json`.
     3. Reset MetaDrive with seed -> Run RL Agent -> Save `telemetry/rl/seed_XXXX.json`.

### Step 6: Statistical Analysis & Visualization
1. **`evaluation/metrics.py`**:
   - Computes Task Completion Rate, Collision Count, Off-road events, RMSE Lane Deviation, MAE Speed Error, Control Smoothness ($J_{\text{steering}} = \frac{1}{T-1}\sum |u_{t+1}-u_t|$), Completion Time, LLM Latency.
2. **`evaluation/statistical_tests.py`**:
   - Paired t-test and Wilcoxon signed-rank test for paired seed-level differences ($RL \text{ vs } LLM$, $RL \text{ vs } Human$, $LLM \text{ vs } Human$).
3. **`visualization/`**:
   - Plots trajectory overlays, speed profile comparison curves, lane deviation over time, bar/box plots of metrics, and RL training curves.

---

## Verification Plan

### Automated Tests
1. **Feasibility Test**: Run `python test_feasibility.py` to ensure MetaDrive engine launches, steps, and resets deterministically.
2. **Unit & Module Integration Tests**:
   - Test `StateAdapter`: Ensure MetaDrive states extract correctly without `NaN` or invalid bounds.
   - Test `LowLevelController`: Input high-level maneuvers and check output steering/throttle/brake bounds.
   - Test `OutputParser`: Send valid and malformed LLM JSON strings to verify fallback mechanism.
   - Test `TelemetryLogger`: Verify JSON output structure against spec schema.
3. **RL Pipeline Verification**: Train SAC for a short sanity-check run (e.g. 5,000 steps) to verify training loop, reward calculation, and model saving/checkpoint loading.
4. **LLM Connection Test**: Test local Ollama / LMStudio endpoint with a mock state.

### Manual & Benchmark Verification
1. **Human Driver Test**: Run `python run.py --agent human --seed 1100` and drive using keyboard controls to verify rendering, controls responsiveness, and telemetry recording.
2. **Paired Benchmark Execution**: Run `python evaluate.py --seeds test` to verify all 20 test seeds run cleanly for RL and LLM agents.
3. **Statistical Summary Report**: Run `python analyze.py` to generate statistical tables and plots in `results/`.
