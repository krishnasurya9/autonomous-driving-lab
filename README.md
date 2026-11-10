# Autonomous Driving Lab

A seed-matched driving benchmark with three control sources: interactive human driving, a local language-model planner with a deterministic controller, and a Soft Actor-Critic (SAC) policy. The active simulator is the project's lightweight 2D environment. The older MetaDrive wrapper remains in the repository as a legacy implementation; the demo, training, and evaluation entry points use 2D.

## Research setup

Each episode uses a deterministic procedural road, a single target speed, an eight-value observation, and a two-value action interface. The observation includes signed obstacle lateral offset so SAC and the LLM planner can pass on a known clear side. The 2D simulator advances at 20 Hz. The LLM plans at 1 Hz and the low-level controller applies steering and throttle/brake at every simulator step. SAC predicts continuous steering and throttle/brake actions directly.

The benchmark records route completion, collisions, road departures, lane deviation, speed error, steering smoothness, planner latency, and whether the LLM made a real model request. Rule-based LLM fallback episodes remain visible in the data but are excluded from statistical LLM comparisons.

| Split | Seeds | Use |
|---|---:|---|
| Train | 1000–1099 (100) | SAC learning |
| Validation | 1100–1124 (25) | Checkpoint selection |
| Development audit | 2000–2039 (40) | Used for earlier 7-value model audits |
| Held-out comparison | 2040–2059 (20) | Used for one 8-value SAC evaluation; collect matching human and real-LLM runs |

Seed manifests are in `experiments/seeds/`. The user has chosen to keep `2040–2059` and stop retraining. Continue the comparison on these seeds without changing the policy; the RL results are already recorded, and matching human and real-LLM data are still needed.

## Setup

Use Python 3.13 and install the pinned-by-range project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not on PATH, set the interpreter used by the launcher before starting the GUI:

```powershell
$env:PYTHON = "C:\path\to\python.exe"
& $env:PYTHON app.py
```

## Run the demonstration

```powershell
python run.py --agent human --seed 2040
python run.py --agent llm --seed 2040
python run.py --agent rl --seed 2040
```

Human controls: **W/Up** accelerate, **S/Down/Space** brake, brief taps of **A/Left** or **D/Right** steer, and **Esc** closes the window. A terminal result remains visible until the window is closed. The language-model endpoint and checkpoint path are configured in `config.yaml`. LLM telemetry identifies `OllamaBackend` or `LMStudioBackend` when a real inference request succeeds and `rule_fallback` when it does not.

The RL launcher loads the checkpoint configured in `config.yaml` (the new 8-value candidate path). A model file loading successfully does not establish that its driving performance is safe; inspect episode completion and safety metrics before presenting performance claims.

## Train, evaluate, and analyze

```powershell
python evaluate.py --seeds test --count 20 --agents llm
python analyze.py
python analyze.py
```

The trainer cycles through all seeds in the training manifest and selects the best checkpoint using validation seeds. Evaluation defaults to the full chosen split and runs LLM and SAC headlessly. Human runs are interactive and should be collected separately on matching seeds before making three-agent claims. Generated JSON telemetry is written to `telemetry_2d/`; summaries, paired tests, and plots are written to `results/`.

Use a short run while checking a change:

```powershell
python evaluate.py --seeds test --count 1 --agents rl
```

## Verification status and limits

- `test_feasibility.py` checks reset, observation shape, deterministic seeded initialization, and a 100-step episode in the 2D simulator.
- The training and analysis pipeline uses the shared 2D environment and manifest-based seeds.
- LM Studio's `nvidia/nemotron-3-nano-4b` endpoint is currently reachable. A 40-step development-seed smoke run returned a parsed decision and telemetry reported `LMStudioBackend` with `llm_real_inference: true`. This is a connectivity check, not an evaluation result.
- An LLM episode is a valid LLM result only when telemetry reports `llm_real_inference: true`. If Ollama or LM Studio is offline, the fallback is labeled and excluded from paired LLM statistics.
- The 7-value SAC candidate completed 0/20 routes in the 2020–2039 audit. Its checkpoint is incompatible with the active 8-value interface and is not loaded.
- The 8-value SAC candidate completed 20/20 routes on 2040–2059 with no collisions or road departures, but averaged about 86 km/h against a 40 km/h target. It has a substantial speed-tracking weakness.
- A full faculty comparison still requires matching human and real-LLM episodes with SAC. Until those are collected and analysis reports matching seeds, present the current results as an RL-only evaluation, not a paired comparison.
- The legacy MetaDrive code is not the active demo path.

See [`docs/review_1_2_study_guide.md`](docs/review_1_2_study_guide.md) for the project explanation and [`experiments/seeds/README.md`](experiments/seeds/README.md) for split and provenance notes.
For the remaining data collection and presentation checks, follow [`docs/faculty_demo_runbook.md`](docs/faculty_demo_runbook.md).
