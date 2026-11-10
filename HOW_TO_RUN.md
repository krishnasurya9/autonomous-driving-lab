# HOW TO RUN — Autonomous Driving Lab (2D Simulator)

> Tested on **Python 3.13** (Windows 11, PowerShell). The `.venv/` folder is intentionally empty; use your system Python.

---

## 1. Prerequisites

### Python
Python **3.13** was used for development; Python 3.9+ should also work. Confirm with:

```powershell
python --version
```

### LM Studio (for the LLM agent)
1. Download from [https://lmstudio.ai](https://lmstudio.ai) and install.
2. Inside LM Studio, search for and load **`nvidia/nemotron-3-nano-4b`** (or any instruction-tuned model).
3. Start the **Local Server** — it listens on `http://127.0.0.1:1234` by default.
4. The config already points there (`llm.lmstudio_url` in `config.yaml`).

### Ollama (alternative LLM backend)
If you prefer Ollama instead of LM Studio:

```powershell
ollama pull qwen2.5:7b
ollama serve
```

Then change `config.yaml`:

```yaml
llm:
  backend: ollama   # was: lmstudio
```

### Windows — protobuf environment variable
Set this **before every Python call** to avoid a `google.protobuf` crash at startup:

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
```

You can add it to your PowerShell profile (`$PROFILE`) so it is set automatically.

---

## 2. Installation

```powershell
cd "a:\MSC\sem 3\Rl\project"
pip install -r requirements.txt
```

The `requirements.txt` pins the following packages (notable ones):

| Package | Version constraint |
|---|---|
| `torch` | >=2.8 |
| `stable-baselines3` | >=2.8 |
| `gymnasium` | >=1.0, <2.0 |
| `numpy` | >=2.1 |
| `pandas` | >=2.2 |
| `matplotlib` | >=3.9 |
| `seaborn` | >=0.13 |
| `scipy` | >=1.14 |
| `PyYAML` | >=6.0 |
| `requests` | >=2.32 |
| `tqdm` | >=4.66 |
| `tensorboard` | >=2.18 |

If `torch` installation takes too long or fails, install it separately first from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) with the correct CUDA/CPU variant.

---

## 3. Demo Game Window (interactive, single episode)

These commands open the top-down 2D driving window for one episode. No batch data is collected; no training happens.

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Human player — drive with WASD or arrow keys
python run.py --agent human --seed 2040

# RL agent (pre-trained SAC policy)
python run.py --agent rl --seed 2040

# LLM agent — requires LM Studio running on port 1234
python run.py --agent llm --seed 2040

# Headless (no window) — useful for quick smoke tests
python run.py --agent rl --seed 2040 --no_render
```

**Available flags for `run.py`:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--agent` | `human` / `rl` / `llm` | *(required)* | Agent to run |
| `--seed` | int | `2037` | Procedural scenario seed |
| `--no_render` | flag | off | Suppress the game window |
| `--video` | flag | off | Reserved; not yet configured |

---

## 4. Full Evaluation Pipeline (batch, headless, 20 seeds)

This runs all 20 test seeds for LLM and RL agents and writes telemetry JSON files to `telemetry_2d/`. It must be run **before** `analyze.py`.

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Both agents together (default)
python evaluate.py --seeds test --agents llm,rl
```

Or evaluate each agent separately:

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

python evaluate.py --seeds test --agents llm
python evaluate.py --seeds test --agents rl
```

Run only a subset (e.g. first 5 seeds):

```powershell
python evaluate.py --seeds test --agents llm,rl --count 5
```

**Available flags for `evaluate.py`:**

| Flag | Choices / type | Default | Description |
|---|---|---|---|
| `--seeds` | `test` / `validation` / `train` | `test` | Seed split to use |
| `--agents` | comma-separated `llm`, `rl` | `llm,rl` | Agents to evaluate (human not supported headless) |
| `--count` | int | *(full split)* | Limit to first N seeds |

Seed manifests live in `experiments/seeds/` (`test_seeds.json`, `validation_seeds.json`, `train_seeds.json`).

---

## 5. Train RL from Scratch (optional)

> **Do NOT retrain unless you need a new model.** The existing checkpoint (`rl/checkpoints/candidate_8d/sac_2d.zip`) was trained for **50,000 timesteps** and is already configured in `config.yaml`.

If you do want to retrain:

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Default: 50,000 timesteps from config.yaml
python train.py --algorithm sac

# Override timesteps
python train.py --algorithm sac --timesteps 50000

# Custom checkpoint output directory
python train.py --algorithm sac --checkpoint-dir rl/checkpoints/my_run

# Use specific seed manifests
python train.py --algorithm sac --train-seeds experiments/seeds/train_seeds.json --validation-seeds experiments/seeds/validation_seeds.json
```

**Available flags for `train.py`:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--algorithm` | `sac` | `sac` | RL algorithm |
| `--timesteps` | int | from `config.yaml` (`50000`) | Total training steps |
| `--train-seeds` | path | from `config.yaml` | Training seed manifest JSON |
| `--validation-seeds` | path | from `config.yaml` | Validation seed manifest JSON |
| `--checkpoint-dir` | path | auto | Output directory for checkpoints |

Training writes checkpoints to `rl/checkpoints/` and TensorBoard logs to `rl/tensorboard_logs/`. Monitor with:

```powershell
tensorboard --logdir rl/tensorboard_logs
```

---

## 6. Run Analysis and Generate Plots

Run this **after** `evaluate.py` has populated `telemetry_2d/`.

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
python analyze.py
```

This runs the full pipeline in sequence:
1. Computes per-episode metrics and paired statistical tests.
2. Generates bar charts, trajectory overlays, and speed profile plots.
3. Plots the RL learning curve from TensorBoard logs.

**Outputs written to `results/`:**

| File | Contents |
|---|---|
| `results/summary_metrics.csv` | Per-episode metrics for all agents (speed, lane error, task completion, etc.) |
| `results/paired_statistical_tests.csv` | Wilcoxon signed-rank / t-test comparisons between agents |
| `results/comparison_report.json` | Full structured report (all metrics + test results) |
| `results/plots/*.png` | Performance bar charts, trajectory overlays, speed profiles, RL learning curve |

---

## 7. Project File Structure

```
a:\MSC\sem 3\Rl\project\
│
├── run.py                  Single-episode launcher (interactive or headless)
├── evaluate.py             Batch evaluation launcher (20-seed test suite)
├── train.py                RL training launcher
├── analyze.py              Statistical analysis and plot generation
├── config.yaml             All configuration (env, RL, LLM, evaluation settings)
├── requirements.txt        Python dependencies
│
├── agents/                 Human, LLM, and RL agent implementations
├── environment/            2D driving environment and state adapter (Driving2DEnv)
├── experiments/            Episode runner and seed manifests (test/train/val JSON files)
├── evaluation/             Metrics computation and statistical tests (compare_agents.py)
├── visualization/          Plot generation (performance, trajectories, speed, RL curve)
├── telemetry/              TelemetryLogger — writes per-step JSON to telemetry_2d/
├── telemetry_2d/           Episode telemetry JSON files (gitignored — generated at runtime)
├── results/                CSV, JSON outputs, and PNG plots (gitignored — generated at runtime)
├── rl/                     SAC reward function, train_sac.py, checkpoints/, tensorboard_logs/
│   └── checkpoints/
│       └── candidate_8d/
│           └── sac_2d.zip  ← active checkpoint (set in config.yaml)
├── llm/                    LLM backends: LM Studio, Ollama, Transformers fallback
├── controller/             Low-level continuous controller
└── docs/                   Project documentation and literature review
```

---

## 8. Recommended Run Order (Full Workflow)

```powershell
# 0. Set environment variable (do this once per terminal session)
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# 1. (Optional) Watch the RL agent drive in the game window
python run.py --agent rl --seed 2040

# 2. (Optional) Watch the LLM agent (requires LM Studio on port 1234)
python run.py --agent llm --seed 2040

# 3. Run the full batch evaluation (all 20 test seeds)
python evaluate.py --seeds test --agents llm,rl

# 4. Run analysis and generate all plots
python analyze.py

# 5. Find outputs in results/
```

---

## 9. Common Errors and Fixes

### `google.protobuf` error at startup
```
RuntimeError: ... google.protobuf ...
```
**Fix:** Set the environment variable before every Python call:
```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
```

---

### DLL load failed (Windows — PyTorch or other native extension)
**Fix:** Add `import torch` as the very first import in `run.py`, before any other project imports. This ensures the PyTorch DLLs are resolved first.

---

### LM Studio not running / connection refused
The LLM agent automatically falls back to the rule-based `TransformersBackend` if the LM Studio server is unreachable.

**Alternative:** Use Ollama instead:
```powershell
ollama serve
```
Then set in `config.yaml`:
```yaml
llm:
  backend: ollama
```

---

### RL checkpoint not found
```
FileNotFoundError: ... sac_2d.zip ...
```
**Fix:** Confirm the checkpoint exists at `rl/checkpoints/candidate_8d/sac_2d.zip`. Verify `config.yaml` matches:
```yaml
rl:
  checkpoint: rl/checkpoints/candidate_8d/sac_2d.zip
```

---

### `seaborn` not installed
```
ModuleNotFoundError: No module named 'seaborn'
```
**Fix:**
```powershell
pip install seaborn
```

---

### `telemetry_2d/` is empty — analyze.py produces no output
**Fix:** Run `evaluate.py` first to populate the telemetry directory:
```powershell
python evaluate.py --seeds test --agents llm,rl
```
Then re-run `analyze.py`.

---

### All metrics are zero in `summary_metrics.csv`
**Cause:** `StateAdapter` is returning zero speed / zero lane deviation.
**Fix:** Verify that `Driving2DEnv.get_driving_state()` is defined and returns valid values. Check `environment/` for the method definition.

---

### `results/` or `telemetry_2d/` folders missing
Both directories are gitignored and created at runtime. If they are absent, `evaluate.py` and `analyze.py` create them automatically on first run.

---

*Last updated: generated from live project source.*
