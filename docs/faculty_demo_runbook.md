# Faculty demo runbook

This runbook reflects the current project state: the active simulator is 2D, the 8-value SAC policy is frozen, and seeds 2040–2059 are retained for the comparison. Do not run training for this demo. The SAC result is already collected on all 20 seeds.

## 1. Check the local project

From the project folder, use the Python environment with the project dependencies installed:

```powershell
python test_feasibility.py
```

The check should report deterministic reset and rollout, an 8-value observation, finite rewards, and the 100-step horizon. For a live demonstration, launch `python app.py` or run a single agent with `python run.py --agent rl --seed 2040`.

Human controls are **W/Up** to accelerate, **S/Down/Space** to brake, brief taps of **A/Left** or **D/Right** to steer, and **Esc** to close the simulator. Steering is intentionally less sensitive than the earlier build. When an episode ends, its outcome stays visible until you close the window.

## 2. Collect real local-LLM results

Start the configured Ollama or LM Studio service and ensure the configured model is available. Run a one-seed check first:

```powershell
python run.py --agent llm --seed 2000 --no_render
```

Inspect the new LLM telemetry in `telemetry_2d/`. It must identify a real backend and set `llm_real_inference` to `true`. A `rule_fallback` run is not an LLM result; if inference falls back, restore the model service and repeat the check before collecting the full split.

Once real inference is confirmed, collect the full range:

```powershell
python evaluate.py --seeds test --count 20 --agents llm
```

## 3. Collect human results on the matching seeds

Run one interactive episode for every seed in `experiments/seeds/test_seeds.json`, from 2040 through 2059. In PowerShell, this runs the episodes in order and waits for each window to close:

```powershell
2040..2059 | ForEach-Object { python run.py --agent human --seed $_ }
```

Each run is manual and takes about as long as driving the route. Confirm that a `HUMAN_SCENARIO_<seed>_*.json` telemetry file is saved for every seed. If a route is interrupted or left unfinished, retain its record, repeat that seed if appropriate, and use the successful complete run consistently in the analysis.

## 4. Analyze the matched comparison

The SAC episodes already exist as `RL_SCENARIO_2040_*.json` through `RL_SCENARIO_2059_*.json`. After collecting the LLM and human episodes, run:

```powershell
python analyze.py
```

Review `results/comparison_report.json` for 20 eligible seeds per agent and paired counts of 20 for all agent pairs. Check `results/summary_metrics.csv` and the generated plots for missing, incomplete, duplicate, or fallback episodes before presenting comparative conclusions.

## Current result to disclose

The 8-value SAC policy completed 20/20 routes on seeds 2040–2059 with zero collisions and zero road departures. Its average speed was about 86 km/h against the 40 km/h target, so speed tracking is a substantial weakness. The current analysis contains the SAC evaluation only; there are no eligible human or real-LLM episodes yet, and no three-agent comparison or winner can be claimed.

LM Studio and the selected Nemotron model have now passed a local connectivity and project integration smoke check. Recheck the service before the faculty session. The revised human controls passed keyboard mapping checks, but live keyboard driving still needs a final hands-on check in the simulator window.
