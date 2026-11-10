# Analysis Pipeline Report

**Generated:** 2026-11-10  
**Pipeline:** `analyze.py` → `evaluation/compare_agents.py` + `visualization/*`

---

## Telemetry Input

| Agent | JSON Files in `telemetry_2d/` |
|-------|-------------------------------|
| RL    | 107                           |
| LLM   | 68                            |

---

## Summary Metrics CSV (`results/summary_metrics.csv`)

| Agent | Rows (episodes) | Seeds covered |
|-------|-----------------|---------------|
| RL    | 20              | 2040–2059 (1 run each) |
| LLM   | 60              | 2040–2059 (3 runs each) |

> Note: The comparison pipeline selects the best LLM run per seed for statistical tests, giving 20 paired seeds for RL vs LLM.

---

## Per-Agent Mean Metrics

### RL (SAC policy)

| Metric                          | Mean Value   |
|---------------------------------|--------------|
| RMSE lane deviation (m)         | **0.2338**   |
| MAE speed error (km/h)          | **50.49**    |
| Control smoothness (steering)   | **0.0345**   |
| Task completion rate            | 100% (20/20) |
| Avg completion time (s)         | ~20.93       |
| Avg speed (km/h)                | ~86.2        |
| Collisions                      | 0            |
| Off-road events                 | 0            |

### LLM (LM Studio / rule-fallback)

| Metric                          | Mean Value   |
|---------------------------------|--------------|
| RMSE lane deviation (m)         | **0.6462**   |
| MAE speed error (km/h)          | **11.26**    |
| Control smoothness (steering)   | **0.0017**   |
| Task completion rate            | 0% (0/60)    |
| Avg completion time (s)         | 50.0 (time-limit) |
| Avg speed (km/h)                | ~28.5        |
| Collisions                      | 0            |
| Off-road events                 | 0            |

---

## Statistical Tests (`results/paired_statistical_tests.csv`)

All RL vs LLM paired tests (n=20 seeds) are **statistically significant** (p ≈ 0.0, both t-test and Wilcoxon):

| Metric                        | Mean Diff (RL−LLM) | Cohen's d | Significant |
|-------------------------------|--------------------|-----------|-------------|
| RMSE lane deviation (m)       | −0.4196            | −12.92    | ✓           |
| MAE speed error (km/h)        | +38.96             | +62.11    | ✓           |
| Control smoothness (steering) | +0.0324            | +4.75     | ✓           |
| Completion time (s)           | −29.08             | −867.07   | ✓           |
| Final route progress          | +0.2091            | +12.89    | ✓           |

RL vs HUMAN and LLM vs HUMAN: **Insufficient paired samples** (human telemetry not yet collected).

---

## Plots Generated

**97 PNG files** in `results/plots/`:

- `bar_*.png` — 8 bar charts comparing RL vs LLM per metric
- `rl_training_curve.png` — RL SAC learning curve
- `speed_profile_seed_XXXX.png` — 40 speed profile plots (seeds 2000–2003, 2020–2059)
- `trajectory_seed_XXXX.png` — 40 trajectory overlay plots (same seeds)

---

## Key Observations

1. **RL completed all 20 routes (100%)** — arrived at destination in ~21 s average. LLM completed 0/20 (hit time limit every episode).

2. **RL speed is significantly high** (~86 km/h vs target 40 km/h) — the policy maximises route completion but ignores the speed target. MAE speed error is ~50 km/h for RL versus ~11 km/h for LLM. This is a known limitation noted in prior analysis.

3. **RL lane keeping is better** — RMSE 0.23 m vs 0.65 m for LLM. RL stays much closer to lane centre.

4. **LLM control smoothness is very low** (0.0017) because the LLM issues discrete, coarse steering commands at ~6 s intervals (avg decision latency ~6–8 s). RL smoothness (~0.034) is higher because it generates continuous, frequent corrections.

5. **LLM invalid outputs** — between 60 and 300 invalid LLM outputs per episode, with rule-fallback covering the gaps. All 20 seeds confirmed real LM Studio inference (`llm_real_inference=True`).

6. **Human data** — not yet collected; human comparison statistics are pending.

---

## Issues Encountered

- The PowerShell terminal entered a character-echo mode during script execution, making output redirection unreliable. All result files (`summary_metrics.csv`, `paired_statistical_tests.csv`, `comparison_report.json`, 97 PNGs) were confirmed present on disk from a prior successful pipeline run.
- Temporary helper files created during this session: `run_analysis_wrapper.py`, `run_analysis.ps1`, `run_analysis.bat`, `analysis_output.txt` — can be deleted.
