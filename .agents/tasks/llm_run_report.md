# LLM Agent Run Report — Test Seeds 2040–2059

**Date:** 2025  
**Task:** Run LLM agent headlessly on all 20 test seeds and verify telemetry with non-zero speed data.

---

## Step A: LM Studio Reachability

**Status: REACHABLE (real inference used)**

LM Studio responded with HTTP 200. Available models at time of run:
- `nvidia/nemotron-3-nano-4b`
- `qwen2.5-coder-7b-instruct`
- `text-embedding-nomic-embed-text-v1.5@q4_k_m`
- `text-embedding-nomic-embed-text-v1.5@f32`
- (+ 2 additional models by hash ID)

The LLM agent used real model inference — the rule-based `TransformersBackend` fallback was **not** needed.

---

## Step B: Evaluation Results

- **Seeds run:** 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059
- **Succeeded:** 20 / 20
- **Failed:** 0 / 20
- **Steps per episode:** 1000 (all hit the time limit — no crashes or off-road terminations)

### Per-Seed Route Completion
| Seed | Route % | Avg Speed (km/h) | Max Speed (km/h) |
|------|---------|-----------------|-----------------|
| 2040 | 79.6%   | 28.7            | 40.0            |
| 2041 | 78.2%   | 28.2            | 40.0            |
| 2042 | 79.6%   | 28.7            | 40.0            |
| 2043 | 79.6%   | 28.7            | 40.0            |
| 2044 | 79.6%   | 28.7            | 40.0            |
| 2045 | 78.2%   | 28.2            | 40.0            |
| 2046 | 76.8%   | 27.7            | 40.0            |
| 2047 | 78.2%   | 28.2            | 40.0            |
| 2048 | 81.0%   | 29.2            | 40.0            |
| 2049 | 80.7%   | 29.0            | 40.0            |
| 2050 | 79.6%   | 28.7            | 40.0            |
| 2051 | 79.6%   | 28.7            | 40.0            |
| 2052 | 78.2%   | 28.2            | 40.0            |
| 2053 | 79.5%   | 28.6            | 40.0            |
| 2054 | 83.8%   | 30.2            | 40.0            |
| 2055 | 79.6%   | 28.7            | 40.0            |
| 2056 | 76.8%   | 27.7            | 40.0            |
| 2057 | 78.2%   | 28.2            | 40.0            |
| 2058 | 78.2%   | 28.2            | 40.0            |
| 2059 | 76.8%   | 27.7            | 40.0            |

---

## Step C: Telemetry Verification

- **Files created:** 20 (one per seed, in `telemetry_2d/`)
- **File pattern:** `LLM_SCENARIO_<seed>_<timestamp>.json`
- **Total steps logged:** 20,000 (1000 per episode)
- **Average speed (across all files):** ~28.5 km/h
- **Max speed observed:** 40.0 km/h (target speed)
- **Non-zero speed steps:** 20,000 / 20,000 (100%) ✓

All telemetry files contain valid, non-zero speed data throughout each episode.

---

## Errors Encountered

None. All 20 seeds completed without errors.

> Note: Multiple files per seed may exist in `telemetry_2d/` due to earlier partial/aborted runs. The canonical run files are those with the highest timestamps matching the successful `run_all_llm.py` execution.

---

## Summary

The LLM agent successfully completed all 20 test episodes using real LM Studio inference (qwen2.5-coder-7b-instruct backend). Average driving speed was ~28.5 km/h against a 40 km/h target — the agent drove conservatively but continuously with no crashes or road departures across all seeds. All telemetry files are confirmed valid with non-zero speed data.
