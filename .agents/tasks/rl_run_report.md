# RL Agent Run Report — 20 Test Seeds (2040–2059)

## Checkpoint

- **Path used:** `rl/checkpoints/candidate_8d/sac_2d.zip`
- **Policy:** Real SAC policy (stable-baselines3). The checkpoint observation shape (8,) matched the environment's `observation_space.shape` exactly, so the trained SAC model was used for all 20 seeds. The fallback controller was **not** invoked.

## Telemetry Files

- **Files created:** 20 (one per seed, naming pattern `RL_SCENARIO_<seed>_<timestamp>.json`)
- **Seeds covered:** 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059
- **Output directory:** `telemetry_2d/`

## Spot-Check (Seed 2040)

| Metric | Value |
|---|---|
| File | `RL_SCENARIO_2040_1794034750282.json` |
| Steps | 419 |
| Average speed | 86.1 km/h |
| Max speed | 100.0 km/h |
| Seed | 2040 |
| Task completed | True (`arrived_dest`) |
| Collisions | 0 |
| Off-road events | 0 |
| Final route progress | 100% |

> Note: The RL policy drives at ~86 km/h average, well above the 40 km/h target speed. This is a known characteristic of the trained SAC policy — it was noted in earlier sessions. Retraining was deliberately skipped per project instructions.

## Errors Encountered

None. The checkpoint existed at the configured path, all 20 seeds produced valid telemetry with non-zero speed data, and the SAC model loaded without observation-shape mismatch.

## Summary

All 20 test-seed RL episodes completed successfully using the real SAC policy from `rl/checkpoints/candidate_8d/sac_2d.zip`. Telemetry is available in `telemetry_2d/` for downstream metrics analysis.
