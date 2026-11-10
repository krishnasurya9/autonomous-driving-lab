# Oral Examination — Decisions & Justifications Reference
## Autonomous Driving Benchmarking: SAC vs LLM vs Human

This document covers every reward component and every architectural/design decision made in the project, with the rationale for each choice. Use it as your single reference sheet for the oral.

---

## Part 1 — Reward Function: Every Component Explained

The reward function lives in `rl/reward.py`. Its formula is:

```
R = w_progress × R_progress
  + w_lane     × R_lane
  + w_speed    × R_speed
  + w_safety   × R_safety
  + w_task     × R_task
```

Weights are loaded from `config.yaml` at runtime:

| Weight | Value | Component |
|---|---|---|
| `w_progress` | 0.5 | Forward progress |
| `w_lane` | 1.5 | Lane keeping |
| `w_speed` | 2.0 | Speed tracking |
| `w_safety` | 1.0 | Crash / off-road penalty |
| `w_task` | 10.0 | Task completion |
| `crash_penalty` | −10.0 | On collision |
| `out_of_road_penalty` | −10.0 | On road departure |
| `progress_bonus` | 2.0 | Route fraction multiplier |
| `steering_penalty` | −0.1 | Jerk smoothness |

---

### R_progress — Progress Reward

**Formula:**
```python
r_progress = (speed_kmh / target_speed_kmh) + (route_progress * progress_bonus)
```

**What it rewards:** Moving forward. The speed term rewards the agent for going fast (up to target). The route-progress term gives a growing bonus the further along the route the car gets.

**Why we took it:**
Without a progress signal the agent has no incentive to move at all. A pure sparse reward (only +reward at the goal) gives almost zero gradient signal for 95 % of training because the car crashes or leaves the road before ever reaching the goal. The dense progress term, as shown by Ng et al. (1999), allows reward shaping without changing the optimal policy, and it eliminates the exploration problem.

**Why the route_progress × 2.0 bonus:**
Scaled by `progress_bonus = 2.0` to emphasise the goal direction over raw speed. Without it the agent can earn progress reward by simply driving in circles at high speed.

---

### R_lane — Lane Keeping Penalty

**Formula:**
```python
r_lane = -(abs(lane_deviation_m) ** 1.5) - 0.1 * abs(radians(heading_error_deg))
```

**What it penalises:** Lateral displacement from the road centre, and misalignment of the car's heading with the road direction.

**Why we took it:**
Lane keeping is the core low-level driving skill. Without this term the agent quickly learns to drive off the side of the road because there is no cost to doing so until the binary off-road terminal signal fires.

**Why the exponent 1.5 (not 1.0 or 2.0):**
- Exponent 1.0 (linear): small and large deviations are penalised proportionally — the agent tolerates moderate drift.
- Exponent 2.0 (quadratic): extreme deviations dominate, making the gradient unstable at large errors early in training.
- **Exponent 1.5 (chosen):** progressive — small deviations are tolerated, medium deviations are noticeably penalised, large deviations are harshly penalised. This balances stable early training with tight final lane keeping.

**Why the heading term:**
Pure lateral position can be small even when the car is about to exit the lane (high heading error but just crossed centre). Adding the heading error catches this case early.

---

### R_speed — Speed Tracking Reward

**Formula:**
```python
speed_diff = abs(speed_kmh - target_speed_kmh)
r_speed = 1.0 - min(speed_diff / target_speed_kmh, 1.0)
r_speed += steering_penalty * (steering ** 2)
```

**What it rewards:** Driving at exactly the target speed (40 km/h), and smooth steering.

**Why we took it:**
MAE speed error is a primary evaluation metric in the three-agent comparison. Without a speed reward the SAC agent learns either to drive as fast as possible (maximising progress) or to crawl (minimising lane and safety risk). Neither matches human-like driving.

**Why the normalised form `1 − min(|Δv|/v*, 1)`:**
This maps perfect speed compliance to +1.0 and any speed that is more than `target` km/h away from the target to 0.0. It is bounded, smooth, and scale-free — it does not change if the target speed changes.

**Why the steering-squared penalty is included here (not separately):**
The steering penalty is conceptually a smoothness reward that complements speed tracking: a car that tracks speed well but steers jerkily is not driving well. Combining them in `r_speed` avoids adding an extra weighted term.

**Why steering_penalty = −0.1:**
Small enough not to suppress necessary steering, large enough to discourage oscillation. Larger values (−0.5 and above) caused the agent to under-steer on curves during tuning.

---

### R_safety — Safety Penalties

**Formula:**
```python
r_safety = 0.0
if crashed:
    r_safety += crash_penalty       # −10.0
if out_of_road:
    r_safety += out_of_road_penalty # −10.0
```

**What it penalises:** Colliding with the obstacle, and crossing the road boundary.

**Why we took it:**
Safety is non-negotiable in driving. Without a penalty the agent can learn a policy that scores well on progress and lane metrics while still occasionally crashing.

**Why −10.0 (not a smaller value):**
The initial penalty was −2.0. The agent learned to occasionally crash when the crash allowed faster route progress (the progress reward of a full route was larger than a single −2 penalty). Increasing to −10.0 made safety dominant over short-term progress gains. This matches the design principle: **the magnitude of a terminal penalty must exceed the maximum cumulative reward that can be earned by exploiting the terminal state.**

**Why both penalties can apply simultaneously:**
In corner cases the agent can be both off-road and colliding in the same step (e.g., clipping the obstacle near the lane edge). Both conditions fire their penalty, reinforcing the combined no-crash, no-departure constraint.

---

### R_task — Task Completion Reward

**Formula:**
```python
r_task = 5.0 if arrived_at_destination else 0.0
```
Multiplied by `w_task = 10.0`, giving an effective terminal bonus of **+50** for route completion.

**What it rewards:** Reaching the end of the 500 m route.

**Why we took it:**
Without a goal reward the agent has no explicit incentive to finish the route — it can sit near the start earning progress and speed rewards indefinitely. The large terminal bonus ensures that completing the route is the dominant strategy.

**Why +5.0 × w_task = +50 (a large value):**
The total reward accumulated over a ~500-step successful episode is approximately:
- Progress ≈ 0.5 × (average 1.5) × 500 = 375
- Speed ≈ 2.0 × 0.6 × 500 = 600
- Lane ≈ 1.5 × (−0.3) × 500 = −225
- Net per-step ≈ +750

A terminal bonus of +50 is roughly 6 % of a good-episode cumulative reward — large enough to distinguish "arrived" from "timed out", not so large that it swamps the dense shaping signal.

---

## Part 2 — Every Design Decision Explained

### Decision 1: SAC as the RL Algorithm

**What we chose:** Soft Actor-Critic (Haarnoja et al., 2018) via Stable-Baselines3.

**Why:**
- The action space is **continuous** (steering ∈ [−1, 1], throttle/brake ∈ [−1, 1]). SAC handles this natively; discrete algorithms like DQN require discretisation.
- SAC is **off-policy**: it stores transitions in a 50,000-step replay buffer and re-uses them, making it far more sample-efficient than on-policy methods (PPO, A2C) for a 50,000-step training budget.
- SAC uses a **maximum-entropy objective**: it maximises reward *and* entropy (unpredictability of actions). This encourages exploration and produces policies that are more robust to environment variations — important when evaluating on unseen seeds.
- SAC is the standard choice for continuous control benchmarks in the RL community (MuJoCo, etc.), ensuring our baseline is comparable to published work.

**Why not PPO:** PPO is on-policy — every sample is used once then discarded. At 50,000 steps on a 20 Hz simulator, PPO sees far fewer unique situations. PPO also does not naturally handle the entropy-regularisation that improves robustness.

**Why not DDPG/TD3:** SAC outperforms DDPG on most continuous control tasks due to entropy regularisation. TD3 is similar to SAC but slightly less stable in practice.

---

### Decision 2: 8-Value Observation Space

**What we chose:**
```
[normalized_speed, normalized_target_speed, lane_deviation,
 heading_error, obstacle_distance, route_progress, road_curvature,
 signed_obstacle_lateral_offset]
```

**Why each value:**

| Value | Reason |
|---|---|
| `speed / 80` | The agent must know its own speed to control throttle/brake |
| `target_speed / 80` | The target may change; the agent needs to see it, not hardcode it |
| `lane_deviation / 3.0` | Core signal for lateral control |
| `heading_error / 45.0` | Needed to anticipate curves and correct before deviating |
| `obstacle_distance / 100.0` | Warn the agent before collision; without it the agent cannot brake |
| `route_progress` | Gives the agent a sense of how far it is from the goal |
| `curvature` | Helps anticipate steering need before heading error accumulates |
| `signed_obstacle_lateral_offset` | **Added in v2.** The 7-value policy could detect an obstacle but not which side. Without the sign, the agent could not choose the correct passing direction, leading to 0/20 route completions on the audit seeds. |

**Why the 7-value policy failed:**
The old policy saw "obstacle 15 m ahead" but had no information about whether it was on the left or right. It could not learn a consistent avoidance strategy. The signed offset directly resolved this: positive = obstacle to the right → go left; negative = obstacle to the left → go right.

**Why not raw pixels:**
Pixels require a CNN, vastly more training data, and much longer wall-clock training. The 8-value symbolic state is fully sufficient for the task and keeps training within the 50,000-step budget on a CPU.

---

### Decision 3: Hierarchical LLM Architecture (1 Hz + PD Controller)

**What we chose:** LLM runs at 1 Hz and outputs a JSON maneuver; a PD controller runs at 20 Hz and translates the maneuver into continuous actions.

**Why:**
- LLMs are slow (Nemotron 4B via LM Studio: ~1–3 s per call). They **cannot** run at 20 Hz.
- The separation follows the SayCan architecture (Ahn et al., 2022): LLMs are good at **semantic reasoning** (what to do), while deterministic controllers are good at **physical execution** (how to do it at high frequency).
- The JSON maneuver vocabulary (ACCELERATE, MAINTAIN, SLOW_DOWN, BRAKE, TURN_LEFT, TURN_RIGHT, PREPARE_LEFT, PREPARE_RIGHT, STOP) is **interpretable** — a reviewer can read the LLM log and understand exactly why the car turned or slowed.

**Why 1 Hz (not 0.5 Hz or 2 Hz):**
1 Hz matches typical human re-planning frequency for simple driving. At 0.5 Hz the car can drift significantly before correction; at 2 Hz the LLM latency may exceed the interval, causing queuing.

**Why not run the LLM end-to-end at 20 Hz:**
Latency alone makes this impossible. Even with a fast model, every step would stall waiting for the LLM response.

---

### Decision 4: LM Studio + Nemotron 3 Nano 4B

**What we chose:** LM Studio serving `nvidia/nemotron-3-nano-4b` on `127.0.0.1:1234`.

**Why:**
- **Local inference, no cloud dependency.** The comparison must be reproducible offline.
- **4B parameters fits in RAM.** Larger models (7B+) require more GPU VRAM than available on the demo machine.
- **LM Studio's OpenAI-compatible endpoint** integrates directly with the existing `LMStudioBackend` class using standard `chat/completions` API.
- Nemotron is NVIDIA's instruction-tuned model — it reliably follows JSON output format instructions, reducing invalid-parse failures.

**Fallback chain:**
1. LM Studio (Nemotron) — primary.
2. Ollama backend — if LM Studio is unavailable.
3. Rule-based `TransformersBackend` — absolute fallback; **not counted as a real LLM result** in statistics (`llm_real_inference: false`).

---

### Decision 5: Seed Protocol and Split Design

**What we chose:**

| Split | Seeds | Count | Purpose |
|---|---|---|---|
| Train | 1000–1099 | 100 | SAC learning |
| Validation | 1100–1124 | 25 | Checkpoint selection |
| Development audit | 2000–2039 | 40 | 7-value policy auditing (now legacy) |
| Held-out comparison | 2040–2059 | 20 | Final three-agent comparison |

**Why separate splits:**
- Using the same seeds for training and evaluation gives an optimistically biased result (the agent has seen those road shapes). The held-out split was never used during training or validation.
- 20 paired seeds give enough power for a Wilcoxon signed-rank test to detect medium effect sizes at conventional significance levels.

**Why we stopped retraining and kept 2040–2059:**
The 8-value SAC candidate completed 20/20 routes on these seeds with zero crashes. Retraining would invalidate the already-collected SAC results, forcing re-collection. The speed-tracking weakness is acknowledged and disclosed rather than hidden through further tuning.

**Why the development audit seeds (2000–2039) are excluded from the comparison:**
They were used to audit the 7-value policy and informed the decision to add the signed obstacle offset. Using them for the final comparison would be a form of test-set contamination (design decisions were made based on results from those seeds).

---

### Decision 6: 2D Lightweight Simulator (not MetaDrive 3D)

**What we chose:** A custom `Driving2DEnv` Tkinter-based top-down 2D simulator.

**Why:**
- MetaDrive required Windows-incompatible GPU rendering dependencies that could not be installed in the demo environment.
- The 2D simulator provides all required signals (8-value obs, continuous action, deterministic seeding, obstacle, 500 m route, termination conditions) in pure Python with no external dependencies beyond numpy and tkinter.
- Physics are simplified (1D kinematics + sinusoidal road) but sufficient to test continuous control, lane keeping, obstacle avoidance, and route completion — all the key metrics.

**What is preserved from the original MetaDrive design:**
- Gymnasium interface (reset, step, observation_space, action_space)
- Same seed-to-environment determinism principle
- Same 8-value observation schema
- Same terminal conditions (collision, off-road, arrived, timeout)
- Same evaluation metrics (RMSE lane, MAE speed, steering smoothness, task completion)

**MetaDrive wrapper status:** Retained as legacy code in `environment/metadrive_env.py` but not used by any active launcher, trainer, or evaluator.

---

### Decision 7: Paired Statistical Test Design

**What we chose:** Paired t-test + Wilcoxon signed-rank test, one per metric per agent pair.

**Why paired (not independent samples):**
Different seeds generate roads of different difficulty. An independent-samples test would confound road difficulty with agent capability. Pairing on seed removes this confounding variable — the difference `RL_seed_X − LLM_seed_X` isolates the agent effect.

**Why both tests:**
- **Paired t-test:** Valid when per-seed differences are approximately normal. Parametric, higher power.
- **Wilcoxon signed-rank:** No normality assumption. With only 20 seeds, normality of differences cannot be reliably checked.
- Reporting both provides robustness: if they agree, the conclusion is well-supported; if they disagree, it signals that the distribution of differences is non-normal and the Wilcoxon result should be preferred.

**Why the comparison is not yet complete:**
The held-out split has SAC results on all 20 seeds. Human and real-LLM results are still pending. The paired counts are 0/20 for all three agent pairs. **No winner, significance, or ranking should be claimed until the counts reach 20.**

---

### Decision 8: Telemetry Schema

**What we chose:** Per-episode JSON files with `experiment_metadata`, `summary_metrics`, and `time_series_telemetry` (one object per step).

**Why JSON (not CSV):**
- Episodes have variable length (truncated at 1000 steps or terminated early).
- JSON natively represents nested metadata (agent config, LLM backend, seed) alongside flat per-step arrays.
- Readable without special tools — reviewers can inspect any episode file.

**Why per-step series:**
All derived metrics (RMSE lane, MAE speed, steering smoothness) are computed from the step-level time series, not from summary fields. Storing raw steps preserves the ability to recompute metrics with different formulas without re-running episodes.

**Key telemetry flag: `llm_real_inference`:**
Set to `true` only when the LM Studio or Ollama backend returned a real model response. Episodes where the rule-fallback fired have `llm_real_inference: false` and are excluded from all LLM comparison statistics. This is non-negotiable for scientific validity — a rule-based controller result cannot be reported as an LLM result.

---

## Part 3 — What Is Complete vs. What Is Pending

### Complete ✅
- Seeded 2D driving environment with full Gymnasium interface
- 8-value observation space (including signed obstacle lateral offset)
- Continuous `[steering, throttle_brake]` action space
- Multi-objective shaped reward function (all 5 components)
- SAC training pipeline (100 train seeds, 25 validation seeds)
- 8-value SAC checkpoint: 20/20 routes completed, 0 collisions, 0 road departures
- LLM agent with hierarchical 1 Hz planner + 20 Hz PD controller
- LM Studio backend confirmed working (Nemotron 4B, `llm_real_inference: true`)
- Human agent with keyboard interface (sensitivity tuned, result overlay)
- Telemetry logger (JSON per episode, per step)
- Evaluation metrics: RMSE lane, MAE speed, smoothness, task completion, latency
- Statistical test code (paired t-test + Wilcoxon)
- Analysis pipeline (`analyze.py`) generating plots and comparison report
- 97 result plots (bar charts, speed profiles, trajectories)
- Literature review (9 papers)
- Full seed protocol (train/val/dev/held-out, disjoint, documented)

### Pending ⚠️ (Human evaluation not yet collected)
- 20 × human interactive episodes on seeds 2040–2059
- 20 × real-LLM episodes on seeds 2040–2059 (LM Studio must be running)
- Three-agent paired comparison (RL vs LLM, RL vs Human, LLM vs Human)
- Final statistical results and significance conclusions
- Disclosure of SAC speed-tracking weakness alongside route completion in final report

### Known weakness to disclose
- SAC averaged ~86 km/h vs. 40 km/h target. The policy learned to complete routes but over-speeds substantially. This must be reported in any presentation. The metric is `mae_speed_error_kmh`; the bar chart `bar_mae_speed_error_kmh.png` is in `results/plots/`.

---

## Part 4 — Quick Answer Sheet for Likely Questions

**Q: Why SAC?**
Continuous action space. Off-policy (sample efficient). Maximum-entropy objective improves robustness. Industry standard for continuous control.

**Q: Why reward shaping?**
Sparse rewards provide almost no gradient signal — the car crashes before ever reaching the goal. Dense shaping guides learning without changing the optimal policy (Ng et al., 1999).

**Q: Why is the safety penalty −10 and not −2?**
With −2, the agent learned that the progress reward of a full route outweighed a single penalty, so it occasionally crashed deliberately for faster progress. −10 makes crashes always net-negative regardless of subsequent progress.

**Q: Why the exponent 1.5 in the lane penalty?**
Progressive: tolerates small errors, harshly punishes large ones. Exponent 1 is too flat; exponent 2 creates unstable gradients early in training.

**Q: Why add the signed obstacle lateral offset?**
Without the sign, the 7-value policy saw "obstacle ahead" but could not determine which side to pass on. It completed 0/20 routes in the audit. The sign resolved the ambiguity; the 8-value policy completed 20/20.

**Q: Why is the LLM running at 1 Hz?**
LLM latency is 1–3 seconds per call. 20 Hz is physically impossible. 1 Hz matches human re-planning frequency and keeps the PD controller responsive between decisions.

**Q: Why is the comparison incomplete?**
Human and LLM data collection on seeds 2040–2059 has not been completed yet. The statistical tests require matched pairs. No comparison or winner can be claimed with 0 pairs.

**Q: What is the SAC speed weakness?**
Average speed ~86 km/h vs. 40 km/h target. The reward balance prioritised route completion and progress over speed compliance. The speed penalty weight (`w_speed = 2.0`) was insufficient to fully constrain speed during the available 50,000 training steps.

**Q: Why use paired statistics?**
To eliminate road-difficulty as a confounding variable. A harder seed makes any agent look worse. Pairing by seed removes this bias and isolates the agent capability difference.
