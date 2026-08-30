# Autonomous Driving RL Project — Review 1 and Review 2 Study Guide

Use this guide to prepare the project explanation and answer questions. Keep the central framing consistent: **SAC is the primary RL method; the local LLM and human driver are comparison baselines.**

## Project overview

- **Project type:** Reinforcement-learning project with an applied experimental comparison.
- **Formal project title:** *Autonomous Driving Benchmarking: A Seed-Matched Paired Evaluation of Reinforcement Learning, Large Language Models, and Human Baselines.*
- **Research area:** Autonomous driving and continuous control in a seeded simulator.
- **Primary agent:** Soft Actor-Critic (SAC).
- **Baselines:** A local language-model planner and a live human driver.
- **Main idea:** Run each available agent on the same scenarios, record common measures, and compare their behavior per seed.

## Review 1 — Motivation, objective, literature, and problem

### Motivation

Driving combines several competing goals: stay on the road, track a target speed, make progress, and avoid collisions. SAC can learn continuous controls from trial and reward. An LLM planner and a human driver make useful comparison baselines because they use different decision processes.

Agent results are difficult to compare if they face different scenarios. Reusing the same procedural scenario seed helps make the comparison fairer by holding the task constant.

### Problem statement

Develop a continuous-control RL driving policy and assess its behavior relative to a local LLM planner and a human driver under matched scenario conditions.

### Research question

How does a trained SAC policy compare with a local LLM planner and a human driver on the same seeded driving scenarios?

### Objectives

1. Formulate driving as a continuous-control RL task.
2. Train SAC for lane keeping, target-speed tracking, and safe progress.
3. Define LLM and human baselines using the same driving task.
4. Record comparable state, action, and episode telemetry.
5. Compare per-seed safety, tracking, smoothness, and task-completion measures.

### Literature review notes

| Work | Main idea | Connection to this project |
|---|---|---|
| Haarnoja et al. (2018), SAC | Off-policy maximum-entropy RL for continuous actions | Primary method for steering and throttle/brake control |
| Ahn et al. (2022), SayCan | Language planning grounded by lower-level skills | Supports separating high-level decisions from vehicle control |
| Mao et al. (2023), GPT-Driver | Applies language generation to driving motion planning | Related example of language models used for driving decisions |
| Sha et al. (2023), LanguageMPC | Integrates language-model decisions with a controller | Related hierarchical LLM-plus-control approach |
| Dosovitskiy et al. (2017), CARLA | Introduces an autonomous-driving simulator | Background for simulator-based driving research and human baselines |
| Demšar (2006) | Statistical comparison across multiple datasets | Motivates paired comparisons across shared scenario seeds |

**How to state the research gap carefully:** “This project investigates a seed-matched comparison of SAC, a local LLM planner, and live human control.” Avoid saying that no prior work exists unless the broader literature review verifies that claim.

### Short Review 1 explanation

> This is an autonomous-driving reinforcement-learning project. I use SAC as the main agent because the task requires continuous steering and speed control. I compare it with a local LLM planner and a human driver on shared seeded scenarios. I will evaluate safety, lane and speed tracking, smoothness, and task completion. The literature motivates the learning method, the LLM-controller baseline, and paired evaluation; the project is designed to test these agents within one controlled task.

## Review 2 — Methodology, architecture, and experimental setup

### System architecture

1. A seeded scenario initializes the driving environment.
2. The environment provides a shared seven-value observation.
3. The selected agent chooses an action:
   - **SAC:** continuous action each simulation step.
   - **LLM:** high-level maneuver at a configured 1 Hz, followed by a deterministic low-level controller.
   - **Human:** interactive keyboard control.
4. The environment applies steering and throttle/brake and returns the next state and reward.
5. Telemetry records the episode for feature extraction, metrics, visualization, and later comparison.

### Environment and action interface

- The working environment is a lightweight, top-down seeded road simulator.
- The simulation advances at 20 Hz and uses a target speed of 40 km/h.
- The observation has seven values: normalized speed, target speed, lane deviation, heading error, obstacle distance, route progress, and road curvature.
- The action has two continuous values: steering and throttle/brake, each bounded by `[-1, 1]`.
- The current task has simplified vehicle dynamics and a limited set of scenarios; results apply to this task.

### SAC configuration and reward

The project uses Stable-Baselines3 SAC with an MLP policy, learning rate `0.0003`, replay buffer size `50,000`, batch size `256`, and discount factor `0.99`.

The reward combines:

- progress along the route,
- lane and heading error,
- speed error relative to 40 km/h,
- collision and road-departure penalties,
- destination completion reward, and
- a steering smoothness penalty.

If asked why reward shaping is used, explain that it gives the learner feedback before it completes an entire route. Then explain that weights can change what behavior the policy learns, so reward tuning and validation are essential.

### Seed protocol

| Split | Current project seeds | Use |
|---|---:|---|
| Training | 20 (`1000–1019`) | SAC learning; the runner cycles through this split |
| Validation | 10 (`1100–1109`) | Checkpoints and model selection |
| Held-out test | 20 (`2000–2019`) | Final comparison; keep separate from tuning |

For a fair comparison, use the same held-out seed for each agent and compare outcomes within that seed. Do not combine results from different tasks or treat a fallback controller as an LLM result.

### Data collection, preprocessing, features, and EDA

- **Data collection:** per-step JSON telemetry includes seed, agent, state, action, termination, and LLM backend fields.
- **Feature extraction:** speed, lane deviation, heading error, route progress, steering, throttle/brake, and episode outcomes.
- **Preprocessing:** the metrics code parses logs and derives episode-level measures. Before a full experiment, check for missing fields, invalid values, duplicate runs, and incomplete agent/seed sets.
- **EDA:** the analysis pipeline can produce metric summaries, trajectory overlays, speed profiles, and an SAC evaluation curve.
- **Current data limitation:** only a small number of smoke runs exist; plots and summary files are not final benchmark evidence.

### Evaluation measures and planned statistics

- Safety: collisions and road-departure events.
- Tracking: lane-deviation RMSE and speed-error MAE.
- Control: steering smoothness.
- Task performance: route progress, completion, and episode time.
- LLM operation: backend, request latency, and fallback status.

The proposed analysis compares per-seed differences. A paired t-test may be suitable when its assumptions are reasonable; Wilcoxon signed-rank is a non-parametric alternative. Report the number of matched seeds, effect sizes, and uncertainty, and account for multiple comparisons. The current sample is too small to support performance or significance claims.

## Current status to describe honestly

- The dependency, compilation, Gymnasium/SB3, environment, and analysis checks have passed.
- A 5,000-step SAC pilot produced a five-episode validation reward of `886.61 ± 472.52`. The variation is wide; this is an early pilot, not proof of a capable policy.
- An 80-step matched smoke pair on seed 2003 reached the time limit without collision for SAC and the LLM fallback.
- Ollama was not running during the check. The LLM telemetry recorded `rule_fallback`; no real model inference has been verified.
- No human episode has been collected yet.

## Likely review questions

**Why is SAC the primary method?**  
The project is an RL study, and SAC supports continuous actions such as steering and throttle/brake. Its replay buffer reuses experience, while its entropy objective encourages exploration.

**Why include an LLM and a human?**  
They provide comparison baselines with different decision processes. The comparison is meaningful only when they use the same scenario and their actual control source is recorded.

**How do you make the comparison fair?**  
Use the same seeded scenario and common measures for each agent. Keep training, validation, and held-out test seeds separate, and compare per-seed outcomes.

**What does the reward encourage?**  
Progress, lane keeping, target-speed tracking, safety, and destination completion, with a penalty for abrupt steering. The balance of these terms needs validation.

**What have you demonstrated so far?**  
The pipeline runs and the 5,000-step pilot produces a real validation record. A short SAC/fallback smoke pair was also collected. These checks confirm integration, not which agent performs best.

**Has the LLM baseline been tested?**  
The interface and fallback path have been checked, but Ollama was unavailable. The current run is a rule-based fallback, not real LLM inference.

**What remains before drawing conclusions?**  
Train longer and inspect validation variation, run the real local model, collect human episodes, complete matched held-out runs, and then perform the planned analysis.

## Before presenting

- Explain SAC as the main project method; introduce LLM and human agents as baselines.
- Know the observation, action, reward components, and seed splits.
- Distinguish implemented components from planned experiments.
- Describe the pilot as preliminary and disclose the LLM fallback and missing human data.
- Verify bibliographic details and research-gap claims against the original papers.
