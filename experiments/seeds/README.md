# Experiment seed protocol

The manifests use disjoint integer ranges so training, model selection, and final comparison stay separated.

- `train_seeds.json`: 100 seeds, 1000–1099. SAC cycles through these seeds at episode resets.
- `validation_seeds.json`: 25 seeds, 1100–1124. The trainer uses these only for checkpoint selection.
- `development_seeds.json`: 40 seeds, 2000–2039. These were used to audit 7-value SAC candidates and are not a final holdout.
- `test_seeds.json`: 20 seeds, 2040–2059. The 8-value SAC policy has already been evaluated once on this range. The project owner chose to stop retraining and retain this range; collect human and real-LLM baseline episodes on these same seeds for the comparison.

Each seed initializes the deterministic 2D road phases, obstacle position, and obstacle lane offset. Use the same seed for each agent. Keep the manifests non-overlapping and do not use the held-out range for policy training or tuning.
