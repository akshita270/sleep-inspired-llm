# Implementation details

## Environment
Apple M2, 16 GB. Python 3.11, MLX 0.32.4, mlx-lm 0.32.2. Model `Qwen/Qwen3-4B` in bf16 (about 8 GB), not 4-bit, by explicit choice. Chat template with `enable_thinking=False`, identical at train and inference time. Greedy decoding (temperature 0), max 256 tokens, system prompt "Answer the question in one short sentence."

## Configs (`configs/`)
- `training_config.yaml`: LoRA rank 16, scale 20, dropout 0, 16 layers, batch 4, 5 epochs, lr 5e-5, max length 256, prompt masking on, seed 42, replay ratio 0 by default, leak-guard similarity 0.80.
- `evaluation_config.yaml`: gate thresholds: taught accuracy >= 0.80, new-knowledge drop <= 0, old-knowledge drop <= 0.02, reasoning drop <= 0.05, forgetting rate <= 0.05, base weights unchanged.
- `system_config.yaml`: DB path, episodic memory (top_k 3, min_similarity 0.30), validation rules, trusted sources.

## Evaluation
220 questions: 30 new (fictional facts), 120 old knowledge, 70 reasoning (arithmetic, logic, word problems, coding). Rule-based whole-word scoring with aliases. A separate 60-question dev set (40 knowledge, 20 reasoning) was used for choosing settings so the final exam was not tuned on. Forgetting = fraction of the original model's correct old-knowledge and reasoning answers that the candidate gets wrong.

## Training
LoRA on 16 layers, 14.68M trainable parameters (0.365%), adapter 58.7 MB. `verify_base_weights` hashes the base weight files (SHA-256) before and after training; in Phase 9 this was done at cycle 1 and cycle 5 only because hashing 8 GB is slow. Cycles are chained with `resume_from`, so one adapter keeps training.

## Replay
- General replay: 100 hand-written general questions answered by the original model (self-distillation), sampled at ratio 1x of the fact examples, filtered against exam and dev questions.
- Previous-facts replay: examples of facts from earlier cycles, sampled at ratio 1x.
Replay goes only into the train split.

## Engineering choices
SQLite in WAL mode with transactions and a partial unique index; cycle status flow RUNNING -> PREPARED (dry run) -> TRAINED -> SUCCESS or REJECTED, or FAILED; versioned dataset manifests with sha256; experiments use fresh throw-away databases and refuse to reuse an existing one. 109 unit tests.

## Study mode
In `experiments/continual_learning.py` the gate verdict is recorded for every cycle but the chain continues regardless. Otherwise a rejection in cycle 1 would make cycles 2 to 5 impossible to study.
