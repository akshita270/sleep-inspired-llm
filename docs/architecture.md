# Architecture

## Goal
Let a local LLM keep learning from its own conversations without retraining from scratch and without silently getting worse.

## Two phases
**Wake** (model answers, nothing is trained)
1. `interaction_classifier` labels each message (QUESTION, NEW_FACT, CORRECTION, FEEDBACK, CHAT) with regex rules and flags personal information.
2. `experience_collector` turns learnable messages into *experiences*.
3. `experience_validator` checks quality (length, completeness), provenance (trusted source or explicit approval), duplicates (Jaccard >= 0.85) and conflicts with stored facts. It checks *form and source*, not truth.
4. Experiences are stored in SQLite as PENDING, APPROVED or REJECTED. Personal information needs explicit approval and is excluded from training by default.

**Sleep** (chat is blocked; shared state AWAKE -> SLEEPING -> TRAINING -> EVALUATING -> AWAKE)
1. `experience_processor` selects APPROVED experiences not yet learned, excluding personal and conflicting ones.
2. `question_generator` asks Qwen for several differently worded questions per fact (cached).
3. `dataset_builder` splits per fact into train/valid/test, removes questions too close to exam questions (leak guard, Jaccard >= 0.8), writes a versioned manifest with sha256.
4. `replay_pool` optionally adds (a) general questions answered by the original model, (b) examples of facts from earlier cycles. Replay goes to the train split only.
5. `lora_trainer` trains a LoRA adapter (rank 16, 16 layers, lr 5e-5, 5 epochs; 14.68M trainable parameters = 0.365%). The base weights are frozen; their SHA-256 is compared before and after training.
6. `evaluation_runner` asks 220 held-out questions (30 new facts, 120 old knowledge, 70 reasoning) and measures forgetting against the original model's correct answers.
7. `champion_manager.run_gate` applies thresholds fixed before the runs. Pass -> promote to champion. Fail -> REJECTED, old champion stays. `rollback_manager` restores the previous champion.

## Why LoRA
Output = W_base x + (B A) x. Only the small matrices A and B train (adapter file about 58.7 MB). The adapter is not merged, so removing it restores the original model exactly.

## Why episodic memory is also tested
Experiment B retrieves stored facts (all-MiniLM-L6-v2, cosine top-3, min similarity 0.30) and puts them in the prompt. It needs no training and is the fair baseline for "does LoRA add anything?".

## Data stores
SQLite (WAL, transactions): `experiences`, `sleep_cycles`, `cycle_experiences`, `system_state`, `model_versions` (with `rolled_back`).
