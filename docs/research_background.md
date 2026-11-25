# Research background

This project is a simplified, laptop-scale take on the idea from *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*. This is an independent implementation of the idea; it is not a reproduction of the paper's experiments or numbers.

## Ideas borrowed
- **Wake/sleep split.** Experiences are gathered while the model is in use, then consolidated into weights in a separate phase.
- **Consolidation into parameters.** New knowledge is moved into a small trainable module (LoRA) instead of a growing prompt.
- **Safe self-modification.** A candidate adapter must beat pre-agreed checks before it replaces the champion, with rollback.

## Not implemented (on purpose)
Knowledge Seeding and Expandable Experts.

## Concepts used
- **Catastrophic forgetting:** training on new data degrades performance on old data.
- **Rehearsal / replay:** mixing old data into new training. General replay here is self-distillation (the original model's own answers).
- **Retrieval (episodic memory):** keep facts outside the weights and look them up.
- **Champion/candidate:** the deployed model is the champion; a newly trained adapter is a candidate until it passes the gate.
