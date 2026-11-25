# LinkedIn drafts (not posted)

All numbers come from `docs/experimental_results.md`.

## Post 1: the idea
I tried to make a local LLM "sleep".

While awake, it collects facts people tell it and checks them. At night, it fine-tunes a small LoRA adapter on them, takes a 220-question exam, and only replaces the live model if it passes pre-agreed rules.

Everything runs on a MacBook M2 with 16 GB, using Qwen3-4B in full precision and MLX.

First lesson: teaching is easy. Not breaking the model is the hard part.

## Post 2: the forgetting result
I taught Qwen3-4B 30 made-up facts with LoRA.

It learned them (83%). But its real-world knowledge fell from 99% to 59%.

Mixing in revision questions answered by the original model cut the damage: old knowledge 93%, forgetting 7.6%. Better, but still above my 5% limit, so the gate rejected it and the original model stayed in charge.

Replay helps. It did not make the problem go away.

## Post 3: five nights of learning
I ran 5 "sleep cycles", 6 new facts each, three ways.

After cycle 5, share of earlier facts still remembered:
- no replay: 8%
- replay of real questions only: 12.5%
- replay of real questions + earlier facts: 62.5%

But general knowledge still eroded in every version (old knowledge 17%, 38%, 40%), and none of the 15 cycles passed my safety gate.

One run per setting, 6 facts per group, so treat small gaps as noise. Full write-up, code and limits in the repo.
