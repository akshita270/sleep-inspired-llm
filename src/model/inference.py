"""Ask the model a question and measure how long it takes."""
import time

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from src.model.model_loader import ModelBundle


def ask(bundle: ModelBundle, question: str, context: str | None = None) -> dict:
    """Return the model's answer plus latency and token stats.

    `context` is optional extra text placed before the question. We will use it
    in Experiment B (episodic memory) to hand retrieved facts to the model.
    """
    gen = bundle.config["generation"]
    user_text = f"{context}\n\n{question}" if context else question
    messages = [
        {"role": "system", "content": gen["system_prompt"]},
        {"role": "user", "content": user_text},
    ]
    prompt = bundle.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=gen["enable_thinking"]
    )

    start = time.perf_counter()
    text = generate(
        bundle.model,
        bundle.tokenizer,
        prompt=prompt,
        max_tokens=gen["max_tokens"],
        sampler=make_sampler(temp=gen["temperature"]),
        verbose=False,
    )
    seconds = time.perf_counter() - start

    n_tokens = len(bundle.tokenizer.encode(text))
    return {
        "answer": text.strip(),
        "latency_s": seconds,
        "tokens": n_tokens,
        "tokens_per_s": n_tokens / seconds if seconds > 0 else 0.0,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
