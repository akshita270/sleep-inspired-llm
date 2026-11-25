"""Phase 1 check: load Qwen3-4B (full precision, MLX) and time one generation."""
import time
import mlx.core as mx
from mlx_lm import load, generate

MODEL = "Qwen/Qwen3-4B"

t0 = time.time()
model, tokenizer = load(MODEL)
print(f"Load time: {time.time() - t0:.1f}s")

messages = [{"role": "user", "content": "What is the capital of Australia? Answer in one sentence."}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False
)

t0 = time.time()
text = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
dt = time.time() - t0
n_tokens = len(tokenizer.encode(text))
print("Answer:", text.strip())
print(f"Generation: {dt:.2f}s, {n_tokens} tokens, ~{n_tokens/dt:.1f} tok/s")
print(f"Peak memory: {mx.get_peak_memory()/1e9:.2f} GB")
