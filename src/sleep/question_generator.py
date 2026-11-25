"""Use the local Qwen model to word each fact as several different questions.

Results are cached on disk (keyed by fact + settings), so re-running a cycle
gives the identical dataset and costs nothing the second time. The model is
loaded LAZILY: if every fact is already cached, Qwen is never loaded, which
leaves the whole 16 GB free for training.
"""
import gc
import hashlib
import json
import re

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from src.model.model_loader import ModelBundle, load_model
from src.utils.config_loader import PROJECT_ROOT, load_config
from src.validation.duplicate_detector import normalise

PROMPT_VERSION = "v1"
CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "question_cache.json"

_PROMPT = (
    "Write {n} different questions that are each answered by the fact below.\n"
    "Rules: every question must be self-contained and name the specific thing it asks about; "
    "never include the answer inside the question; use clearly different wording each time.\n"
    "Output exactly one question per line, with no numbering and no other text.\n\n"
    "Fact: {fact}"
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class LLMQuestionGenerator:
    def __init__(self, bundle: ModelBundle | None = None, cache_file=CACHE_FILE):
        self._bundle = bundle
        self.cache_file = cache_file
        model_name = load_config("model_config")["model"]["name"]
        self.name = f"{model_name}:question-prompt-{PROMPT_VERSION}"
        self.cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    def generate(self, statement: str, n: int) -> list[str]:
        key = hashlib.sha256(f"{PROMPT_VERSION}|{n}|{statement}".encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self._ask_model(statement, n)
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False))
        return self.cache[key]

    def release(self) -> None:
        """Drop the model from memory (call before training)."""
        self._bundle = None
        gc.collect()

    def _ask_model(self, statement: str, n: int) -> list[str]:
        if self._bundle is None:
            self._bundle = load_model()
        tok = self._bundle.tokenizer
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": _PROMPT.format(n=n, fact=statement)}],
            add_generation_prompt=True, enable_thinking=False)
        text = generate(self._bundle.model, tok, prompt=prompt, max_tokens=60 * n,
                        sampler=make_sampler(temp=0.0), verbose=False)
        return parse_questions(text, n)


def parse_questions(text: str, n: int) -> list[str]:
    """Keep real questions only: one per line, bullets removed, no repeats."""
    out, seen = [], set()
    for line in text.splitlines():
        q = _BULLET.sub("", line).strip()
        if q.endswith("?") and normalise(q) not in seen:
            seen.add(normalise(q))
            out.append(q)
    return out[:n]
