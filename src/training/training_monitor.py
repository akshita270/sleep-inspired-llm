"""Read the trainer's log text and pull out the numbers worth keeping.

mlx-lm prints a coloured table; after removing colour codes it looks like:
    Trainable parameters: 0.365% (14.680M/4022.468M)
       1    val 5.423    8.07s
       5    4.044 ▼       35    0.5k          <- iter, train loss, tokens/s, tokens
      29    val 12.651    7.45s
    Test loss 5.060, Test ppl 157.600.
"""
import re

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TRAIN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▼▲=]?\s+(\d+)\s+[\d.]+k?\s*$", re.M)
_VAL = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)", re.M)
_TEST = re.compile(r"Test loss ([\d.]+)")
_PARAMS = re.compile(r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/([\d.]+)M\)")


def parse_log(text: str) -> dict:
    text = _ANSI.sub("", text)
    train = [(int(i), float(v)) for i, v, _ in _TRAIN.findall(text)]
    tok_s = sorted(int(t) for _, _, t in _TRAIN.findall(text))
    val = [(int(i), float(v)) for i, v in _VAL.findall(text)]
    test = _TEST.findall(text)
    params = _PARAMS.search(text)
    return {
        "train_loss": train,
        "val_loss": val,
        "first_train_loss": train[0][1] if train else None,
        "final_train_loss": train[-1][1] if train else None,
        "max_train_loss": max(v for _, v in train) if train else None,
        "first_val_loss": val[0][1] if val else None,
        "final_val_loss": val[-1][1] if val else None,
        "best_val_loss": min(v for _, v in val) if val else None,
        "test_loss": float(test[-1]) if test else None,
        "median_tokens_per_s": tok_s[len(tok_s) // 2] if tok_s else None,
        "trainable_percent": float(params.group(1)) if params else None,
        "trainable_millions": float(params.group(2)) if params else None,
    }
