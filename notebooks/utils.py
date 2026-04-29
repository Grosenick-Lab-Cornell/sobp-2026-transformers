"""Helpers for the SOBP 2026 LLM-Basics talk notebooks.

Imported by both NB1 and NB2 setup cells. Designed for fast iteration:
notebooks pull this file via `git pull` so changes here propagate
without re-loading the model (which is the slow part).

Two generation paths:
  call_llm(...)         live model generation (the default)
  cached_call_llm(...)  replays a saved output from
                        content/cached_outputs.json with a small
                        per-character delay so it looks live; falls
                        through to call_llm() if the label is missing.

The cached path is what the §4 long-chart demos use on stage so the
talk does not depend on multi-minute prefills working perfectly.
"""
from __future__ import annotations

import os

# Set the CUDA allocator config before torch initializes its CUDA backend.
# `expandable_segments:True` reduces fragmentation on long-context calls.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json as _json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

_OUTLINES_CACHE: dict = {}
_OUTPUT_CACHE: dict | None = None
_CACHE_PATH = Path(__file__).resolve().parent.parent / "content" / "cached_outputs.json"

# Flip to True at the notebook scope (`utils.RERUN_CACHE = True`) to force every
# subsequent cached_call_llm call to run the model live, bypassing the JSON
# cache. Per-call override is also available via cached_call_llm(..., rerun=True).
RERUN_CACHE: bool = False


def load_model():
    """Load Phi-3.5 mini at 4-bit on GPU (or fp32 on CPU as a fallback).

    Returns (model, tokenizer). Slow on first call (~30 s with cached
    weights, ~1-2 min if downloading); cheap on subsequent calls in the
    same Colab session because HuggingFace caches to /root/.cache.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  No GPU detected. Phi-3.5 will run on CPU and be very slow.")
        print("   In Colab: Runtime → Change runtime type → T4 GPU.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # trust_remote_code=False (default): use transformers' native Phi3 impl
    # rather than the model repo's modeling_phi3.py, which calls a removed
    # DynamicCache.from_legacy_cache method on recent transformers versions.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if device == "cuda" else None,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.eval()
    return model, tokenizer


def call_llm(prompt: str, *, model, tokenizer, system: str | None = None,
             max_new_tokens: int = 1024, response_schema=None):
    """Run a generation against the loaded model.

    If `response_schema` is provided (a Pydantic class), returns a parsed
    Python object via constrained generation through `outlines`. Otherwise
    returns text.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Render chat template to a string, then tokenize ourselves. Avoids the
    # transformers footgun where apply_chat_template can return a
    # BatchEncoding (dict-like) instead of a bare tensor.
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if response_schema is not None:
        return _generate_schema(
            prompt_text, model=model, tokenizer=tokenizer,
            response_schema=response_schema, max_new_tokens=max_new_tokens,
        )
    return _generate_text(
        prompt_text, model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )


def cached_call_llm(label: str, prompt: str | None = None, *,
                    model=None, tokenizer=None,
                    system: str | None = None,
                    max_new_tokens: int = 1024,
                    response_schema=None,
                    chars_per_second: int = 200,
                    display_width: int = 88,
                    rerun: bool | None = None) -> str:
    """Replay a cached model output for `label`, with simulated streaming.

    Reads content/cached_outputs.json. If the label is present and
    `rerun` is False, the cached text is hard-wrapped to `display_width`
    columns and streamed character-by-character at `chars_per_second`.

    To force live regeneration:
      - per call: pass `rerun=True`
      - notebook-wide: set `utils.RERUN_CACHE = True` once at the top

    Falls through to a live call_llm if the label is missing OR rerun is
    in effect, provided `prompt` plus `model` and `tokenizer` are given.

    Returns the full text (unwrapped, cached or freshly generated) so
    callers can further process the result.
    """
    if rerun is None:
        rerun = RERUN_CACHE
    cache = _load_output_cache()
    if not rerun and label in cache:
        text = cache[label]
        wrapped = _wrap_for_display(text, width=display_width)
        delay = 1.0 / max(chars_per_second, 1)
        for ch in wrapped:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return text

    # Cache miss: live call.
    if prompt is None or model is None or tokenizer is None:
        raise RuntimeError(
            f"cached_call_llm: no entry for label {label!r} in "
            f"content/cached_outputs.json, and a live fallback "
            f"requires `prompt`, `model`, and `tokenizer`."
        )
    print(f"[cache miss for {label!r}; running model live]")
    return call_llm(
        prompt, model=model, tokenizer=tokenizer,
        system=system, max_new_tokens=max_new_tokens,
        response_schema=response_schema,
    )


def save_to_cache(label: str, text: str) -> None:
    """Write a label/text pair into content/cached_outputs.json. Useful at
    rehearsal time: run the model once, then save its output so the talk
    can replay it deterministically."""
    cache = _load_output_cache()
    cache[label] = text
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(_json.dumps(cache, indent=2, ensure_ascii=False))
    # Force a reload on next read.
    global _OUTPUT_CACHE
    _OUTPUT_CACHE = None


def _wrap_for_display(text: str, width: int = 88) -> str:
    """Hard-wrap each line to `width` columns, preserving blank lines and
    not joining lines together. Numbered lists and multi-paragraph blocks
    survive intact; only over-wide lines actually get broken.
    """
    import textwrap
    out_lines = []
    for line in text.split("\n"):
        if line.strip() == "":
            out_lines.append("")
            continue
        # Preserve any leading whitespace (numbered-list indents etc.) when
        # wrapping; subsequent wrapped lines align with the first content
        # character.
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]
        wrapped = textwrap.fill(
            stripped,
            width=max(20, width - len(leading)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        out_lines.append(
            "\n".join((leading + w) for w in wrapped.split("\n"))
        )
    return "\n".join(out_lines)


def _load_output_cache() -> dict:
    global _OUTPUT_CACHE
    if _OUTPUT_CACHE is None:
        try:
            _OUTPUT_CACHE = _json.loads(_CACHE_PATH.read_text())
        except FileNotFoundError:
            _OUTPUT_CACHE = {}
    return _OUTPUT_CACHE


def _free_cuda_memory():
    """Clear CUDA cache + run gc."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_text(prompt_text, *, model, tokenizer, max_new_tokens):
    _free_cuda_memory()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    del inputs, output_ids
    _free_cuda_memory()
    return response.strip()


def _generate_schema(prompt_text, *, model, tokenizer, response_schema,
                     max_new_tokens):
    """Constrained generation via outlines. Guarantees valid JSON matching
    the schema, parsed into a Python dict."""
    import outlines

    key = (id(model), id(tokenizer))
    if key not in _OUTLINES_CACHE:
        _OUTLINES_CACHE[key] = outlines.from_transformers(model, tokenizer)
    outlines_model = _OUTLINES_CACHE[key]

    result = outlines_model(
        prompt_text, response_schema, max_new_tokens=max_new_tokens,
    )
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, str):
        return _json.loads(result)
    return result
