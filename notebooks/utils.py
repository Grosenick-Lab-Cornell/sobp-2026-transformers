"""Helpers for the SOBP 2026 LLM-Basics talk notebooks.

Imported by both NB1 and NB2 setup cells. Designed for fast iteration:
notebooks pull this file via `git pull` + `%autoreload 2` so changes here
propagate without re-loading the model (which is the slow part).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

_OUTLINES_CACHE: dict = {}


def load_model():
    """Load Phi-3.5 mini at 4-bit on GPU (or fp32 on CPU as a fallback).

    Returns (model, tokenizer). Slow on first call (~30 s with cached
    weights, ~1–2 min if downloading); cheap on subsequent calls in the
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
    # rather than the model repo's modeling_phi3.py — which calls a removed
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


def _free_cuda_memory():
    """Clear CUDA cache + run gc. Cheap (~ms), prevents fragmentation
    accumulating across long-context calls (matters for the §4 needle
    demo which runs the model 3x sequentially on a multi-thousand-token
    prompt)."""
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
    # Free the input tensors and output ids to reduce peak memory before
    # the next call. The decoded string is what we actually return.
    del inputs, output_ids
    _free_cuda_memory()
    return response.strip()


def _generate_schema(prompt_text, *, model, tokenizer, response_schema,
                     max_new_tokens):
    """Constrained generation via outlines. Guarantees valid JSON matching
    the schema, parsed into the corresponding Pydantic instance."""
    import outlines

    key = (id(model), id(tokenizer))
    if key not in _OUTLINES_CACHE:
        _OUTLINES_CACHE[key] = outlines.from_transformers(model, tokenizer)
    outlines_model = _OUTLINES_CACHE[key]

    result = outlines_model(
        prompt_text, response_schema, max_new_tokens=max_new_tokens,
    )
    return result.model_dump() if hasattr(result, "model_dump") else result
