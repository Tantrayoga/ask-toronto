"""Smoke test: load the base model on MPS and generate one SQL query."""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ask_toronto.prompt import SMOKE_SCHEMA, SMOKE_WINDOW, render_prompt

MODEL_DIR = "models/qwen2.5-coder-1.5b-instruct"
QUESTIONS = [
    "How many subway delays were there at Union station in March 2025?",
    "What's the population of Scarborough?",
]


def main() -> None:
    assert torch.backends.mps.is_available(), "MPS not available"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # fp16 matches the final merged artifact. fp32 would double memory for no gain at inference.
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).to("mps").eval()

    for q in QUESTIONS:
        prompt = render_prompt(tok, q, SMOKE_SCHEMA, SMOKE_WINDOW)
        ids = tok(prompt, return_tensors="pt").to("mps")
        # Warm-up run so Metal kernel compilation doesn't pollute the timing.
        model.generate(**ids, max_new_tokens=4, do_sample=False)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=128, do_sample=False)
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        new = out[0, ids["input_ids"].shape[1]:]
        print(f"\nQ: {q}\nprompt tokens: {ids['input_ids'].shape[1]}")
        print(f"A: {tok.decode(new, skip_special_tokens=True)}")
        print(f"{len(new)} tokens in {dt:.2f}s -> {len(new) / dt:.1f} tok/s")


if __name__ == "__main__":
    main()
