"""Smoke test: HF -> F16 GGUF -> Q4_K_M, then read step-1 logprobs with llama-cpp-python.

Convert, quantize and eval all use the llama.cpp commit vendored by
llama-cpp-python, so the GGUF format and the runtime cannot drift apart.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from llama_cpp import Llama
from transformers import AutoTokenizer

from ask_toronto.prompt import ABSTAIN, SMOKE_SCHEMA, SMOKE_WINDOW, render_prompt

HF_DIR = Path("models/qwen2.5-coder-1.5b-instruct")
OUT = Path("models/gguf")
LLAMA_CPP = Path("vendor/sdist/llama_cpp_python-0.3.36/vendor/llama.cpp")
QUANTIZE = LLAMA_CPP / "build/bin/llama-quantize"
QUESTIONS = [
    "How many subway delays were there at Union station in March 2025?",
    "What's the population of Scarborough?",
]


def convert_and_quantize() -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    f16 = OUT / "base-f16.gguf"
    q4 = OUT / "base-q4_k_m.gguf"
    if not f16.exists():
        # Use the vendored gguf-py so the writer matches the runtime's reader.
        env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py")}
        subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(HF_DIR),
                        "--outtype", "f16", "--outfile", str(f16)], check=True, env=env)
    if not q4.exists():
        subprocess.run([str(QUANTIZE), str(f16), str(q4), "Q4_K_M"], check=True)
    return {"F16": f16, "Q4_K_M": q4}


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main() -> None:
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    # The first completion token of ABSTAIN, as it appears right after "assistant\n".
    probe = render_prompt(tok, "x", SMOKE_SCHEMA, SMOKE_WINDOW)
    joint = tok(probe + ABSTAIN)["input_ids"]
    alone = tok(probe)["input_ids"]
    assert joint[: len(alone)] == alone, "prompt/target boundary merges tokens"
    abstain_ids = joint[len(alone):]
    first_id = abstain_ids[0]
    print("ABSTAIN tokens:", [tok.decode([i]) for i in abstain_ids], "first id:", first_id)

    for name, path in convert_and_quantize().items():
        print(f"\n=== {name}  {path.stat().st_size / 2**20:.0f} MiB")
        # logits_all keeps logits for every position, needed for logprobs.
        # n_ctx 2048 keeps that buffer at ~1.2 GB (n_ctx x vocab x 4 bytes).
        llm = Llama(str(path), n_gpu_layers=-1, n_ctx=2048, logits_all=True, seed=0, verbose=False)
        for q in QUESTIONS:
            prompt = render_prompt(tok, q, SMOKE_SCHEMA, SMOKE_WINDOW)
            ids_hf = tok(prompt)["input_ids"]
            ids_ll = llm.tokenize(prompt.encode(), add_bos=False, special=True)
            assert ids_hf == ids_ll, "HF and llama.cpp tokenize the prompt differently"

            llm.reset()
            t0 = time.perf_counter()
            # Pass token ids so llama.cpp cannot re-tokenize or add a BOS.
            out = llm.create_completion(ids_hf, max_tokens=128, temperature=0.0, logprobs=5)
            dt = time.perf_counter() - t0
            choice = out["choices"][0]
            n_gen = out["usage"]["completion_tokens"]

            lp = log_softmax(llm.scores[len(ids_hf) - 1])
            top5 = choice["logprobs"]["top_logprobs"][0]
            print(f"\nQ: {q}")
            print(f"A: {choice['text']!r}  [{n_gen} tok, {n_gen / dt:.1f} tok/s incl. prefill]")
            print("step-1 top-5 (llama-cpp):", {k: round(v, 3) for k, v in top5.items()})
            print(f"P(first ABSTAIN token {tok.decode([first_id])!r}) = {np.exp(lp[first_id]):.3e}"
                  f"  (log {lp[first_id]:.2f}, rank {int((lp > lp[first_id]).sum()) + 1})")
            # Cross-check: our softmax must agree with llama-cpp's own top-1 logprob.
            top_tok = int(lp.argmax())
            print(f"argmax {tok.decode([top_tok])!r} logp {lp[top_tok]:.4f}")
        del llm


if __name__ == "__main__":
    main()
