# Smoke test results (2026-11-10)

A smoke test is a quick run that checks each part works at all before we rely on it.

Machine: M4 Pro, 24 GB unified memory, macOS 15.5.
Timings were taken while the laptop was in other use. Treat every speed number as provisional.

## Versions

| Component | Version |
|---|---|
| Python | 3.12.5 (uv 0.11.26) |
| torch | 2.15.0 (MPS available) |
| transformers / peft / trl / accelerate | 5.22.2 / 0.21.2 / 1.17.1 / 1.15.0 |
| llama-cpp-python | 0.3.36, Metal build |
| llama.cpp (convert + quantize + runtime) | commit `0c1e570` (vendored by llama-cpp-python 0.3.36) |
| sqlglot | 30.30.0 |
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct, HF commit `2e1fd397ee46e1388853d2af2c993145b0f1098a` |

## 1. HF on MPS (`smoke_hf.py`)

- fp16, greedy, 149-token prompt. About 58 tok/s decode.
- Answerable question gives plausible SQL.
- "Population of Scarborough?" gives SQL against a made-up `census_2021` table. The base model does not abstain.

## 2. LoRA on MPS (`smoke_lora.py`)

- 10 examples, 20 optimizer steps, r=16 on all linear layers (placeholder knobs). 18.5M trainable params (1.18%).
- Sequence length about 1,650 tokens. The dummy schema is padded to the 1,500-token budget so timing is realistic.
- Label masking checked by decoding a batch. 1,627 of 1,646 tokens masked. Labels are exactly the SQL plus `<|im_end|>`.
- Loss 0.72 (first 3 avg) to 0.03 (last 3 avg). No NaN or inf.
- No MPS op fallbacks. `PYTORCH_ENABLE_MPS_FALLBACK` was unset and nothing raised.
- First attempt (fp32, bs=2, no checkpointing) hit 27 GB and OOMed. It also pushed the machine into swap.
- Working setup: fp32, bs=1, grad accum 2, gradient checkpointing. Peak MPS memory 12.3 GB. 10.5 s per optimizer step (2 examples).

Re-timed on an idle machine (85% memory free), fp32 vs bf16 weights:

| Weights | sec/step | Peak MPS memory | Loss first 3 to last 3 | 2,500 x 2 ep | 2,500 x 3 ep | 3,000 x 3 ep |
|---|---|---|---|---|---|---|
| fp32 | 10.4 | 12.3 GB | 0.719 to 0.028 | ~7.2 h | ~10.9 h | ~13.0 h |
| bf16 | 10.1 | 8.4 GB | 0.715 to 0.028 | ~7.0 h | ~10.5 h | ~12.6 h |

- The idle fp32 time matches the busy-machine run, so contention was not the bottleneck.
- bf16 is stable (no NaN, loss tracks fp32 within 0.01) and saves 4 GB, but it is not faster. MPS is compute-bound here.
- Every config is well over the 3 h per-seed limit. Recommendation: train on Colab CUDA.

### Colab T4 (2026-11-14, `notebooks/colab_timing.ipynb`)

Same script and settings. fp32 weights with fp16 mixed precision, since the T4 has no bf16.
Colab ran Python 3.13 with its own CUDA torch. HF stack pinned to the Mac versions.

| Device | sec/step | Peak memory | Loss first 3 to last 3 | 2,500 x 2 ep | 2,500 x 3 ep | 3,000 x 3 ep |
|---|---|---|---|---|---|---|
| Tesla T4 (fp16 AMP) | 2.69 | 6.8 GB of 15 GB | 0.711 to 0.024 | ~1.9 h | ~2.8 h | ~3.4 h |

- About 3.9x faster than the M4 Pro. No NaN. Loss tracks the Mac run closely.
- Label masking checked again on CUDA. Only the SQL plus `<|im_end|>` carry loss.
- Memory headroom means gradient checkpointing could be turned off for more speed. Test on real data first.
- The estimate covers training steps only. Per-epoch val generation adds time on top.
- Decision: train on free Colab, with a checkpoint every epoch saved to Drive. Eval and serving stay on the Mac.

## 3. GGUF via llama-cpp-python (`smoke_gguf.py`)

| Variant | File size |
|---|---|
| F16 | 2,950 MiB |
| Q4_K_M | 940 MiB |

- HF tokenizer and llama.cpp tokenizer give identical token ids for the rendered prompt.
- `ABSTAIN` tokenizes as `AB` + `ST` + `AIN`. First token id 1867. No token merge across the prompt boundary.
- P(`AB`) at step 1 is readable from the full softmax, not only the top 5. Our log-softmax matches llama-cpp's own top-1 logprob.

| Question | Variant | P(`AB`) at step 1 | Greedy output |
|---|---|---|---|
| Union delays, March 2025 | F16 | 0.042 | SQL |
| Union delays, March 2025 | Q4_K_M | 0.059 | same SQL |
| Population of Scarborough | F16 | 0.080 | SQL on invented `census_2021` |
| Population of Scarborough | Q4_K_M | 0.080 | SQL on a different invented table |

- Decode with `logits_all=True` was about 11 tok/s. That mode computes full-vocab logits at every prompt position, so it is not a fair speed number.
