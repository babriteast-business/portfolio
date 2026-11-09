---
title: KV cache memory math, worked through
date: 2026-04-13
summary: What the KV cache stores, how to size it from a model's config.json, and how GQA, FP8, PagedAttention and prefix caching change the numbers
tags: LLMs, Inference
---

When people plan GPU capacity for serving an LLM, they usually start with the weights. That is the easy part. The number that decides how many users you can serve at once, and how long their prompts can be, is the KV cache. This post works through the arithmetic with a real model's published config, so you can do the same for whatever you deploy.

## What the KV cache stores and why

A decoder-only transformer generates one token at a time. At every step, each attention layer compares the new token's query with the keys of every earlier token, then takes a weighted mix of their values. The keys and values for earlier tokens don't change, so recomputing them on every step would be wasted work.

The KV cache keeps them instead. During prefill, the model processes the whole prompt and writes a key and a value vector for every token, in every layer, for every KV head. During decode, each new token adds one more entry per layer. The cache grows linearly with the number of tokens in the sequence, and it has to stay in GPU memory for as long as the request is running.

## The formula

```text
KV cache bytes = 2 x layers x kv_heads x head_dim x bytes_per_value x tokens
```

- **2**: one tensor for keys and one for values.
- **layers**: `num_hidden_layers` in `config.json`.
- **kv_heads**: `num_key_value_heads`, not `num_attention_heads`. This difference matters a lot, as we'll see.
- **head_dim**: the size of each head's vector. Some configs list it as `head_dim`; otherwise it is usually `hidden_size / num_attention_heads`.
- **bytes_per_value**: 2 for BF16 or FP16, 1 for FP8.
- **tokens**: prompt plus generated tokens, summed over every sequence in the batch.

## A worked example with Qwen3-8B

I'll use Qwen3-8B because its `config.json` on Hugging Face lists everything we need:

| Field | Value |
|---|---|
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `max_position_embeddings` | 40,960 |
| `torch_dtype` | bfloat16 |

Per token, in BF16:

```text
2 x 36 x 8 x 128 x 2 bytes = 147,456 bytes = 144 KiB
```

That looks small until you multiply it out:

| Tokens (one sequence) | KV cache (BF16) |
|---|---|
| 4,096 | 0.5625 GiB |
| 32,768 | 4.5 GiB |
| 40,960 | 5.625 GiB |

For comparison, an 8-billion-parameter model needs roughly 16 GB just for its BF16 weights (8 billion x 2 bytes). One long-context request adds over a quarter of that again.

## How GQA and MQA shrink it

In classic multi-head attention (MHA), every query head has its own key and value head. Multi-query attention (MQA) shares a single key and value head across all query heads. Grouped-query attention (GQA) sits in between: query heads are split into groups, and each group shares one KV head.

Qwen3-8B uses GQA with 32 query heads and 8 KV heads, so four query heads share each KV head. Keeping every other part of the shape the same, here is what the choice does at 32,768 tokens:

| Attention type | KV heads | KV cache at 32,768 tokens |
|---|---|---|
| MHA | 32 | 18 GiB |
| GQA (as shipped) | 8 | 4.5 GiB |
| MQA | 1 | 0.5625 GiB |

The saving is exactly the ratio `num_attention_heads / num_key_value_heads`. This is why you should always read `num_key_value_heads` from the config. Using the query head count would overestimate this model's cache by a factor of four.

## Batch size and context length

The cache is linear in both. Doubling the batch doubles it, and doubling the context doubles it. With Qwen3-8B in BF16 at 32,768 tokens per sequence:

| Concurrent sequences | KV cache |
|---|---|
| 1 | 4.5 GiB |
| 8 | 36 GiB |
| 16 | 72 GiB |

So on an 80 GB card, after about 16 GB of weights and some headroom for activations, sixteen full-length conversations at once don't fit. It's often more useful to turn the formula around and ask how many tokens fit in the memory you have left. With 20 GiB free for the cache, this model holds 145,635 tokens in BF16. You can spend that on four long conversations or dozens of short ones. The serving engine shares it out between them.

## KV cache quantisation

Storing keys and values in FP8 instead of BF16 halves `bytes_per_value`, so it halves the cache. The 16-sequence case drops from 72 GiB to 36 GiB, and the 20 GiB budget now holds 291,271 tokens.

In vLLM this is the `kv_cache_dtype` setting, with values such as `fp8`, `fp8_e4m3` and `fp8_e5m2`. The vLLM docs note that without calibration, the scaling factors default to 1.0. For the best accuracy they recommend calibrating scales on a dataset with llm-compressor. Either way, treat it as a quality trade-off and run your own evals before and after, particularly on long-context tasks.

## PagedAttention and vLLM

The formula tells you how much memory the tokens need. Early serving systems often wasted far more than that, because they reserved one contiguous slab per request for its maximum possible length. A request that reserves 32,768 tokens but only uses 1,000 leaves about 97% of its reservation idle.

PagedAttention, the idea behind vLLM, borrows from virtual memory in operating systems. The cache is split into fixed-size blocks, and each sequence keeps a block table that maps its logical positions to physical blocks anywhere in GPU memory. Blocks are allocated only as tokens arrive, so waste is limited to the last, partly filled block of each sequence. With a 16-token block, one Qwen3-8B block is 2.25 MiB in BF16. The paper reports that vLLM improves throughput by 2 to 4 times at the same level of latency, compared with the systems it was tested against.

## Prefix caching

Blocks also make sharing possible. If many requests start with the same system prompt or document, their first blocks hold identical keys and values. Automatic prefix caching in vLLM (`enable_prefix_caching=True`) reuses those blocks instead of recomputing them.

Be clear about what it buys you. The vLLM docs point out that it only shortens prefill. It does nothing for generation time, and nothing when requests don't share a prefix. Hosted APIs offer the same idea as prompt caching, and the same rule applies: put the stable part of the prompt first.

## A small calculator

```python
def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_value=2, batch=1):
    """Size of the KV cache in bytes. The leading 2 is one K and one V tensor."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value * tokens * batch


def gib(n_bytes):
    return n_bytes / 1024**3


# Qwen3-8B, from its config.json
QWEN3_8B = dict(layers=36, kv_heads=8, head_dim=128)

per_token = kv_cache_bytes(**QWEN3_8B, tokens=1)
print(f"Per token (BF16): {per_token:,} bytes = {per_token / 1024:.0f} KiB")

for tokens in (4_096, 32_768, 40_960):
    print(f"{tokens:>6} tokens, batch 1: {gib(kv_cache_bytes(**QWEN3_8B, tokens=tokens)):.3f} GiB")

for batch in (1, 8, 16):
    size = kv_cache_bytes(**QWEN3_8B, tokens=32_768, batch=batch)
    print(f"32,768 tokens, batch {batch:>2}: {gib(size):.1f} GiB")

# Same model shape with different numbers of KV heads
for name, kv_heads in (("MHA (32 KV heads)", 32), ("GQA (8 KV heads)", 8), ("MQA (1 KV head)", 1)):
    size = kv_cache_bytes(36, kv_heads, 128, tokens=32_768)
    print(f"{name:<18} 32,768 tokens: {gib(size):.3f} GiB")

# FP8 KV cache: one byte per value
fp8 = kv_cache_bytes(**QWEN3_8B, tokens=32_768, batch=16, bytes_per_value=1)
print(f"FP8, 32,768 tokens, batch 16: {gib(fp8):.1f} GiB")
```

Running it prints:

```text
Per token (BF16): 147,456 bytes = 144 KiB
  4096 tokens, batch 1: 0.562 GiB
 32768 tokens, batch 1: 4.500 GiB
 40960 tokens, batch 1: 5.625 GiB
32,768 tokens, batch  1: 4.5 GiB
32,768 tokens, batch  8: 36.0 GiB
32,768 tokens, batch 16: 72.0 GiB
MHA (32 KV heads)  32,768 tokens: 18.000 GiB
GQA (8 KV heads)   32,768 tokens: 4.500 GiB
MQA (1 KV head)    32,768 tokens: 0.562 GiB
FP8, 32,768 tokens, batch 16: 36.0 GiB
```

To use it for another model, open its `config.json` and plug in `num_hidden_layers`, `num_key_value_heads` and `head_dim`. Check the architecture first, though. Models with sliding-window layers, or attention schemes that compress the cache, need a modified formula. For Qwen3-8B the config has `use_sliding_window` set to false, so the plain formula holds.

## Sources

- [Qwen3-8B config.json, Hugging Face](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)](https://huggingface.co/papers/2309.06180)
- [vLLM docs: Automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM docs: Quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)
