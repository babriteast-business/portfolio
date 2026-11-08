---
title: Fine-tuning a 14B model on one GPU with QLoRA
date: 2026-11-02
summary: How I took Qwen3-14B from open weights to a served API on a single NVIDIA A40, and the order of steps that kept it sane.
tags: Fine-tuning, LLMs, MLOps
---

Full fine-tuning a 14-billion-parameter model needs far more GPU memory than one card has. QLoRA changes that maths: the base model is loaded in 4-bit, its weights stay frozen, and only small low-rank adapter matrices are trained. That's how I fine-tuned **Qwen3-14B** on a single **NVIDIA A40** (48 GB) with **Unsloth**.

## The order of steps

1. **Start from the right base.** An instruct-tuned open-weights model is a better starting point than a raw base model when the goal is a helpful assistant in one domain.
2. **Supervised fine-tuning (SFT) first.** This teaches the format and the domain behaviour from instruction and response pairs.
3. **Preference optimisation (DPO) second.** With the behaviour in place, DPO nudges the model towards the answers people prefer, using chosen and rejected pairs, without training a separate reward model.
4. **Guardrails and benchmarks before anyone uses it.** A guardrail layer checks outputs, and a benchmarking suite tells me whether a new checkpoint is actually better.
5. **Serve it behind an OpenAI-compatible API.** A FastAPI gateway means any client that already talks to OpenAI can talk to my model, and Docker makes it run the same everywhere.

## What mattered most

- **The chat template.** Training data has to be formatted exactly the way the model will be prompted at inference time. A mismatch here quietly ruins results.
- **A held-out evaluation set from day one.** Loss going down only tells you the model fits the training data. You need questions it has never seen.
- **Watching for forgetting.** LoRA keeps the base weights frozen, which limits the damage, but I still check general ability before and after.
- **Keeping data small and clean.** A few thousand consistent, deduplicated examples beat a large noisy dump.

## When I wouldn't fine-tune

If the problem is *missing knowledge*, retrieval (RAG) is usually the better tool. Fine-tuning shines when you need a consistent behaviour, format or style, or when a smaller tuned model can replace calls to a large one.
