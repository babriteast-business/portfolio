---
title: DPO explained, with a TRL training script
date: 2026-05-18
summary: How Direct Preference Optimisation replaces the reward model and PPO loop, what its loss does, and a minimal TRL DPOTrainer and LoRA script to run
tags: Fine-tuning, LLMs
---

Supervised fine-tuning teaches a model what a good answer looks like. It doesn't teach it which of two plausible answers people prefer. Preference optimisation does, and Direct Preference Optimisation (DPO) is the method I reach for first, because it turns the problem into something close to ordinary supervised training. This post covers why it works, what the loss is doing, and a complete script using Hugging Face TRL.

## RLHF the long way: reward model plus PPO

The classic RLHF pipeline has three stages:

1. **SFT.** Fine-tune the base model on demonstrations.
2. **Reward model.** Collect pairs of responses to the same prompt, where a human marked one as better. Train a separate model to score responses so that preferred ones score higher, usually with a Bradley-Terry objective.
3. **RL with PPO.** Generate responses from the policy, score them with the reward model, and update the policy to increase reward. A KL penalty keeps it close to the SFT model so it doesn't drift into nonsense that happens to fool the reward model.

It works, but it's heavy. During PPO you have the policy, a frozen reference copy, the reward model and usually a value model, plus generation inside the training loop. There are many hyperparameters, and training can be unstable.

## The DPO idea

Rafailov et al. (2023) noticed that the KL-constrained reward objective has a closed-form optimal policy. Rearranged, it says the reward is a scaled log-ratio between the optimal policy and the reference policy, plus a term that depends only on the prompt. Substitute that into the Bradley-Terry preference model and the prompt-only term cancels. What remains is a loss written purely in terms of the policy being trained and a frozen reference model. No reward model and no sampling loop are needed, just a classification-style loss over preference pairs.

The paper's title sums it up: your language model is secretly a reward model.

## The loss, intuitively and formally

For a prompt `x`, a preferred response `y_w` and a rejected response `y_l`, define the implicit reward of a response as how much more likely the policy makes it compared with the reference:

```text
r(x, y) = beta * log( pi_theta(y | x) / pi_ref(y | x) )
```

The DPO loss is:

```text
L_DPO = -E[ log sigmoid( r(x, y_w) - r(x, y_l) ) ]
```

In words: push up the policy's probability of the chosen response relative to the reference, push down the rejected one, and care about the gap between them, not the absolute values. The sigmoid also gives a useful weighting. Pairs the model already ranks correctly by a wide margin contribute little gradient. Pairs it gets wrong contribute the most.

One check you can do in your head: at the start of training, the policy equals the reference, so both implicit rewards are zero and the loss is `-log sigmoid(0) = log 2`, about 0.693. If your first logged loss isn't close to that, something is wrong with the setup. With LoRA this holds exactly, because PEFT initialises the adapter so that it starts as a no-op.

## What beta does

`beta` sets how strongly the policy is tied to the reference. The TRL docs put it this way: a higher beta means less deviation from the reference model. A low beta lets the model move further to satisfy the preferences, which can help with clean data but makes it easier to overfit or drift. TRL's default is 0.1, and it's a sensible place to start. Change it only when your evals give you a reason.

## Preference data format

Each example needs a prompt, a chosen response and a rejected response. TRL accepts plain text or conversational messages, and applies the chat template automatically for the conversational form. An explicit prompt is recommended:

```python
example = {
    "prompt": [{"role": "user", "content": "What colour is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

The pair should differ in the quality you care about, such as correctness, tone, refusal behaviour or format. If chosen answers are also consistently longer, the model will learn to be longer. Check that before training.

## A minimal DPOTrainer + LoRA script

This targets TRL 1.17 with Transformers 5 and PEFT. A few things changed from older tutorials. The tokenizer goes in `processing_class`, not `tokenizer`. `max_prompt_length` is no longer a `DPOConfig` field, so `max_length` covers the whole sequence. Warmup is set with `warmup_steps`, which accepts a ratio when given a float below 1. I checked these against the installed library and ran the script end to end on a tiny local model on CPU, to confirm the arguments and data flow. That run says nothing about results quality.

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"  # swap in your own SFT checkpoint
OUTPUT_DIR = "qwen3-0.6b-dpo-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

# Conversational preference data: prompt / chosen / rejected
dataset = load_dataset("trl-lib/ultrafeedback_binarized")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2_000))
eval_dataset = dataset["test"].select(range(200))

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,
    learning_rate=1e-5,  # TRL docs suggest about 1e-5 for adapters
    lr_scheduler_type="cosine",
    warmup_steps=0.1,  # a float below 1 is read as a ratio of total steps
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=1024,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # with a LoRA adapter, the frozen base acts as the reference
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(trainer.evaluate())
```

Some notes on the choices:

- **No separate reference model.** With `ref_model=None` and a PEFT config, TRL computes reference log-probabilities from the same weights with the adapter switched off, so you only hold one copy of the model in memory.
- **Load the model yourself.** If you pass a model name as a string instead, TRL loads it in float32 unless you set `model_init_kwargs={"dtype": torch.bfloat16}`. Loading it yourself makes the dtype explicit.
- **Effective batch size.** 2 per device x 8 accumulation steps gives 16, so 2,000 examples is 125 optimiser steps for one epoch.
- **Start from an SFT model.** DPO refines behaviour the model can already produce. Running it on a raw base model rarely goes well.
- **For QLoRA**, pass a `BitsAndBytesConfig` through the trainer's `quantization_config` argument, together with `peft_config`.

## Evaluating the result

TRL logs four reward metrics, and they are worth understanding:

- `rewards/chosen` and `rewards/rejected`: the mean implicit reward for each side.
- `rewards/margins`: the mean gap between them.
- `rewards/accuracies`: the share of pairs where chosen beats rejected.

On the held-out split, you want margins and accuracy to rise. If training accuracy reaches 1.0 quickly while eval accuracy stalls, you are overfitting, so lower the learning rate, raise beta or add data. These metrics only tell you that the model fits the preference data, though. They don't tell you it got better. For that:

- Generate answers to held-out prompts from the SFT and DPO models, and compare them with human review or an LLM judge, using a fixed rubric and randomised order.
- Run a few general benchmarks before and after to catch regressions.
- Track response length, because a win-rate gain that comes only from longer answers isn't a real gain.

## Common pitfalls

- **Chat template mismatch.** Train with the same template you'll use at inference.
- **Truncation.** Pairs longer than `max_length` get truncated, and fully truncated examples are dropped. Check your length distribution.
- **Length bias.** If chosen responses are systematically longer, the model learns verbosity.
- **Falling chosen likelihood.** The loss only cares about the gap, so the log-probability of chosen responses can fall too, as long as rejected ones fall faster. Watch the `logps/chosen` metric, and consider variants designed to counter this.
- **Noisy or near-identical pairs.** If annotators barely agree, or the pair differs only cosmetically, the signal is mostly noise.
- **Too many epochs.** One epoch is often enough for preference data. More passes tend to overfit.

## Variants worth knowing

- **IPO** replaces the log-sigmoid with a squared loss to reduce overfitting to the preference data. In TRL it's `loss_type="ipo"` on `DPOConfig`, where beta plays the role of the paper's regularisation parameter.
- **KTO** works with unpaired data, where each completion is simply labelled desirable or undesirable. That's useful when you have thumbs-up and thumbs-down logs rather than pairs. TRL provides `KTOTrainer`.
- **ORPO** drops the reference model and folds preference learning into SFT with an odds-ratio penalty, so it can run as one stage.
- **SimPO** is also reference-free. It uses the length-normalised average log-probability as the reward and adds a target margin. In TRL 1.17, both ORPO and SimPO (via the CPO trainer with `loss_type="simpo"`) live under `trl.experimental`.

For reasoning tasks with checkable answers, such as maths or code with tests, people increasingly use online RL instead. **GRPO**, introduced in the DeepSeekMath paper, samples a group of answers per prompt, scores them with a reward function, and uses each answer's reward relative to the group as its advantage, so there's no value model. TRL's `GRPOTrainer` implements it. DPO remains the simpler choice when what you have is a fixed set of human or AI preference pairs.

## Sources

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Hugging Face TRL docs: DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Hugging Face TRL docs: Paper index](https://huggingface.co/docs/trl/paper_index)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO, Azar et al., 2023)](https://arxiv.org/abs/2310.12036)
- [KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)](https://arxiv.org/abs/2402.01306)
- [ORPO: Monolithic Preference Optimization without Reference Model (Hong et al., 2024)](https://arxiv.org/abs/2403.07691)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward (Meng et al., 2024)](https://arxiv.org/abs/2405.14734)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO, Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
