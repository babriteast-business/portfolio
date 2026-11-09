---
title: Cutting LLM costs with prompt caching and routing
date: 2026-09-21
summary: A practical order of work for lowering LLM API spend: measure per feature, cache the static prefix, route by difficulty, batch offline jobs and check quality
tags: LLMs, Inference
---

LLM bills tend to grow quietly. A feature ships, usage climbs, and one month the invoice is the biggest line in the infrastructure budget. The fixes are well known, but the order matters. If you start optimising before you know where the tokens go, you will spend a week shaving the wrong feature. This is the order I work through.

## Measure first

Every provider returns token usage with each response. Log it, along with the feature or endpoint that made the call, the model, and the user or tenant. Then you can answer the questions that matter:

- Which feature costs the most per day, and per request?
- How much of each request is input, and how much is output?
- How much of the input repeats across requests?

The third question tells you whether caching will help. The second tells you whether output length is the bigger problem. Keep the cost calculation in code, with prices as configuration, because prices and models change. This helper turns an Anthropic usage object into "input-equivalent" tokens using the multipliers in Anthropic's prompt caching docs (5-minute cache writes at 1.25x base input, reads at 0.1x on many models and lower on some newer ones):

```python
def input_cost(usage, price_per_mtok, write_mult=1.25, read_mult=0.1):
    """Input cost of one Anthropic request. Multipliers come from the docs, prices from config."""
    tokens = (
        usage.input_tokens
        + usage.cache_creation_input_tokens * write_mult
        + usage.cache_read_input_tokens * read_mult
    )
    return tokens * price_per_mtok / 1_000_000
```

## How provider prompt caching works

Both Anthropic and OpenAI can reuse the work done on a prompt prefix they have seen recently. The prefix has to match exactly, so even a small change near the start of a prompt means a cache miss.

**Anthropic.** Caching is opt-in. You mark a cache point by adding `"cache_control": {"type": "ephemeral"}` to a content block, and everything up to and including that block is cached. You can set up to four breakpoints. There is also an automatic mode: a single top-level `cache_control` field, which places the breakpoint on the last cacheable block and moves it forward as a conversation grows. The prefix is built in a fixed order, `tools`, then `system`, then `messages`. Changing something at one level invalidates that level and everything after it, so editing a tool definition invalidates the whole cache.

The default lifetime is five minutes, and it is refreshed each time the entry is used. A one-hour option is available with `"ttl": "1h"`, and those writes cost 2x base input instead of 1.25x. Prompts below a minimum length are not cached, with no error. The minimum depends on the model, from 512 to 4,096 tokens in the current table, so check yours. Caches are never shared across organisations, and on the Claude API they are isolated per workspace.

**OpenAI.** Caching is automatic for supported models, with no code change. The guide gives a 1,024-token minimum for its newest models and says the threshold varies with request settings on earlier ones. Cached tokens show up in `usage.input_tokens_details.cached_tokens`. Requests are routed by a hash of the start of the prompt, and the optional `prompt_cache_key` parameter helps requests that share a prefix land on the same cache. How long entries are kept depends on the model and the retention setting, so read the current guide for the model you use. OpenAI describes the saving as cached input being discounted by up to 95%.

## The break-even sum

With Anthropic's 5-minute cache and a model where reads cost 0.1x, sending a prefix twice without caching costs 2.0 units of input. With caching, it costs 1.25 for the first write plus 0.1 for the read, 1.35 units. Caching pays for itself on the first reuse.

With the 1-hour cache at 2x, two calls cost 2.1 units, which is slightly more than not caching. Three calls cost 2.2 units against 3.0. Use the longer TTL when a prefix will be reused across gaps longer than five minutes, not by default.

## Structure prompts for cache hits

Since only an exact prefix can match, the layout of the prompt decides your hit rate:

1. Tool definitions, kept stable and in a fixed order.
2. System instructions.
3. Large reference material: documents, policies, few-shot examples.
4. Conversation history, appended rather than rewritten.
5. The new user message and anything dynamic.

The usual mistakes are a timestamp or request ID at the top of the system prompt, user names written into the instructions, tools listed in a different order each call, and few-shot examples sampled at random per request. Each of these moves a changing value into the prefix and gives you a miss every time. OpenAI's guide also warns that summarising or truncating history can reset reuse. If you have to compact a conversation, do it rarely and in one go.

## A cached request with the Anthropic Python SDK

Here a large handbook sits in the system prompt with a cache breakpoint after it. Only the question changes between calls.

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

with open("product_handbook.md", encoding="utf-8") as f:
    handbook = f.read()  # large, rarely changing reference text

SYSTEM = [
    {
        "type": "text",
        "text": "You answer support questions using only the handbook below. "
                "If the handbook does not cover it, say so.",
    },
    {
        "type": "text",
        "text": handbook,
        "cache_control": {"type": "ephemeral"},  # cache everything up to here
    },
]


def ask(question: str):
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    u = response.usage
    print(
        f"written={u.cache_creation_input_tokens} "
        f"read={u.cache_read_input_tokens} "
        f"uncached={u.input_tokens} output={u.output_tokens}"
    )
    return response.content[0].text


ask("How do I reset my password?")       # first call writes the cache
ask("Can I change my billing date?")     # a call within the TTL should read it
```

On the first call, the handbook should appear in `cache_creation_input_tokens`. On a second call within five minutes, it should appear in `cache_read_input_tokens`. If both calls show writes, something in the prefix is changing, or the prefix is below the model's minimum length. For a one-hour lifetime, use `{"type": "ephemeral", "ttl": "1h"}`.

## Semantic caching, carefully

A semantic cache stores whole responses and returns one when a new query's embedding is close enough to an old one. When it works, it skips the model call entirely. The risks are real, though:

- **Near-duplicates aren't duplicates.** "Cancel my order" and "don't cancel my order" can sit close together in embedding space.
- **Leaking across users.** A cached answer that mentions one customer's data must never reach another. Scope keys by tenant, or only cache answers that contain no user data.
- **Stale answers.** When the source data changes, cached responses don't. Set an expiry and clear entries when the underlying content changes.
- **Threshold tuning.** Too loose gives wrong answers, too tight gives no hits. Tune it against labelled pairs, not by eye.

I'd only use it for high-volume, non-personal, slow-changing questions, such as FAQ-style lookups.

## Route by difficulty

Not every request needs your most capable model. Two patterns help:

- **Routing.** A cheap classifier, rules, or a small model decides which model gets the request. Extraction and classification can go to a small model, and open-ended reasoning to a large one.
- **Cascades.** Try the cheap model first, check the answer with a validator (schema check, tests, a confidence signal or a grader), and escalate only when it fails. FrugalGPT studied this approach.

Both only make sense if you can measure quality per route, which is where evals come in.

## Batch anything that can wait

Offline work, such as nightly classification, backfills, evaluation runs or document enrichment, doesn't need an instant answer. Anthropic's Message Batches API charges 50% of standard prices, with results within 24 hours, and the docs say most batches finish within an hour. The caching and batch discounts stack. Because a batch can take longer than five minutes, Anthropic recommends the one-hour cache for batches that share context. OpenAI's Batch API also gives a 50% discount with a 24-hour completion window.

## Control output length

Output tokens are often priced higher than input tokens (check your provider's price list), and they also set your latency. Set `max_tokens` to what the feature actually needs. Ask for the format you want: a JSON object instead of prose, a label instead of an explanation, three bullet points instead of an essay. Use structured outputs where the provider supports them, and stop sequences where they fit. Watch the logs for truncated responses after you tighten limits.

## Verify quality with evals

Every change above can lower quality without anyone noticing. Before rolling one out, run a fixed evaluation set (real, anonymised requests with known good answers or a grading rubric) against the old and new setups. Compare quality and cost side by side, and keep the set for the next change. Caching changes like reordering a prompt deserve this check too, because moving instructions around can change behaviour.

## Sources

- [Anthropic docs: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic docs: Message Batches](https://platform.claude.com/docs/en/build-with-claude/message-batches)
- [OpenAI docs: Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [OpenAI docs: Batch API](https://developers.openai.com/api/docs/guides/batch)
- [vLLM docs: Automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance (Chen et al., 2023)](https://arxiv.org/abs/2305.05176)
