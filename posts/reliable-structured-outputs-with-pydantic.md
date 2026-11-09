---
title: Reliable structured outputs from LLMs with Pydantic
date: 2026-06-01
summary: JSON mode, structured outputs and tool calling compared, plus a provider-agnostic Pydantic v2 validate-and-retry loop that handles refusals and truncation
tags: LLMs, Python
---

Most LLM features I build end in code, not in a chat window. A classifier feeds a router, an extractor fills a database row, an agent step hands arguments to a function. All of these need the model's answer as data with a known shape, and the way you ask for that shape decides how often your pipeline falls over.

This post walks through the options and then builds a small, provider-agnostic layer with Pydantic v2 that validates the output, asks the model to repair it when it fails, and treats refusals and truncated output as separate cases.

## Why parsing free text breaks

The first version of most LLM integrations says "reply in JSON" in the prompt and calls `json.loads` on the result. It works in the demo and fails in production in predictable ways:

- The model wraps the JSON in a Markdown code fence or adds a friendly sentence before it.
- A field comes back as `"High"` when your code expects `"high"`, or as a string when you wanted a number.
- A required field is missing, or an extra one appears.
- The output stops halfway through because it hit the token limit.
- The model declines the request, and you get an apology where the JSON should be.

Regex clean-up handles the first case and none of the others. You need a schema that the model is held to, and a validator in your own code that checks it anyway.

## JSON mode, structured outputs and tool calling

There are three mechanisms, and they give different guarantees.

**JSON mode** only guarantees syntactically valid JSON. OpenAI's cookbook puts the difference plainly: "Unlike JSON mode, Structured Outputs constrains the shape of the JSON, not only its validity." With JSON mode you still get missing fields and wrong types. OpenAI's guide recommends using Structured Outputs instead of JSON mode wherever the model supports it.

**Schema-constrained structured outputs** decode against a JSON Schema you supply.

- On OpenAI's Responses API, you pass the schema under `text.format` with `type: "json_schema"` and `strict: true`. Chat Completions uses `response_format` the same way. The Python SDK's `client.responses.parse(..., text_format=MyModel)` accepts a Pydantic model and returns the parsed object.
- On the Claude API, the current parameter is `output_config.format` with `type: "json_schema"`, and no beta header is needed. The older `output_format` parameter is deprecated. The Python SDK's `client.messages.parse(..., output_format=MyModel)` takes a Pydantic model and returns `parsed_output`.

**Tool calling** (function calling) is the third route. You describe a function with an input schema and the model replies with a tool call whose arguments match it. Both providers support a strict mode for tool schemas (`strict: true` on the tool definition). Anthropic's docs separate the two features neatly: JSON outputs control what Claude says, and strict tool use validates how Claude calls your functions. My rule: if the structured data is the final answer, use structured outputs. If the model is choosing between actions, use tools.

Constrained decoding is not the end of the story. Both providers document cases where the output may not match your schema: refusals, and responses cut off by the token limit. Your own validation stays in place either way.

## Define the shape once with Pydantic v2

A Pydantic model gives you three things from one definition: the JSON Schema to send to the provider, validation of what comes back, and a typed Python object for the rest of your code.

```python
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SupportTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(description="One sentence, under 200 characters")
    customer_email: Optional[str] = Field(
        description="Email address if the message contains one, otherwise null"
    )

    @field_validator("category", "priority", mode="before")
    @classmethod
    def lowercase(cls, v):
        return v.lower().strip() if isinstance(v, str) else v

    @field_validator("summary")
    @classmethod
    def short_summary(cls, v: str) -> str:
        if len(v) > 200:
            raise ValueError("summary must be under 200 characters")
        return v
```

A few choices here are deliberate:

- `extra="forbid"` makes `model_json_schema()` emit `"additionalProperties": false`. The Claude API says it must be `false` on objects, and OpenAI's strict-mode examples set it too.
- `customer_email` is `Optional[str]` with **no default**. That keeps it in the schema's `required` list as `anyOf: [string, null]`. Strict mode wants every field present, so "optional" becomes "required but nullable".
- `Literal` becomes an `enum`. The `mode="before"` validator lowercases values first, because Anthropic's docs note that a model can return an enum value that differs only in capitalisation.
- The 200-character limit lives in a Python validator, not in `maxLength`. The Claude API rejects `minLength` and `maxLength` with a 400 error. Its SDK helpers strip unsupported constraints from the request and move them into the field description. Keeping such rules in Python means the same schema works everywhere, and Pydantic still enforces them.

## A validate-and-repair loop

The core loop is provider-agnostic. It takes any function that sends messages and returns text plus a normalised stop reason.

```python
import json
from typing import Callable, Literal
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json


class LLMResult(BaseModel):
    text: str
    stop_reason: Literal["complete", "refusal", "max_tokens"]


class ExtractionFailed(Exception):
    pass


def extract(call_llm: Callable[[list[dict]], LLMResult],
            model_cls: type[BaseModel], user_text: str, max_attempts: int = 3):
    schema = json.dumps(model_cls.model_json_schema())
    messages = [
        {"role": "system", "content": f"Reply with JSON matching this schema:\n{schema}"},
        {"role": "user", "content": user_text},
    ]
    last_error = None
    for attempt in range(1, max_attempts + 1):
        result = call_llm(messages)

        if result.stop_reason == "refusal":
            # Do not retry a refusal with the same input; surface it.
            raise ExtractionFailed(f"model refused: {result.text[:200]}")

        if result.stop_reason == "max_tokens":
            last_error = "output was cut off by the token limit"
            partial = from_json(result.text, allow_partial=True)  # salvage for logging
            print(f"  attempt {attempt}: truncated, partial keys {sorted(partial)}")
            messages.append({"role": "user",
                             "content": "Your reply was cut off. Reply again, shorter."})
            continue

        try:
            return model_cls.model_validate_json(result.text)
        except ValidationError as exc:
            last_error = exc.errors(include_url=False)
            print(f"  attempt {attempt}: {exc.error_count()} validation error(s)")
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "user", "content":
                "That JSON failed validation:\n"
                f"{json.dumps(last_error, default=str)}\n"
                "Return corrected JSON only."})

    raise ExtractionFailed(f"gave up after {max_attempts} attempts: {last_error}")
```

The repair message sends back Pydantic's own error list. Each error has a `loc` (the field path), a `msg` and the offending `input`, which tells the model exactly what to fix. That beats "please try again".

To test it without an API key, I used a scripted fake LLM that misbehaves on purpose:

```python
def scripted_llm(replies):
    queue = list(replies)
    return lambda messages: queue.pop(0)

fake = scripted_llm([
    LLMResult(text='{"category": "Billing", "priority": "urgent", "summary": "Charged twice"}',
              stop_reason="complete"),
    LLMResult(text='{"category": "billing", "priority": "high", "summ',
              stop_reason="max_tokens"),
    LLMResult(text='{"category": "billing", "priority": "high", '
                   '"summary": "Charged twice for the March invoice", "customer_email": null}',
              stop_reason="complete"),
])
print(repr(extract(fake, SupportTicket, "I was billed twice in March, please refund.")))
```

Running it printed:

```text
  attempt 1: 2 validation error(s)
  attempt 2: truncated, partial keys ['category', 'priority']
SupportTicket(category='billing', priority='high', summary='Charged twice for the March invoice', customer_email=None)
```

The first reply had two problems: `"urgent"` is not an allowed priority, and `customer_email` was missing. `"Billing"` was not counted as an error because the before-validator lowercased it. The second reply was truncated, and `from_json(..., allow_partial=True)` still recovered the keys that had arrived. The third reply validated. A scripted refusal raised `ExtractionFailed` straight away, without retrying.

## Refusals and partial outputs

These two cases look like a validation failure, but you should not treat them as one.

- **Refusals.** OpenAI returns a separate `refusal` content item instead of your schema. Claude returns `stop_reason: "refusal"` with a 200 status, and its docs warn that the output might not match your schema. Retrying the same input usually wastes tokens. Log it, and return a clear error or a fallback to the caller.
- **Truncation.** OpenAI marks the response `status: "incomplete"` with `incomplete_details.reason == "max_output_tokens"`. Claude returns `stop_reason: "max_tokens"`. Anthropic's advice is to retry with a higher `max_tokens`. Partial parsing is useful for logs and progress bars, but I would not persist a partially parsed object as if it were complete.

## Adapters for real providers

Each provider gets a thin adapter that maps its response onto `LLMResult`. Check the method and field names against the SDK version you install, because both providers evolve their APIs. Model names are placeholders.

```python
from openai import OpenAI
import anthropic

def openai_call(messages, model="YOUR_MODEL"):
    resp = OpenAI().responses.create(
        model=model, input=messages, max_output_tokens=1024,
        text={"format": {"type": "json_schema", "name": "SupportTicket",
                         "schema": SupportTicket.model_json_schema(), "strict": True}},
    )
    if resp.status == "incomplete" and resp.incomplete_details.reason == "max_output_tokens":
        return LLMResult(text=resp.output_text or "", stop_reason="max_tokens")
    for item in resp.output:
        for part in getattr(item, "content", None) or []:
            if part.type == "refusal":
                return LLMResult(text=part.refusal, stop_reason="refusal")
    return LLMResult(text=resp.output_text, stop_reason="complete")

def claude_call(messages, model="YOUR_MODEL"):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    resp = anthropic.Anthropic().messages.create(
        model=model, max_tokens=1024, system=system, messages=convo,
        output_config={"format": {"type": "json_schema",
                                  "schema": SupportTicket.model_json_schema()}},
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    stop = {"refusal": "refusal", "max_tokens": "max_tokens"}.get(resp.stop_reason, "complete")
    return LLMResult(text=text, stop_reason=stop)
```

Swapping providers is now a one-argument change: `extract(claude_call, SupportTicket, text)`.

## Keep schemas simple

The constraint engines have limits. The Claude API does not support recursive schemas, numeric bounds or string length bounds. It caps a request at 24 optional parameters and 16 union-typed parameters, and rejects anything beyond that as too complex. What helps:

- Prefer flat objects with a handful of fields over deep nesting.
- Use enums for anything your code branches on.
- Put business rules (ranges, lengths, cross-field checks) in Pydantic validators, not in the schema.
- Do not add a "reasoning" field asking for step-by-step thinking. Anthropic notes it can trigger a refusal. A short "explanation" field is safer.
- Version your models. When the schema changes, old prompts and stored outputs need to know which version they belong to.

The pattern stays the same whichever provider you use: let the API constrain what it can, validate everything yourself, repair with specific errors, and treat refusals and truncation as distinct outcomes.

## Sources

- [OpenAI: Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI Cookbook: Introduction to Structured Outputs](https://developers.openai.com/cookbook/examples/structured_outputs_intro)
- [Claude API: Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Pydantic: JSON](https://docs.pydantic.dev/latest/concepts/json/)
- [Pydantic: JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Pydantic: Models](https://docs.pydantic.dev/latest/concepts/models/)
