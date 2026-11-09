---
title: Orchestrator-workers, and when one agent is enough
date: 2026-07-27
summary: Workflows vs agents, the five common patterns, when a single agent beats a multi-agent system, and a small orchestrator with budgets and stop conditions
tags: Agents, System design
---

Multi-agent systems are an easy thing to reach for. A lead agent that hands work to a team of specialists sounds like it should be better than one agent doing everything. Sometimes it is. Often it is slower, more expensive and harder to debug than a single agent with good tools. This post is how I think about the choice, based mostly on two pieces from Anthropic's engineering team, plus a small provider-agnostic orchestrator you can adapt.

## Workflows and agents

Anthropic's "Building effective agents" draws a line I find useful. Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents are systems where the LLM dynamically directs its own process and tool usage.

The distinction is about who owns the control flow. In a workflow, your code decides what happens next. In an agent, the model does. Workflows are more predictable and cheaper to test. Agents are more flexible, at the cost of latency, tokens and the risk of compounding errors. The same article's main advice is to find the simplest solution that works and only add complexity when it clearly improves results, which for many applications means a single well-prompted LLM call with retrieval and good examples.

## The common patterns

The article describes five workflow patterns built on an "augmented LLM", a model with access to retrieval, tools and memory.

- **Prompt chaining.** Split a task into fixed steps, each LLM call working on the previous output, with programmatic checks ("gates") between them. Good when the task decomposes cleanly and you are happy to trade latency for accuracy. Example: write an outline, check it meets criteria, then write the document.
- **Routing.** Classify the input, then send it to a specialised prompt or model. Good when there are distinct categories that are better handled separately, such as sending easy questions to a smaller model.
- **Parallelisation.** Run calls at the same time and combine them programmatically. *Sectioning* splits a task into independent parts; *voting* runs the same task several times for more confidence. Example: one call answers the user while another screens for inappropriate content.
- **Orchestrator-workers.** A central LLM breaks the task down at run time, delegates to workers and synthesises their results. The key difference from parallelisation is that the subtasks are not known in advance. Good for things like code changes across an unknown number of files, or research across many sources.
- **Evaluator-optimiser.** One call generates, another evaluates and gives feedback, in a loop. Good when there are clear evaluation criteria and iteration measurably helps, such as translation or search that needs several rounds.

Agents proper sit beyond these: a model using tools in a loop based on feedback from the environment, until it finishes or hits a stopping condition.

## When one agent is enough

Anthropic's write-up of its multi-agent research system is the best public data I know of on the trade-off. Their system uses an orchestrator-worker design: a lead agent plans, then spins up subagents that search in parallel, each with its own context window. On their internal research evaluation, a lead on Claude Opus 4 with Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2%.

The same article is clear about the cost. In their data, agents typically use about four times more tokens than chat interactions, and multi-agent systems about fifteen times more. In their analysis of the BrowseComp evaluation, token usage by itself explained 80% of the performance variance. Part of why multi-agent works is simply that it spends more tokens on the problem, in parallel.

They also say where it is a poor fit: domains where all agents need to share the same context, or where there are many dependencies between agents. They call out most coding tasks as having fewer truly parallelisable parts than research.

From that, my rule of thumb is to start with one agent and move to orchestrator-workers only when most of these hold:

- The task breaks into parts that are **genuinely independent**, so workers do not need each other's intermediate results.
- The total information involved is **larger than one context window** can hold well.
- The value of the task justifies a **much larger token bill**.
- You can **evaluate** the output, so you will know whether the extra complexity helped.

If the work is sequential, tightly coupled or needs one consistent view of the state (most code edits, most form-filling, most customer support), a single agent with well-designed tools is usually the better system. Improving tool descriptions, error messages and examples often helps more than adding agents.

## Context isolation is the real benefit

The most useful thing a subagent gives you is not a "specialist persona". It is a clean context window. A worker starts with only its instruction, does a lot of messy exploration (searches, page reads, dead ends) and returns a short summary. The orchestrator never sees the mess, so its own context stays small and focused.

That suggests how to use subagents even in an otherwise single-agent system: delegate a noisy, self-contained subtask ("find every place this config value is read and report the file paths"), take back a summary, and carry on. You get the context benefit without designing a whole multi-agent architecture.

## Coordination costs

Every agent boundary is a lossy interface. The things that go wrong are predictable:

- **Vague delegation.** A worker told "research the market" duplicates another worker or goes off in the wrong direction. Each worker needs an objective, an output format, guidance on tools and sources, and clear boundaries.
- **Mis-sized effort.** Anthropic's lead agent is prompted with explicit scaling rules: simple fact-finding needs one agent with 3 to 10 tool calls, direct comparisons 2 to 4 subagents with 10 to 15 calls each, and complex research more than 10 subagents with clearly divided responsibilities.
- **Synthesis loss.** The orchestrator only sees summaries. If workers drop the caveats, the final answer is confidently wrong.
- **Harder debugging.** Non-deterministic agents calling non-deterministic agents is hard to trace. You need logging of each worker's instruction, tool calls and output from day one.

## A small orchestrator

Here is a compact orchestrator in Python. It is provider-agnostic: `LLMClient` is a protocol with one method, so you can wrap any SDK behind it. The planner returns a JSON list of subtasks, workers run in parallel with a fresh context and their own step cap, and everything shares one token and step budget.

```python
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass
class Completion:
    text: str
    tool_call: dict | None  # {"name": ..., "args": {...}} or None for a final answer
    tokens: int             # input + output tokens for this call


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[str]) -> Completion: ...


class BudgetExceeded(Exception):
    pass


class Budget:
    """Shared, thread-safe token and step budget for one task."""

    def __init__(self, max_tokens: int, max_steps: int):
        self.max_tokens, self.max_steps = max_tokens, max_steps
        self.tokens = self.steps = 0
        self._lock = threading.Lock()

    def charge(self, completion: Completion) -> None:
        with self._lock:
            self.tokens += completion.tokens
            self.steps += 1
            if self.tokens > self.max_tokens or self.steps > self.max_steps:
                raise BudgetExceeded(f"{self.steps} steps, {self.tokens} tokens")


def call(client: LLMClient, budget: Budget, system: str, messages: list[dict],
         tools: list[str] | None = None) -> Completion:
    completion = client.complete(system, messages, tools or [])
    budget.charge(completion)
    return completion


def plan(client: LLMClient, budget: Budget, task: str, max_workers: int) -> list[str]:
    system = ("Split the task into independent subtasks that can run in parallel. "
              f"Return a JSON list of at most {max_workers} instruction strings. "
              "Return [] if the task is simple enough to answer directly.")
    raw = call(client, budget, system, [{"role": "user", "content": task}]).text
    subtasks = json.loads(raw)
    if not isinstance(subtasks, list) or not all(isinstance(s, str) for s in subtasks):
        raise ValueError("planner did not return a list of strings")
    return subtasks[:max_workers]


def run_worker(client: LLMClient, budget: Budget, instruction: str,
               tools: dict[str, Callable[..., str]], max_steps: int = 8) -> str:
    """A worker gets a fresh context: only its instruction, never the whole conversation."""
    system = ("Complete the subtask using the tools. Finish with a concise summary "
              "of findings and the sources you used.")
    messages = [{"role": "user", "content": instruction}]
    for _ in range(max_steps):
        completion = call(client, budget, system, messages, list(tools))
        if completion.tool_call is None:
            return completion.text                         # stop: worker says it is done
        name, args = completion.tool_call["name"], completion.tool_call["args"]
        if name not in tools:
            result = f"Unknown tool {name!r}."
        else:
            try:
                result = tools[name](**args)
            except Exception as exc:                       # surface tool errors to the model
                result = f"Tool error: {exc}"
        messages.append({"role": "assistant", "content": completion.text, "tool_call": completion.tool_call})
        messages.append({"role": "tool", "content": result[:4000]})  # cap what flows back
    return "Stopped: step limit reached before the subtask was finished."


def orchestrate(client: LLMClient, task: str, tools: dict[str, Callable[..., str]],
                max_workers: int = 4, max_tokens: int = 200_000, max_steps: int = 60) -> str:
    budget = Budget(max_tokens, max_steps)
    try:
        subtasks = plan(client, budget, task, max_workers)
        if not subtasks:                                   # one call is enough
            return call(client, budget, "Answer the task.", [{"role": "user", "content": task}]).text
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda s: run_worker(client, budget, s, tools), subtasks))
        report = "\n\n".join(f"## Subtask: {s}\n{r}" for s, r in zip(subtasks, results))
        system = "Combine the worker reports into one answer. Note gaps and conflicts."
        return call(client, budget, system, [{"role": "user", "content": f"{task}\n\n{report}"}]).text
    except BudgetExceeded as exc:
        return f"Stopped early, budget exhausted ({exc})."
```

The default limits are illustrative, not recommendations; set them from your own traces. Note that the planner is allowed to return an empty list, in which case the task is answered with a single call. Making "do not fan out" a valid plan is a cheap way to avoid paying multi-agent costs on easy inputs.

## Budgets and stop conditions

Agents need explicit reasons to stop, because the model's own sense of "done" is unreliable. The sketch has four:

1. **The worker gives a final answer** (no tool call). The normal exit.
2. **A per-worker step cap.** A worker stuck in a loop returns a clear "stopped" message instead of running forever, and the orchestrator can report the gap.
3. **A shared token and step budget** across all calls. When it is exhausted, the whole task stops and says so. Because the check happens after each call, a task can overshoot by the size of the calls already in flight; if you need a hard ceiling, also set a per-call maximum output length on the provider side.
4. **A cap on fan-out** (`max_workers`), so a planner cannot decide to launch fifty workers.

In production I would add a wall-clock timeout, retries with backoff around `complete`, a check that stops a worker repeating the same tool call with the same arguments, and tracing of every call. Truncating tool output (`result[:4000]`) is a crude but useful guard against one huge page eating the budget.

## Summary

Start with the simplest thing: one call, then a workflow, then a single agent with good tools. Move to orchestrator-workers when the work is broad, parallel and valuable enough to justify a large multiple on tokens, and when you can measure the improvement. When you do, the wins come from context isolation and parallelism, and the risks come from vague delegation and lossy synthesis, so put your effort into the instructions you hand each worker and the budgets that stop them.

## Sources

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
