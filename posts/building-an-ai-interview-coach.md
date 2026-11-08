---
title: Building an AI interview coach on free models
date: 2026-11-08
summary: An AI panel that asks one question at a time, grades the answer, corrects it and asks the follow-up. What the design looks like and what broke in production.
tags: Agents, LLMs, System design
---

I wanted to practise for AI engineering interviews the way they actually run: one question, a real answer, honest feedback, and then the cross-question that tests whether I really understood it. So I built an interview coach and deployed it.

## How it works

- **One question at a time** from a question bank, mixed with fresh questions the model writes on the spot.
- **LLM-as-judge grading.** The model scores each answer from 0 to 10 against a hidden checklist of what a strong answer covers, and returns structured JSON.
- **Corrections, not just a score.** Each mistake is shown as what I said next to what is actually true, followed by my own answer rewritten correctly and a short explanation of the concept.
- **A follow-up ladder.** A strong answer escalates to a harder follow-up. A weak one gets a question on the biggest gap. After two follow-ups the panel moves on, like a real interviewer.
- **Coding rounds graded by real tests.** Code runs in a separate, resource-limited process, so pass or fail never depends on the model.

## The stack

FastAPI for the backend, Postgres for users, scores and the question bank, and OpenRouter for the models. The model list is configuration, so the app falls back from one free model to the next when one is rate-limited, and switching to a paid model needs no code change.

## What broke

Importing a 223-question PDF took the whole server down with a 502. The endpoint did its database work one row at a time inside the request, so the server stopped answering health checks and the host restarted it. Running the import in a worker thread and inserting all rows in one batch fixed it: in testing the same import finishes in a couple of seconds and the site stays responsive throughout.

## What I'd do next

Track which follow-ups I fail most often and feed them back into the question mix, so practice time goes where I'm weakest.
