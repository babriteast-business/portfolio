---
title: Evaluating a RAG pipeline: retrieval first, then generation
date: 2026-07-06
summary: How to score retrieval and generation separately, build a golden set, calibrate an LLM judge, and trace a bad answer back to its cause
tags: RAG, Evaluation
---

A RAG system can fail in two quite different places. The retriever can miss the passage that holds the answer, or the generator can be handed the right passage and still produce something wrong. A single end-to-end score such as "answer quality is 72%" mixes the two together and tells you nothing about which one to fix.

So I evaluate in two stages: retrieval first, with cheap deterministic metrics, then generation, with an LLM judge that has been checked against human labels. This post covers the metrics, the golden set they depend on, a small Python harness, and how to use the results to debug.

## Retrieval metrics

Retrieval metrics need only two things per question: the ranked list of ids your retriever returned, and the set of ids that are actually relevant. No LLM is involved, so they are fast, free and repeatable. Here `rel_i` is 1 if the result at rank i is relevant, or a graded score for nDCG.

```text
recall@k    = |relevant ∩ top_k| / |relevant|
precision@k = |relevant ∩ top_k| / k
MRR         = mean over queries of 1 / rank of the first relevant result
              (0 if none is retrieved)
DCG@k       = Σ for i = 1..k of (2^rel_i − 1) / log2(i + 1)
nDCG@k      = DCG@k / ideal DCG@k
```

**Recall@k** is the one I watch most closely. If the answer is not in the top k, the generator cannot use it, however good the model is. Set k to the number of chunks you actually put in the prompt.

**Precision@k** tells you how much of the context is noise. Low precision is not fatal, but irrelevant chunks cost tokens and can distract the model.

**MRR** rewards putting the first relevant result near the top. It matters when you pass few chunks, or when a reranker is supposed to be doing its job.

**nDCG** handles graded relevance, where some passages answer the question fully and others only partly. The ideal DCG is the score of the best possible ordering, so nDCG runs from 0 to 1. Use it when you have graded labels; otherwise recall and MRR are easier to explain.

## Generation metrics

Once retrieval is measured, judge the answer itself. Three questions cover most of it:

- **Faithfulness, or groundedness.** Is every claim in the answer supported by the retrieved context? This is the hallucination check, and it needs no reference answer.
- **Answer relevance.** Does the answer address the question that was asked, or does it drift or dodge?
- **Correctness.** Does the answer agree with a reference answer written by someone who knows? This is the only one of the three that needs a labelled reference.

Faithfulness and correctness are independent. An answer can be faithful to a wrong passage, or correct from the model's own knowledge while ignoring the context. You want to know which of those you have.

## Building a golden set

Every metric above depends on a golden set: questions paired with the ids of the relevant passages and, ideally, a reference answer. A few principles keep it useful:

- **Start from real questions.** Search logs, support tickets and user interviews beat questions you invent, because yours will echo the wording of the documents.
- **Cover the hard cases on purpose.** Include exact-match queries (codes, names), questions that need two passages, and questions the corpus cannot answer, where the right behaviour is to say so.
- **Label passages, not chunk ids.** Record the source document and the span of text that answers the question. Chunk ids change every time you re-chunk; source spans do not.
- **Use an LLM to draft, people to approve.** Generating candidate questions from passages is a fast way to grow the set, but have someone review each one. Synthetic questions tend to repeat the source's phrasing, which flatters retrieval.
- **Version it.** Treat the golden set like code, and never tune against the questions you report on. Keep a held-out slice.

A few dozen carefully labelled questions are enough to start. You can grow it every time a user reports a bad answer.

## A small retrieval harness

This harness takes a golden set and any `search` function that returns ranked ids, and computes the retrieval metrics. It also collects the misses, which are the most useful output.

```python
import math
from statistics import mean


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("Every golden query needs at least one relevant id")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], grades: dict[str, int], k: int) -> float:
    def dcg(gains: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    actual = dcg([grades.get(d, 0) for d in retrieved[:k]])
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def evaluate(golden: list[dict], search, k: int = 5) -> dict:
    rows = []
    for item in golden:
        retrieved = search(item["question"])
        relevant = set(item["relevant_ids"])
        rows.append({
            "id": item["id"],
            "recall": recall_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "rr": reciprocal_rank(retrieved, relevant),
            "retrieved": retrieved[:k],
        })
    return {
        f"recall@{k}": mean(r["recall"] for r in rows),
        f"precision@{k}": mean(r["precision"] for r in rows),
        "mrr": mean(r["rr"] for r in rows),
        "misses": [r for r in rows if r["recall"] == 0],
    }
```

In `ndcg_at_k`, the loop index `i` starts at 0, so `log2(i + 2)` is the same as `log2(rank + 1)` in the formula. Here it is on a toy golden set with canned search results, so you can check the arithmetic by hand:

```python
golden = [
    {"id": "q1", "question": "How long do refresh tokens last?", "relevant_ids": ["d4"]},
    {"id": "q2", "question": "What does ERR-4012 mean?", "relevant_ids": ["d1", "d4"]},
    {"id": "q3", "question": "How do I rotate an API key?", "relevant_ids": ["d3"]},
]
fake_results = {
    "How long do refresh tokens last?": ["d2", "d4", "d1"],
    "What does ERR-4012 mean?": ["d1", "d2", "d3"],
    "How do I rotate an API key?": ["d2", "d1", "d4"],
}
report = evaluate(golden, search=lambda q: fake_results[q], k=3)
for name, value in report.items():
    if name != "misses":
        print(f"{name}: {value:.3f}")
for miss in report["misses"]:
    print("MISS", miss["id"], miss["retrieved"])
print("ndcg@3:", round(ndcg_at_k(["d1", "d2", "d4"], {"d1": 2, "d4": 1}, 3), 3))

# recall@3: 0.500
# precision@3: 0.222
# mrr: 0.500
# MISS q3 ['d2', 'd1', 'd4']
# ndcg@3: 0.964
```

If your golden set is labelled with source spans rather than ids, replace the set intersection with a check for whether a retrieved chunk overlaps a labelled span. The rest stays the same.

## LLM-as-judge, calibrated

Generation metrics usually need an LLM judge. That works, but the judge is a model with its own failure modes. The MT-Bench paper by Zheng and colleagues found that strong judges such as GPT-4 reached over 80% agreement with human preferences, about the level humans reach with each other, and also documented position, verbosity and self-enhancement biases. So a judge needs checking before you trust it.

The calibration loop I use:

1. Label a sample of answers by hand, ideally with two people, using the same rubric the judge gets.
2. Run the judge on the same sample.
3. Measure agreement, and look at every disagreement.
4. Tighten the rubric or the prompt, and repeat until agreement is acceptable for the decisions you will make.
5. Re-check whenever you change the judge model or the prompt.

Prefer binary or short-scale verdicts ("is every claim supported: yes or no") over 1 to 10 scores, which drift. Ask for the reasoning before the verdict, and when comparing two answers, run both orders to cancel out position bias.

Report Cohen's kappa alongside raw agreement, because raw agreement looks good whenever one label dominates:

```python
from collections import Counter


def agreement_and_kappa(human: list[str], judge: list[str]) -> tuple[float, float]:
    """Raw agreement and Cohen's kappa between two sets of labels."""
    n = len(human)
    observed = sum(h == j for h, j in zip(human, judge)) / n
    h_counts, j_counts = Counter(human), Counter(judge)
    expected = sum(h_counts[c] * j_counts[c] for c in h_counts) / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return observed, kappa


human = ["pass", "pass", "fail", "pass", "fail", "pass", "pass", "fail"]
judge = ["pass", "pass", "pass", "pass", "fail", "pass", "fail", "fail"]
print(agreement_and_kappa(human, judge))  # (0.75, 0.466...)
```

In this illustrative sample, 75% raw agreement shrinks to a kappa of about 0.47 once chance agreement is removed.

## Using RAGAS

You do not have to write every judge prompt yourself. RAGAS implements these metrics, and its 0.4 releases moved to a new metrics API under `ragas.metrics.collections`; the older `SingleTurnSample` style is being deprecated. Faithfulness, for example, splits the answer into claims and returns the share supported by the retrieved context:

```python
import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness


async def main() -> None:
    llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
    contexts = ["Refresh tokens last 30 days. Access tokens last one hour."]

    faithfulness = await Faithfulness(llm=llm).ascore(
        user_input="How long do refresh tokens last?",
        response="Refresh tokens last 30 days.",
        retrieved_contexts=contexts,
    )
    recall = await ContextRecall(llm=llm).ascore(
        user_input="How long do refresh tokens last?",
        retrieved_contexts=contexts,
        reference="Refresh tokens are valid for 30 days.",
    )
    print(faithfulness.value, recall.value)


asyncio.run(main())
```

Other metrics in the same module include `AnswerRelevancy` (listed as Response Relevancy in the docs, and it needs an embeddings model as well as an LLM), `ContextPrecision` and `FactualCorrectness`. Note that RAGAS's context recall is judged by an LLM against a reference answer, so it is not the same number as the id-based recall@k above. Library metrics are LLM judges too, so calibrate them against your own labels in the same way.

## Debugging failures

With both stages measured, each bad answer falls into one bucket:

- **Retrieval miss** (low recall@k). Read the chunks that were returned. Common causes are a split table or section, an exact code the embedding model blurred (try hybrid search), or the answer sitting just past k (try a reranker or a larger k).
- **Retrieved but ranked low** (good recall, poor MRR). The passage is there but buried under noise. Reranking is the usual fix.
- **Unfaithful answer** (good retrieval, low faithfulness). The model ignored or embellished the context. Tighten the prompt, require citations, or reduce the number of distracting chunks.
- **Faithful but wrong** (high faithfulness, low correctness). The context itself was wrong, stale or incomplete. That is a data problem, not a model problem.

Change one thing at a time, re-run the same golden set, and keep the per-question results, not only the averages. A shift in the mean tells you something changed; the per-question diff tells you what.

## Sources

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [Cumulated gain-based evaluation of IR techniques (Järvelin and Kekäläinen, 2002)](https://dl.acm.org/doi/10.1145/582415.582418)
- [RAGAS: list of available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [RAGAS: Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
