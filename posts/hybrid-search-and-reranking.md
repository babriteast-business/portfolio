---
title: Hybrid search and reranking, explained with code
date: 2026-03-16
summary: Combining BM25 and dense vectors with reciprocal rank fusion, then reranking with a cross-encoder, in plain Python with the trade-offs explained
tags: RAG, Python
---

Dense embeddings are good at meaning and bad at exact strings. Ask a vector index about "ERR-4012" and it may return a passage about payment errors in general, because to the embedding model one error code looks much like another. Keyword search has the opposite problem: it finds "ERR-4012" instantly but misses a passage that says "the token has expired" when the user typed "my login keeps timing out".

Hybrid search runs both and merges the results. A reranker then reads the merged shortlist properly and puts the best passages first. This post walks through each stage with code you can run, and the costs you are taking on at each step.

## Two retrievers with different blind spots

**BM25** scores a document by how often the query terms appear in it, weighted by how rare each term is across the corpus and normalised for document length. It needs no model and no GPU, and it is very good at exact identifiers: error codes, product SKUs, function names, people's names, legal clause numbers.

**Dense retrieval** embeds the query and every chunk into vectors and returns the nearest neighbours. It handles paraphrase, synonyms and questions phrased nothing like the source text.

The two fail on different queries, which is exactly why combining them works. Anthropic's contextual retrieval experiments show the effect. Measured as the share of queries whose relevant chunk was missing from the top 20, contextualised embeddings alone cut failures by 35% against their baseline, adding a contextualised BM25 index alongside them raised that to 49%, and adding a reranker on top raised it to 67%.

## Setting up both indexes

The `rank_bm25` package does no text processing, so tokenisation is your job, and it matters. A naive split on non-alphanumeric characters turns "ERR-4012" into "err" and "4012", which throws away the exact match you wanted BM25 for. The regex below keeps hyphenated and dotted codes together.

```python
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

docs = {
    "d1": "Error ERR-4012 means the payment token has expired. Request a new token.",
    "d2": "Payment failures are usually caused by expired cards or insufficient funds.",
    "d3": "To rotate an API key, open the dashboard and choose Regenerate.",
    "d4": "Token expiry: access tokens last one hour and refresh tokens last 30 days.",
}
ids = list(docs)
texts = [docs[i] for i in ids]


def tokenise(text: str) -> list[str]:
    # Keep codes like "ERR-4012" and "v2.3" together as single tokens.
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())


# Sparse index
bm25 = BM25Okapi([tokenise(t) for t in texts])

# Dense index
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_vecs = embedder.encode_document(texts, normalize_embeddings=True)


def bm25_search(query: str, top_k: int = 50) -> list[str]:
    scores = bm25.get_scores(tokenise(query))
    order = np.argsort(scores)[::-1][:top_k]
    return [ids[i] for i in order if scores[i] > 0]


def dense_search(query: str, top_k: int = 50) -> list[str]:
    q = embedder.encode_query(query, normalize_embeddings=True)
    sims = doc_vecs @ q  # cosine similarity, because both sides are normalised
    order = np.argsort(sims)[::-1][:top_k]
    return [ids[i] for i in order]
```

`encode_query` and `encode_document` are the current sentence-transformers methods for retrieval. If a model defines separate query and document prompts, they apply them for you; for a model without them, such as this small general-purpose one, they behave like `encode`. In production the dense side would be a vector database and the sparse side a search engine, but the shape of the outputs is the same: two ranked lists of ids.

## Merging with reciprocal rank fusion

You cannot simply add the two scores together. BM25 scores are unbounded and depend on the corpus, while cosine similarities sit in a narrow band. One scale would swamp the other.

Reciprocal rank fusion (RRF), from Cormack, Clarke and Büttcher's 2009 SIGIR paper, avoids the problem by ignoring scores and using only ranks:

```text
RRF(d) = Σ over r in R of  1 / (k + r(d))
```

Here R is the set of ranked lists, r(d) is the 1-based position of document d in list r, and a document missing from a list contributes nothing for it. The paper fixed k = 60 during a pilot study, found it near-optimal, and noted results were not very sensitive to the exact value. The constant stops any single list's top result from dominating.

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60,
                           weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Fuse several ranked lists of document ids. Ranks are 1-based."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

A quick illustration of why this works. With k = 60, a document ranked first by one retriever and absent from the other scores 1/61, about 0.0164. A document ranked fifth by both scores 2/65, about 0.0308. Agreement between retrievers beats a single strong opinion, which is usually what you want.

The optional `weights` argument lets you lean towards one retriever. This is the same idea as the `alpha` parameter in vector databases with built-in hybrid search. Weaviate, for example, offers both a rank-based fusion using 1/(rank + 60) and a relative-score fusion that rescales each list's scores to the range 0 to 1 before combining them. Rank-based fusion is the safer default when you do not trust the score distributions.

## Reranking with a cross-encoder

Both retrievers so far are fast because they compare precomputed representations. A bi-encoder embeds the query and the document separately, so it can never look at how specific query words relate to specific document words.

A cross-encoder reads the query and one document together in a single forward pass and outputs a relevance score. That is far more accurate, and far more expensive: one model call per query-document pair, with nothing precomputed. So you only run it on the fused shortlist.

sentence-transformers has a `rank` method that does the pairing and sorting for you. It returns a list of dicts with `corpus_id` (the index into the list you passed) and `score`, sorted best first.

```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def hybrid_search(query: str, candidates: int = 50, final_k: int = 5) -> list[tuple[str, float]]:
    fused = reciprocal_rank_fusion([bm25_search(query, candidates),
                                    dense_search(query, candidates)])
    pool = [doc_id for doc_id, _ in fused[:candidates]]
    ranked = reranker.rank(query, [docs[d] for d in pool], top_k=final_k)
    return [(pool[r["corpus_id"]], r["score"]) for r in ranked]


for doc_id, score in hybrid_search("What does ERR-4012 mean?", final_k=3):
    print(doc_id, round(score, 3), docs[doc_id])
```

Two details catch people out. First, `corpus_id` indexes into the candidate list you passed, not your document store, which is why the code maps it back through `pool`. Second, treat reranker scores as an ordering, not a calibrated probability. If you want a cut-off threshold, set it from an eval on your own data.

The MS MARCO MiniLM model is small and English-only, which makes it a good default for experiments. For multilingual content or harder domains, swap in a larger reranker or a hosted reranking API; the shape of the code stays the same.

## Latency and cost

Each stage adds a cost, so it helps to know where it goes:

- **BM25** is cheap. The index lives in memory or in a search engine you may already run.
- **Dense retrieval** costs one embedding call per query plus an approximate nearest-neighbour lookup.
- **Fusion** is a dictionary and a sort; it is effectively free.
- **Reranking** is the expensive part. Its cost grows linearly with the number of candidates, because every candidate is a separate model input. Anthropic's write-up reranked 150 candidates down to 20 and still flagged the extra latency as something to tune.

The main lever is the candidate count. Reranking more candidates raises the chance that the right passage is in the pool, at the price of latency. Measure recall of the fused list before reranking: if the right passage is already in the top 30, reranking 150 buys you nothing but time. Batch size and running the reranker on a GPU are the other levers.

## When hybrid is worth it

Hybrid search pays off when queries contain tokens that carry meaning embeddings cannot capture:

- error codes, ticket ids, SKUs, part numbers and version strings
- names of people, products, companies and places, especially rare ones
- code identifiers, configuration keys and command-line flags
- domain jargon and acronyms the embedding model never saw in training

It helps less when queries are conversational and the corpus is general prose, where dense retrieval already does well. It also adds a second index to keep in sync, so do not add it on faith. Build a golden set that includes some exact-match queries, measure recall@k for dense alone, BM25 alone and the fused list, and let those numbers decide. The same goes for the reranker: compare the order before and after it on the same questions.

## Sources

- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (Cormack, Clarke and Büttcher, SIGIR 2009)](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [Introducing Contextual Retrieval, Anthropic](https://www.anthropic.com/news/contextual-retrieval)
- [Sentence Transformers: Cross-Encoder usage](https://sbert.net/docs/cross_encoder/usage/usage.html)
- [cross-encoder/ms-marco-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [rank_bm25 on GitHub](https://github.com/dorianbrown/rank_bm25)
- [Hybrid search fusion algorithms, Weaviate](https://weaviate.io/blog/hybrid-search-fusion-algorithms)
