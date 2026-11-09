---
title: Chunking strategies for RAG that actually retrieve
date: 2026-02-09
summary: Fixed, recursive, structure-aware and semantic chunking compared, with a small Python splitter and a way to tune it with retrieval evals
tags: RAG, Python
---

When a RAG system gives a bad answer, the first thing to check is whether it retrieved the right text, and when it did not, chunking is one of the first suspects. The answer was in the corpus, but it was split across two chunks, or buried in a chunk about something else, or separated from the heading that said what it was about.

This post covers the main chunking strategies, the trade-offs behind chunk size and overlap, how to keep tables and headings intact, and a small splitter written in plain Python. It ends with the part that matters most: picking settings with a retrieval eval rather than a hunch.

## Four ways to cut a document

**Fixed-size chunking** cuts every N tokens, usually with some overlap. It is fast and predictable, but it ignores structure, so it will happily cut a sentence, a table or a code block in half.

**Recursive chunking** tries the coarsest separator first (blank lines), and only falls back to finer ones (newlines, then sentences, then words) when a piece is still too big. Chunks tend to end at natural boundaries, and it costs almost nothing extra.

**Structure-aware chunking** uses the document's own format. For Markdown or HTML that means splitting on headings, keeping each section's heading path as context, and treating tables and code blocks as units that are never split.

**Semantic chunking** embeds sentences and starts a new chunk where the similarity between neighbours drops. It sounds like the obvious winner, but the evidence is mixed. A study from Vectara and the University of Wisconsin–Madison found its benefits "highly context-dependent" and concluded that fixed-size chunking "remains a more efficient and reliable choice for practical RAG applications". It also means embedding every sentence at index time, on top of embedding the chunks themselves.

My default is recursive splitting inside structure-aware sections. I only reach for semantic or LLM-based chunking when an eval shows the simpler approach is failing.

## Chunk size and overlap

Size is a trade-off between precision and context. Small chunks embed one idea each, so they match queries sharply, but a single chunk may not hold enough to answer the question. Large chunks carry more context, but their embeddings average over several topics, and every irrelevant token you retrieve costs prompt space.

Chroma's technical report on chunking is the most useful public data I know of here. Using token-level recall and precision over five corpora, they found a recursive splitter at 200 tokens with no overlap was consistently strong, and that the 800-token, 400-overlap setting used as the OpenAI Assistants default had slightly below-average recall and the lowest precision. The best and worst strategies differed by up to 9% in recall. The authors are clear that their dataset is small, so treat these as a starting range, not a rule.

Overlap is less of a free win than it looks. It helps when an answer straddles a boundary, but it duplicates tokens, which lowers precision and can fill your top-k with near-copies. In the same report, overlap raised recall for one embedding model and hurt precision across the board. I start with no overlap and add it only if the eval shows boundary misses.

## Small-to-big retrieval

Parent-child retrieval, also called small-to-big, separates what you search from what you send to the model. You index small child chunks for sharp matching, but each child stores the id of its parent: the whole section, or a larger window around it. At query time you retrieve children, look up their parents, de-duplicate, and pass the parents to the LLM.

This gets you the precision of small chunks and the context of large ones. The cost is a second lookup and a larger prompt, so cap how many parents you return.

## Keep structure, and attach metadata

Three structural rules prevent most of the silly failures:

- **Never split a table or code block.** A table row without its header row is close to meaningless. If a table is too big for one chunk, split it by rows and repeat the header in each piece.
- **Carry the heading path into the chunk text.** "Refunds are issued within 30 days" retrieves far better as "Billing guide > Refunds" plus that sentence. Put the breadcrumb in the embedded text, not only in metadata, because the embedding model never sees metadata.
- **Store useful metadata.** At minimum: document id, section or parent id, heading path, chunk index, and flags such as whether the chunk contains a table or code. Add source URL, last-updated date and access-control tags if your system needs filtering or citations.

Prepending the breadcrumb is a cheap, deterministic relative of Anthropic's contextual retrieval, which uses an LLM to write a short context for every chunk before embedding and BM25 indexing. In their tests, contextual embeddings alone cut the top-20 retrieval failure rate by 35%. If headings are not enough context for your documents, that technique is the next step.

## A structure-aware splitter in plain Python

This splitter needs no framework. It parses Markdown into blocks under their heading path, keeps fenced code and tables whole, recursively splits oversized prose, and packs the result into chunks. It counts words as a stand-in for tokens; swap `n_tokens` for your embedding model's tokeniser in real use.

```python
import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SEPARATORS = ["\n\n", "\n", ". ", " "]


def n_tokens(text: str) -> int:
    # Rough proxy: swap in your embedding model's tokenizer for real use.
    return len(text.split())


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def split_blocks(markdown: str) -> list[tuple[list[str], str]]:
    """Split Markdown into (heading_path, block) pairs.

    Fenced code blocks and tables are kept as single blocks so they are
    never cut in half.
    """
    path: list[str] = []
    blocks: list[tuple[list[str], str]] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append((list(path), text))
            buf.clear()

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            if not in_fence:
                flush()
            buf.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            continue
        if in_fence:
            buf.append(line)
            continue
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
            continue
        is_table = line.lstrip().startswith("|")
        prev_is_table = bool(buf) and buf[-1].lstrip().startswith("|")
        if not line.strip() or (buf and is_table != prev_is_table):
            flush()
        if line.strip():
            buf.append(line)
    flush()
    return blocks


def recursive_split(text: str, max_tokens: int, seps=SEPARATORS) -> list[str]:
    """Split on the coarsest separator that works, recursing on oversized pieces."""
    if n_tokens(text) <= max_tokens:
        return [text]
    if not seps:
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    # Keep the separator on each piece so no full stops or newlines are lost.
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    pieces = [p for p in pieces if p.strip()]
    if len(pieces) == 1:
        return recursive_split(text, max_tokens, rest)
    out: list[str] = []
    current = ""
    for piece in pieces:
        if n_tokens(current + piece) <= max_tokens:
            current += piece
            continue
        if current.strip():
            out.append(current.strip())
        if n_tokens(piece) > max_tokens:
            out.extend(recursive_split(piece, max_tokens, rest))
            current = ""
        else:
            current = piece
    if current.strip():
        out.append(current.strip())
    return out


def is_atomic(block: str) -> bool:
    return block.startswith("```") or block.lstrip().startswith("|")
```

The second half groups blocks by section, packs them up to the size limit, and adds the breadcrumb and metadata. The `section_id` doubles as the parent id for small-to-big retrieval.

```python
def chunk_markdown(markdown: str, doc_id: str, max_tokens: int = 200,
                   overlap_tokens: int = 0) -> list[Chunk]:
    # 1. Group blocks into sections that share a heading path.
    sections: list[tuple[list[str], list[str]]] = []
    for path, block in split_blocks(markdown):
        if sections and sections[-1][0] == path:
            sections[-1][1].append(block)
        else:
            sections.append((path, [block]))

    chunks: list[Chunk] = []
    for s_idx, (path, blocks) in enumerate(sections):
        # 2. Pack blocks into chunks; only split prose blocks that are too big.
        pieces: list[str] = []
        for block in blocks:
            if is_atomic(block) or n_tokens(block) <= max_tokens:
                pieces.append(block)
            else:
                pieces.extend(recursive_split(block, max_tokens))
        packed: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and n_tokens(candidate) > max_tokens:
                packed.append(current)
                current = piece
            else:
                current = candidate
        if current:
            packed.append(current)

        # 3. Optional overlap: prefix with the tail of the previous chunk.
        for i, body in enumerate(packed):
            if overlap_tokens and i > 0:
                tail = " ".join(packed[i - 1].split()[-overlap_tokens:])
                body = f"{tail}\n\n{body}"
            breadcrumb = " > ".join(path)
            chunks.append(Chunk(
                # The breadcrumb goes into the embedded text, not just metadata.
                text=f"{breadcrumb}\n\n{body}" if breadcrumb else body,
                metadata={
                    "doc_id": doc_id,
                    "section_id": f"{doc_id}#s{s_idx}",
                    "heading_path": path,
                    "chunk_index": len(chunks),
                    "has_table": "\n|" in f"\n{body}",
                    "has_code": "```" in body,
                    "n_tokens": n_tokens(body),
                },
            ))
    return chunks
```

Run it on a short document and the table stays with its sentence and the code block stays with its explanation, each under the right breadcrumb:

```python
doc = """# Billing guide

## Refunds

Refunds are issued to the original payment method. Requests must be made within 30 days of purchase.

| Plan | Refund window |
|------|---------------|
| Monthly | 14 days |
| Annual | 30 days |
"""

for chunk in chunk_markdown(doc, doc_id="billing-guide", max_tokens=40):
    print(chunk.metadata["heading_path"], chunk.metadata["has_table"])
    print(chunk.text)
# ['Billing guide', 'Refunds'] True
# Billing guide > Refunds
#
# Refunds are issued to the original payment method. ...
# | Plan | Refund window |
# ...
```

One limit to know about: an atomic block bigger than `max_tokens` is kept whole and goes over the limit. That is deliberate, but log it, and split large tables by rows with the header repeated.

## Tune chunking with evals, not guesswork

Chunk settings are hyperparameters, and the only honest way to choose them is to measure retrieval on your own data. The loop I use:

1. Build a golden set of real questions, each labelled with the passage or section that answers it. Label at the source-text level, not the chunk level, so the labels survive re-chunking.
2. For each configuration (say 128, 256 and 512 tokens, with and without overlap, with and without breadcrumbs), re-index and run every question.
3. Count a hit when a retrieved chunk overlaps the labelled passage, then compute recall@k and MRR.
4. Look at the misses, not only the averages. A split table or a missing heading shows up immediately when you read the failed chunks.
5. Choose the smallest configuration that meets your recall target at the k you can afford to put in the prompt.

Keep the embedding model, k and the rest of the pipeline fixed while you do this, or you will not know what caused a change. The golden set and the scoring script take an afternoon to build, and you will reuse them every time you touch the retriever.

## Sources

- [Evaluating Chunking Strategies for Retrieval, Chroma technical report (Smith and Troynikov, 2024)](https://www.trychroma.com/research/evaluating-chunking)
- [Is Semantic Chunking Worth the Computational Cost? (Qu, Tu and Bao)](https://arxiv.org/abs/2410.13070)
- [Introducing Contextual Retrieval, Anthropic](https://www.anthropic.com/news/contextual-retrieval)
