# Content Quality Gate

RAG-based content verification pipeline. Checks new content against an existing corpus for contradictions and verifies voice & tone compliance.

## What's here

- **`rag_ask_qn.py`** — Ask questions against the content corpus (Qase blog + help center)
- **`rag_verify.py`** — Verify claims in a text file against the corpus. Per-paragraph retrieval, per-claim categorization (supported / contradicted / not covered)
- **`voice_check.py`** — Check an article against brand voice & tone guidelines. Single LLM call, no RAG
- **`rag_eval.py`** — RAGAS evaluation (faithfulness + context recall)
- **`rag_eval_simple.py`** — Simple LLM-judge evaluation (aligned / not aligned)
- **`blog_slice_plaintext.py`** — Semantic chunker for preparing your own content corpus

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
```

## Usage

### Ask questions to the corpus

```bash
.venv/bin/python rag_ask_qn.py --question "What are test suites in Qase?"
.venv/bin/python rag_ask_qn.py --k 5 --model gpt-4o
```

### Verify claims in an article

```bash
.venv/bin/python rag_verify.py article.txt
.venv/bin/python rag_verify.py article.txt --k 5 --model gpt-4o
```

### Check voice & tone

```bash
.venv/bin/python voice_check.py article.txt
.venv/bin/python voice_check.py article.txt --model gpt-4o --guidelines qase_voice_tone.md
```

### Run evaluation

```bash
# RAGAS evaluation (faithfulness + context recall)
.venv/bin/python rag_eval.py --eval eval_dataset.jsonl

# Simple LLM-judge evaluation
.venv/bin/python rag_eval_simple.py --eval eval_dataset.jsonl
```

## Data

The repo includes pre-chunked content from publicly available sources:

- **`blog_chunks.jsonl`** — Qase blog posts
- **`help_chunks.jsonl`** — Qase help center articles
- **`eval_dataset.jsonl`** — Hand-written ground truths for evaluation
- **`qase_voice_tone.md`** — Qase brand voice & tone guidelines

## Using your own content

To use this with your own content corpus:

1. Prepare a JSON file with your content in the format `[{"slug": "plaintext"}, ...]`
2. Run `blog_slice_plaintext.py` to chunk it (edit `INPUT_FILE` and `OUTPUT_FILE` as needed)
3. Point the scripts at your chunks: `.venv/bin/python rag_verify.py article.txt --input your_chunks.jsonl`

## Architecture

```
Content corpus (JSONL)
    → ChromaDB vector store (text-embedding-3-large, cached to disk)
        → similarity_search(k=8) per paragraph
            → LLM claim verification (gpt-4o-mini)
```

Voice & tone check is a separate path: guidelines + article → single LLM call → issues.

## Environment

Requires `.env` with:
- `OPENAI_API_KEY` — Required for embeddings and LLM
