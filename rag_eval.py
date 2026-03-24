import argparse
import json
import os
from typing import Dict, Iterable, List

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness

from rag_utils import DEFAULT_INPUTS, build_prompt, build_vector_store


def iter_eval_rows(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ragas evaluation runner.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--eval", default="eval_blog_ideas.jsonl")
    parser.add_argument("--output", default="ragas_results.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of questions to evaluate")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # Load eval questions
    eval_rows = list(iter_eval_rows(args.eval))
    if args.limit:
        eval_rows = eval_rows[:args.limit]
    print(f"Loaded {len(eval_rows)} questions from {args.eval}")

    # Build vector store
    print(f"\nBuilding vector store (model: {args.embedding_model})...")
    vector_store = build_vector_store(
        input_paths=args.input,
        embedding_model=args.embedding_model,
    )

    prompt = build_prompt()
    llm = ChatOpenAI(model=args.model)
    print(f"LLM: {args.model}")

    print(f"\nGenerating RAG answers ({len(eval_rows)} questions, k={args.k})...\n")

    rows: List[Dict] = []
    for idx, item in enumerate(eval_rows, 1):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        if not question or not ground_truth:
            continue

        print(f"\033[1m[{idx}/{len(eval_rows)}] {question}\033[0m")

        # Retrieve
        retrieved_docs = vector_store.similarity_search(question, k=args.k)
        sources = set()
        for doc in retrieved_docs:
            source_type = doc.metadata.get("source_type", "blog")
            name = doc.metadata.get("slug") or doc.metadata.get("title", "?")
            sources.add(f"{source_type}:{name}")
        print(f"    Retrieved {len(retrieved_docs)} chunks from: {', '.join(sorted(sources))}")

        # Generate answer
        contexts = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n".join(contexts)
        print(f"    Generating answer...")
        answer = llm.invoke(prompt.invoke({"question": question, "context": context_text}))

        rows.append(
            {
                "user_input": question,
                "response": answer.content,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            }
        )
        print()

    if not rows:
        raise RuntimeError("No evaluation rows loaded. Check eval dataset format.")

    print(f"Running Ragas evaluation (Faithfulness + ContextRecall)...")
    print(f"  This uses an LLM judge to score each answer — may take a while...\n")

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextRecall()],
        llm=llm,
    )

    df = result.to_pandas()

    # Per-question results
    print(f"{'='*60}")
    print(f"Per-question scores:\n")
    for _, row in df.iterrows():
        question = row["user_input"]
        faithfulness = row.get("faithfulness", float("nan"))
        context_recall = row.get("context_recall", float("nan"))
        print(f"  \033[1m{question}\033[0m")
        print(f"    Faithfulness:   {faithfulness:.2f}")
        print(f"    ContextRecall:  {context_recall:.2f}")
        print()

    # Aggregate
    avg_faithfulness = df["faithfulness"].mean()
    avg_context_recall = df["context_recall"].mean()
    print(f"{'='*60}")
    print(f"Aggregate scores ({len(df)} questions):")
    print(f"  Faithfulness:   {avg_faithfulness:.3f}")
    print(f"  ContextRecall:  {avg_context_recall:.3f}")
    print(f"{'='*60}")

    df.to_json(args.output, orient="records", lines=True)
    print(f"Saved per-row results to {args.output}")


if __name__ == "__main__":
    main()
