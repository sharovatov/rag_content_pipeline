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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas evaluation runner.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--eval", default="eval_blog_ideas.jsonl")
    parser.add_argument("--output", default="ragas_results.jsonl")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of questions to evaluate",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    return parser.parse_args()


def validate_env() -> None:
    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")


def load_eval_rows(eval_path: str, limit: int | None) -> List[Dict]:
    eval_rows = list(iter_eval_rows(eval_path))
    if limit:
        eval_rows = eval_rows[:limit]
    print(f"Loaded {len(eval_rows)} questions from {eval_path}")
    return eval_rows


def initialize_rag_components(args: argparse.Namespace):
    print(f"\nBuilding vector store (model: {args.embedding_model})...")
    vector_store = build_vector_store(
        input_paths=args.input,
        embedding_model=args.embedding_model,
    )
    prompt = build_prompt()
    llm = ChatOpenAI(model=args.model)
    print(f"LLM: {args.model}")
    return vector_store, prompt, llm


def collect_sources(retrieved_docs) -> List[str]:
    sources = set()
    for doc in retrieved_docs:
        source_type = doc.metadata.get("source_type", "blog")
        name = doc.metadata.get("slug") or doc.metadata.get("title", "?")
        sources.add(f"{source_type}:{name}")
    return sorted(sources)


def generate_answer_rows(
    eval_rows: List[Dict], vector_store, prompt, llm, k: int
) -> List[Dict]:
    print(f"\nGenerating RAG answers ({len(eval_rows)} questions, k={k})...\n")

    rows: List[Dict] = []
    for idx, item in enumerate(eval_rows, 1):
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        if not question or not ground_truth:
            continue

        print(f"\033[1m[{idx}/{len(eval_rows)}] {question}\033[0m")
        retrieved_docs = vector_store.similarity_search(question, k=k)
        source_names = collect_sources(retrieved_docs)
        print(
            f"    Retrieved {len(retrieved_docs)} chunks from: {', '.join(source_names)}"
        )

        contexts = [doc.page_content for doc in retrieved_docs]
        context_text = "\n\n".join(contexts)
        print("    Generating answer...")
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
    return rows


def run_ragas_eval(rows: List[Dict], llm):
    print("Running Ragas evaluation (Faithfulness + ContextRecall)...")
    print("  This uses an LLM judge to score each answer — may take a while...\n")
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextRecall()],
        llm=llm,
    )
    return result.to_pandas()


def print_scores(df) -> None:
    print(f"{'='*60}")
    print("Per-question scores:\n")
    for _, row in df.iterrows():
        question = row["user_input"]
        faithfulness = row.get("faithfulness", float("nan"))
        context_recall = row.get("context_recall", float("nan"))
        print(f"  \033[1m{question}\033[0m")
        print(f"    Faithfulness:   {faithfulness:.2f}")
        print(f"    ContextRecall:  {context_recall:.2f}")
        print()

    avg_faithfulness = df["faithfulness"].mean()
    avg_context_recall = df["context_recall"].mean()
    print(f"{'='*60}")
    print(f"Aggregate scores ({len(df)} questions):")
    print(f"  Faithfulness:   {avg_faithfulness:.3f}")
    print(f"  ContextRecall:  {avg_context_recall:.3f}")
    print(f"{'='*60}")


def main() -> None:
    args = parse_args()
    validate_env()
    eval_rows = load_eval_rows(args.eval, args.limit)
    vector_store, prompt, llm = initialize_rag_components(args)
    rows = generate_answer_rows(eval_rows, vector_store, prompt, llm, args.k)

    if not rows:
        raise RuntimeError("No evaluation rows loaded. Check eval dataset format.")

    df = run_ragas_eval(rows, llm)
    print_scores(df)
    df.to_json(args.output, orient="records", lines=True)
    print(f"Saved per-row results to {args.output}")


if __name__ == "__main__":
    main()
