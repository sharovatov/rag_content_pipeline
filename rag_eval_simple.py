import argparse
import json
import os
from typing import Dict, Iterable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_utils import DEFAULT_INPUTS, build_prompt, build_vector_store


def iter_eval_rows(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


JUDGE_PROMPT = """You are evaluating a RAG system's answer against a ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {answer}

Does the RAG answer align with the ground truth? Consider:
1. Does it capture the key points?
2. Does it contradict the ground truth?
3. Does it hallucinate information not supported by the ground truth?

Respond with JSON only:
{{"aligned": true/false, "reason": "brief explanation"}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple RAG evaluation.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--eval", default="eval_blog_ideas.jsonl")
    parser.add_argument("--output", default="eval_results.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    # Count eval questions
    eval_rows = list(iter_eval_rows(args.eval))
    print(f"Loaded {len(eval_rows)} questions from {args.eval}")

    # Build vector store
    print(f"\nBuilding vector store (model: {args.embedding_model})...")
    vector_store = build_vector_store(
        input_paths=args.input,
        embedding_model=args.embedding_model,
        limit=args.limit,
    )

    prompt = build_prompt()
    llm = ChatOpenAI(model=args.model)
    judge_llm = ChatOpenAI(model=args.model, temperature=0)
    print(f"LLM: {args.model}, Judge LLM: {args.model} (temperature=0)")

    results = []
    aligned_count = 0
    total_count = 0

    print(f"\nRunning evaluation ({len(eval_rows)} questions, k={args.k})...\n")

    for item in eval_rows:
        question = item.get("question")
        ground_truth = item.get("ground_truth")
        if not question or not ground_truth:
            continue

        total_count += 1
        print(f"\033[1m[{total_count}/{len(eval_rows)}] {question}\033[0m")

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

        # Judge alignment
        print(f"    Judging alignment...")
        judge_response = judge_llm.invoke(
            JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer.content,
            )
        )

        try:
            judgment = json.loads(judge_response.content)
            aligned = judgment.get("aligned", False)
            reason = judgment.get("reason", "")
        except json.JSONDecodeError:
            aligned = False
            reason = "Failed to parse judgment"

        if aligned:
            aligned_count += 1
            print(f"    -> \033[32mPASS\033[0m: {reason}")
        else:
            print(f"    -> \033[31mFAIL\033[0m: {reason}")
        print()

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer.content,
            "contexts": contexts,
            "aligned": aligned,
            "reason": reason,
        })

    # Summary
    print(f"{'='*60}")
    print(f"Results: {aligned_count}/{total_count} aligned ({100*aligned_count/total_count:.1f}%)")
    print(f"{'='*60}")

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
