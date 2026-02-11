import argparse
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_utils import (
    DEFAULT_INPUTS,
    build_prompt,
    build_vector_store,
    format_links,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over blog and help center content.")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS,
                        help="Input JSONL files (default: blog + help center)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    vector_store = build_vector_store(
        input_paths=args.input,
        embedding_model=args.embedding_model,
        limit=args.limit,
    )

    question = args.question or input("Question: ").strip()
    if not question:
        raise RuntimeError("Question is empty.")

    retrieved_docs = vector_store.similarity_search(question, k=args.k)
    if not retrieved_docs:
        print("I don't know.\n\nNo relevant sources found.")
        return

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = build_prompt()
    llm = ChatOpenAI(model=args.model)
    response = llm.invoke(prompt.invoke({"question": question, "context": context}))
    print(response.content + format_links(retrieved_docs))


if __name__ == "__main__":
    main()
