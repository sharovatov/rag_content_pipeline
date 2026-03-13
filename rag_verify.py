import argparse
import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import Sequence, Dict, Any

from rag_utils import (
    DEFAULT_INPUTS,
    build_vector_store,
    format_links,
)


VERIFY_PROMPT = """You are a fact-checking assistant. You have access to a knowledge base of content from the Qase blog and help center.

Your task: verify whether the claims in the given text are supported by the provided context.

Instructions:
1. Identify each distinct claim in the text.
2. For each claim, categorize it as one of:
   - "supported" — the context provides evidence that supports this claim
   - "contradicted" — the context provides evidence that contradicts this claim
   - "not_covered" — the context does not contain enough information to verify this claim
3. Provide a brief explanation for each categorization.
4. Give an overall assessment.

Text to verify:
{text}

Context from knowledge base:
{context}

Respond with JSON only:
{{
  "claims": [
    {{"claim": "...", "status": "supported|contradicted|not_covered", "explanation": "..."}},
    ...
  ],
  "overall": "brief overall assessment"
}}"""

def read_input_file(file: str) -> str:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise RuntimeError("Input file is empty.")
    return text

def split_text_into_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Text has {len(paragraphs)} paragraph(s)\n")
    return paragraphs

def retrieve_relevant_chunks(vector_store, text: str, paragraphs: list[str], k: int) -> Sequence[Document]:
    if len(paragraphs) <= 1:
        return vector_store.similarity_search(text, k=k)
    else:
        print(f"Retrieving relevant chunks per paragraph...\n")
        seen_ids = set()
        retrieved_docs = []
        for para in paragraphs:
            para_docs = vector_store.similarity_search(para, k=k)

            for doc in para_docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    retrieved_docs.append(doc)
        return retrieved_docs

def generate_response(text: str, context: str, model: str) -> str:
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(
        VERIFY_PROMPT.format(text=text, context=context)
    )
    return response.content

def parse_verification_response(response: str) -> Dict[str, Any]:
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse LLM response as JSON:\n{response}")


def print_formatted_result(result: Dict[str, Any], retrieved_docs: Sequence[Document]):
    # Colored output per claim
    status_colors = {
        "supported": "\033[32m",     # green
        "contradicted": "\033[31m",  # red
        "not_covered": "\033[33m",   # yellow
    }
    reset = "\033[0m"

    claims = result.get("claims", [])
    for i, claim_obj in enumerate(claims, 1):
        status = claim_obj.get("status", "not_covered")
        color = status_colors.get(status, "")
        label = status.upper()
        print(f"  {color}[{label}]{reset} {claim_obj.get('claim', '')}")
        print(f"          {claim_obj.get('explanation', '')}")
        print()

    # Summary counts
    supported = sum(1 for c in claims if c.get("status") == "supported")
    contradicted = sum(1 for c in claims if c.get("status") == "contradicted")
    not_covered = sum(1 for c in claims if c.get("status") == "not_covered")

    print(f"{'='*60}")
    print(f"Claims: {len(claims)} total — "
          f"\033[32m{supported} supported\033[0m, "
          f"\033[31m{contradicted} contradicted\033[0m, "
          f"\033[33m{not_covered} not covered\033[0m")
    print()
    print(f"Overall: {result.get('overall', 'N/A')}")
    print(f"{'='*60}")

    print(format_links(retrieved_docs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify text claims against RAG corpus.")
    parser.add_argument("file", help="Text file to verify")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS,
                        help="Input JSONL files (default: blog + help center)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    text = read_input_file(args.file)
    paragraphs = split_text_into_paragraphs(text)

    print(f"Verifying text from {args.file} ({len(text)} characters)\n")

    vector_store = build_vector_store(
        input_paths=args.input,
        embedding_model=args.embedding_model,limit=args.limit,
    )

    retrieved_docs = retrieve_relevant_chunks(vector_store, text, paragraphs, args.k)

    if not retrieved_docs:
        print("No relevant sources found in the corpus.")
        return

    print(f"Retrieved {len(retrieved_docs)} unique chunks\n")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = generate_response(text, context, args.model)
    result = parse_verification_response(response)
    print_formatted_result(result, retrieved_docs)


if __name__ == "__main__":
    main()
