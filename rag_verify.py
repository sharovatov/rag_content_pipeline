import argparse
import os
from typing import Sequence, List, Literal
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from rag_utils import (
    DEFAULT_INPUTS,
    build_vector_store,
    format_links,
)

DECOMPOSE_PROMPT = """You are a fact-checking assistant. You have been provided with a piece of input text, that you have to decompose into claims.

Your task: decompose the input text into claims. Each claim should be a distinct statement that can be verified or contradicted by the knowledge base.

Instructions:
1. Identify each distinct claim in the text.
2. Respond with JSON only:
{{
  "claims": [
    {{"claim": "...",}},
    ...
  ],
}}

--------EXAMPLES-----------
Example 1
Input: {{
    "input_text": "Large Language Models (LLM) are arguably the most effective thinking machines that are also widely accessible. I use LLMs extensively at work and otherwise - they have multiplied my productivity. But given how much I use them, I took a step back and tested them. Much like the alluring songs of the sirens, they are not without their risks."  
}}
Output: {{
    "claims": [
        "Large Language Models (LLM) are effective thinking machines that are widely accessible.",
        "Large Language Models are not without their risks."
    ]
}}

Example 2
Input: {{
    "input_text": "Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
}}
Output: {{
    "claims": [
        "Albert Einstein was a German physicist.",
        "Albert Einstein developed relativity and contributed to quantum mechanics."
    ]
}}
-----------------------------
Now perform the same with the following input text:
{input_text}
Output:
"""

VERIFY_PROMPT = """You are a fact-checking assistant. You have access to a knowledge base of content from the Qase blog and help center.

Your task: verify whether the claims provided are supported by the context supplied to you.

Instructions:
1. For each claim, categorize it as one of:
   - "supported" — the context provides evidence that supports this claim
   - "contradicted" — the context provides evidence that contradicts this claim
   - "not_covered" — the context does not contain enough information to verify this claim
2. Provide a brief explanation for each categorization.
3. Give an overall assessment.

Text to verify:
{claims}

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

def split_text_into_paragraphs(text: str) -> list[str]:
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

def decompose_input_text_into_claims(input_text: str, model: str) -> list[str]:
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(
        DECOMPOSE_PROMPT.format(input_text=input_text)
    )
    return response.content

def verify_claims(claims: list[str], context: str, model: str) -> list[str]:
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(
        VERIFY_PROMPT.format(claims=claims, context=context)
    )
    return response.content

class Claim(BaseModel):
    claim: str
    status: Literal["supported", "contradicted", "not_covered"]
    explanation: str

class VerificationResponse(BaseModel):
    claims: List[Claim]
    overall: str

def parse_verification_response(response: str) -> VerificationResponse:
    try:
        return VerificationResponse.model_validate_json(response)
    except ValidationError as e:
        print(f"Failed to parse LLM response as JSON:\n{response}\n{e}")
        raise

def print_formatted_result(result: VerificationResponse, retrieved_docs: Sequence[Document]):
    # Colored output per claim
    status_colors = {
        "supported": "\033[32m",     # green
        "contradicted": "\033[31m",  # red
        "not_covered": "\033[33m",   # yellow
    }
    reset = "\033[0m"
    claims = result.claims
    for claim in claims:
        status = claim.status
        color = status_colors.get(status, "")
        label = status.upper()
        print(f"  {color}[{label}]{reset} {claim.claim}")
        print(f"          {claim.explanation}")
        print()

    # Summary counts
    supported = sum(1 for c in claims if c.status == "supported")
    contradicted = sum(1 for c in claims if c.status == "contradicted")
    not_covered = sum(1 for c in claims if c.status == "not_covered")

    print(f"{'='*60}")
    print(f"Claims: {len(claims)} total — "
          f"\033[32m{supported} supported\033[0m, "
          f"\033[31m{contradicted} contradicted\033[0m, "
          f"\033[33m{not_covered} not covered\033[0m")
    print()
    print(f"Overall: {result.overall}")
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
    claims = decompose_input_text_into_claims(input_text=text, model=args.model)
    response = verify_claims(claims=claims, context=context, model=args.model)
    result = parse_verification_response(response)
    print_formatted_result(result, retrieved_docs)


if __name__ == "__main__":
    main()
