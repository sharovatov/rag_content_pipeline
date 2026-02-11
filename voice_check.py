import argparse
import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


VOICE_CHECK_PROMPT = """You are an editorial reviewer. You will be given brand voice & tone guidelines and an article. Your job is to check the article against the guidelines and flag any issues.

Guidelines:
{guidelines}

Article:
{article}

Instructions:
1. Evaluate the article against each core voice principle listed in the guidelines.
2. Check for words and framing the guidelines say to avoid.
3. Run the self-check questions from the guidelines.
4. For each issue found: quote the passage, name the violated guideline, explain why it's a problem, and suggest a rewrite.

Respond with JSON only:
{{
  "issues": [
    {{
      "passage": "quoted text from article",
      "guideline": "which principle or rule it violates",
      "explanation": "why it's a problem",
      "suggestion": "suggested rewrite"
    }}
  ],
  "overall": "brief overall assessment",
  "conforms": true or false
}}"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check article against voice & tone guidelines."
    )
    parser.add_argument("file", help="Article text file to check")
    parser.add_argument(
        "--guidelines",
        default="qase_voice_tone.md",
        help="Guidelines markdown file (default: qase_voice_tone.md)",
    )
    parser.add_argument("--model", default="gpt-5.2")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    with open(args.guidelines, "r", encoding="utf-8") as f:
        guidelines = f.read().strip()
    if not guidelines:
        raise RuntimeError("Guidelines file is empty.")

    with open(args.file, "r", encoding="utf-8") as f:
        article = f.read().strip()
    if not article:
        raise RuntimeError("Article file is empty.")

    print(f"Checking {args.file} ({len(article)} chars) against {args.guidelines}\n")

    llm = ChatOpenAI(model=args.model, temperature=0)
    response = llm.invoke(
        VOICE_CHECK_PROMPT.format(guidelines=guidelines, article=article)
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON:")
        print(response.content)
        return

    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    bold = "\033[1m"
    reset = "\033[0m"

    issues = result.get("issues", [])
    for i, issue in enumerate(issues, 1):
        print(f"  {red}{bold}Issue {i}{reset}")
        print(f"  {red}Passage:{reset}    \"{issue.get('passage', '')}\"")
        print(f"  {yellow}Guideline:{reset}  {issue.get('guideline', '')}")
        print(f"  Explanation: {issue.get('explanation', '')}")
        print(f"  {green}Suggestion:{reset}  {issue.get('suggestion', '')}")
        print()

    conforms = result.get("conforms", False)
    overall = result.get("overall", "N/A")
    status_color = green if conforms else red
    status_label = "PASS" if conforms else "FAIL"

    print(f"{'=' * 60}")
    print(f"Issues found: {len(issues)}")
    print(f"Result: {status_color}{bold}{status_label}{reset}")
    print(f"Overall: {overall}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
