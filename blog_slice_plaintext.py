import json
import re
from typing import Iterable, List, Tuple


INPUT_FILE = "blog_plain.json"
OUTPUT_FILE = "blog_chunks.jsonl"

MIN_TOKENS = 400
MAX_TOKENS = 800
OVERLAP_TOKENS = 80
MIN_SECTION_CHARS = 200


SEPARATOR_RE = re.compile(r"^-{10,}$")


def count_tokens(text: str) -> int:
    return len(text.split())


def split_sections(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    sections: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current.clear()

    for idx, line in enumerate(lines):
        if SEPARATOR_RE.match(line.strip()):
            flush()
            continue

        prev_blank = idx > 0 and lines[idx - 1].strip() == ""
        next_blank = idx + 1 < len(lines) and lines[idx + 1].strip() == ""
        is_heading = line.strip() and prev_blank and next_blank and len(line.strip()) <= 80

        if is_heading:
            flush()
            current.append(line)
            continue

        current.append(line)

    flush()
    return sections


def merge_small_sections(sections: List[str]) -> List[str]:
    merged: List[str] = []
    buffer = ""
    for section in sections:
        if not buffer:
            buffer = section
            continue
        if len(buffer) < MIN_SECTION_CHARS:
            buffer = f"{buffer}\n\n{section}"
        else:
            merged.append(buffer)
            buffer = section
    if buffer:
        merged.append(buffer)
    return merged


def chunk_paragraphs(paragraphs: List[str]) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush_with_overlap() -> Tuple[List[str], int]:
        nonlocal current, current_tokens
        if not current:
            return [], 0
        chunk_text = "\n\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        words = chunk_text.split()
        overlap_words = words[-OVERLAP_TOKENS:] if OVERLAP_TOKENS > 0 else []
        current = [" ".join(overlap_words)] if overlap_words else []
        current_tokens = len(overlap_words)
        return current, current_tokens

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = count_tokens(para)
        if para_tokens > MAX_TOKENS:
            if current_tokens >= MIN_TOKENS or current:
                flush_with_overlap()
            words = para.split()
            start = 0
            while start < len(words):
                end = min(start + MAX_TOKENS, len(words))
                chunks.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = max(0, end - OVERLAP_TOKENS)
            current = []
            current_tokens = 0
            continue

        if current_tokens + para_tokens > MAX_TOKENS:
            flush_with_overlap()

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current).strip())
    return [c for c in chunks if c]


def chunk_text(text: str) -> List[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return chunk_paragraphs(paragraphs)


def iter_posts(input_file: str) -> Iterable[Tuple[str, str]]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if not isinstance(item, dict):
            continue
        if len(item) != 1:
            continue
        slug, text = next(iter(item.items()))
        if slug and text:
            yield slug, text


def main() -> None:
    total_chunks = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for slug, text in iter_posts(INPUT_FILE):
            sections = merge_small_sections(split_sections(text))
            for section in sections:
                chunks = chunk_text(section)
                for idx, chunk in enumerate(chunks):
                    record = {"slug": slug, "chunk_index": idx, "text": chunk}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1
    print(f"Wrote {total_chunks} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
