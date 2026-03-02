import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


DEFAULT_INPUTS = ["blog_chunks.jsonl", "help_chunks.jsonl"]
BLOG_BASE_URL = "https://qase.io/blog/"
CHROMA_DIR = ".chroma"
HASH_FILE = ".chroma_hash"


def iter_chunks(paths: str | List[str]) -> Iterable[Dict]:
    """Iterate over chunks from one or more JSONL files."""
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if not Path(path).exists():
            print(f"  Skipping {path} (not found)")
            continue
        print(f"  Loading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_documents(
    chunks: Iterable[Dict],
    limit: Optional[int] = None,
) -> List[Document]:
    documents: List[Document] = []
    for idx, chunk in enumerate(chunks):
        if limit is not None and idx >= limit:
            break

        text = chunk.get("text")
        if not text:
            continue

        chunk_index = chunk.get("chunk_index", 0)

        if "slug" in chunk:
            # Blog post
            slug = chunk["slug"]
            link = f"{BLOG_BASE_URL}{slug}"
            source_type = "blog"
            doc_id = f"blog:{slug}:{chunk_index}:{idx}"
            metadata = {
                "slug": slug,
                "link": link,
                "chunk_index": chunk_index,
                "source_type": source_type,
            }
        elif "collection" in chunk:
            # Help center article
            link = chunk["url"]
            title = chunk.get("title", "")
            source_type = "help"
            doc_id = f"help:{title}:{chunk_index}:{idx}"
            metadata = {
                "title": title,
                "url": link,
                "link": link,
                "collection": chunk.get("collection", ""),
                "chunk_index": chunk_index,
                "source_type": source_type,
            }
        else:
            continue

        documents.append(Document(page_content=text, metadata=metadata, id=doc_id))
    return documents


def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "context"],
        template=(
            "Answer the question using ONLY the context below. "
            "Do not use your own knowledge. If the context does not contain enough information "
            "to answer the question, say: 'This is not covered in our knowledge base.'\n\n"
            "Do not include any links; the system will append sources.\n"
            "Do not guess or fill in gaps. Partial answers are better than made-up answers.\n"
            "When uncertain, state so.\n"
            "Explicitly list assumptions made.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:\n"
        ),
    )


def format_links(docs: List[Document]) -> str:
    blog_links = {}
    help_links = {}

    for doc in docs:
        link = doc.metadata.get("link")
        source_type = doc.metadata.get("source_type", "blog")

        if source_type == "help":
            title = doc.metadata.get("title", "")
            if link and title and title not in help_links:
                help_links[title] = link
        else:
            slug = doc.metadata.get("slug", "")
            if link and slug and slug not in blog_links:
                blog_links[slug] = link

    if not blog_links and not help_links:
        return "\n\nNo relevant sources found."

    lines = ["\n\nRelevant sources:"]

    if blog_links:
        lines.append("\nBlog posts:")
        for slug, link in blog_links.items():
            lines.append(f"- {slug}: {link}")

    if help_links:
        lines.append("\nHelp center:")
        for title, link in help_links.items():
            lines.append(f"- {title}: {link}")

    return "\n".join(lines)


def _compute_file_hash(paths: str | List[str]) -> str:
    """Compute combined MD5 hash of file contents."""
    if isinstance(paths, str):
        paths = [paths]

    hasher = hashlib.md5()
    for path in sorted(paths):
        if not Path(path).exists():
            continue
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _get_stored_hash(hash_file: str) -> Optional[str]:
    """Read stored hash from file, if it exists."""
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read().strip()
    return None


def _store_hash(hash_file: str, hash_value: str) -> None:
    """Store hash to file."""
    with open(hash_file, "w") as f:
        f.write(hash_value)


def build_vector_store(
    input_paths: str | List[str] = DEFAULT_INPUTS,
    embedding_model: str = "text-embedding-3-large",
    limit: Optional[int] = None,
    persist: bool = True,
    chroma_dir: str = CHROMA_DIR,
    hash_file: str = HASH_FILE,
) -> Chroma | InMemoryVectorStore:
    """
    Build or load a vector store.

    If persist=True (default), uses ChromaDB with hash-based invalidation:
    - If chroma_dir exists and hash matches input files, loads from disk
    - Otherwise, rebuilds embeddings and persists to disk

    If persist=False or limit is set, uses InMemoryVectorStore (no caching).
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Use in-memory store if limit is set (testing mode) or persist disabled
    if limit is not None or not persist:
        chunks = iter_chunks(input_paths)
        documents = build_documents(chunks, limit=limit)
        if not documents:
            raise RuntimeError("No documents found. Check input file contents.")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=documents)
        return vector_store

    # Persistent mode with ChromaDB
    current_hash = _compute_file_hash(input_paths)
    stored_hash = _get_stored_hash(hash_file)
    chroma_exists = os.path.exists(chroma_dir) and os.listdir(chroma_dir)

    if chroma_exists and stored_hash == current_hash:
        # Cache hit: load from disk
        print(f"Loading cached embeddings from {chroma_dir}...")
        return Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )

    # Cache miss: rebuild
    if chroma_exists:
        print("Input files changed. Rebuilding embeddings...")
        shutil.rmtree(chroma_dir)
    else:
        print("Building embeddings (first run)...")

    chunks = iter_chunks(input_paths)
    documents = build_documents(chunks, limit=limit)
    if not documents:
        raise RuntimeError("No documents found. Check input file contents.")

    print(f"  Built {len(documents)} documents, computing embeddings...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )

    _store_hash(hash_file, current_hash)
    print(f"  Cached embeddings to {chroma_dir}")

    return vector_store
