"""Build and update the FAISS knowledge index from RAG/docs."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import faiss
import nltk
import numpy as np

from .chunker import load_and_chunk
from .embedder import BGEEmbedder


class FaissIndexer:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, vectors, texts):
        self.index.add(np.array(vectors).astype("float32"))
        self.texts.extend(texts)

    def save(self, index_path: str, text_path: str):
        faiss.write_index(self.index, index_path)
        with open(text_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path: str, text_path: str):
        self.index = faiss.read_index(index_path)
        with open(text_path, "rb") as f:
            self.texts = pickle.load(f)

    def reset(self):
        self.index = faiss.IndexFlatIP(self.index.d)
        self.texts = []


def ensure_nltk_punkt() -> None:
    """Download NLTK sentence tokenizers if they are missing (needed for PDF chunking)."""
    resources = ("tokenizers/punkt", "tokenizers/punkt_tab")
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            package = resource.rsplit("/", 1)[-1]
            print(f"Downloading NLTK resource: {package}")
            nltk.download(package, quiet=True)


def _resolve_rag_paths(docs_dir, index_path, text_path, registry_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return (
        os.path.join(current_dir, docs_dir),
        os.path.join(current_dir, index_path),
        os.path.join(current_dir, text_path),
        os.path.join(current_dir, registry_path),
    )


def update_index(
    docs_dir="docs",
    index_path="faiss.index",
    text_path="chunks.pkl",
    registry_path="registered_files.json",
    embedder=None,
):
    """Update the FAISS index by processing new or modified files in the docs directory."""
    ensure_nltk_punkt()
    docs_dir, index_path, text_path, registry_path = _resolve_rag_paths(
        docs_dir, index_path, text_path, registry_path
    )
    if embedder is None:
        embedder = BGEEmbedder()

    indexer = None
    if os.path.exists(index_path) and os.path.exists(text_path):
        existing = FaissIndexer(dim=1024)
        existing.load(index_path, text_path)
        indexer = existing

    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            registered_files = json.load(f)
    else:
        registered_files = {}

    added_chunks = 0
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if not os.path.isfile(filepath) or not (filepath.endswith(".pdf") or filepath.endswith(".txt")):
            continue

        last_modified = os.path.getmtime(filepath)
        if filepath in registered_files and registered_files[filepath] >= last_modified:
            continue

        print(f"Processing new file: {filepath}")
        chunks = load_and_chunk(filepath)
        if not chunks:
            print(f"Skipping empty file: {filepath}")
            continue

        vectors = embedder.encode(chunks)
        if indexer is None:
            indexer = FaissIndexer(dim=len(vectors[0]))
        indexer.add(vectors, chunks)
        registered_files[filepath] = last_modified
        added_chunks += len(chunks)

    if indexer is None:
        indexer = FaissIndexer(dim=1024)

    indexer.save(index_path, text_path)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registered_files, f, indent=2)

    print(f"Index update complete. Added {added_chunks} chunks. Total chunks: {len(indexer.texts)}")


def rebuild_index(
    docs_dir="docs",
    index_path="faiss.index",
    text_path="chunks.pkl",
    registry_path="registered_files.json",
    embedder=None,
):
    """Delete the existing index and registry, then re-embed every document."""
    _, resolved_index, resolved_texts, resolved_registry = _resolve_rag_paths(
        docs_dir, index_path, text_path, registry_path
    )
    for path in (resolved_index, resolved_texts, resolved_registry):
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")
    update_index(
        docs_dir=docs_dir,
        index_path=index_path,
        text_path=text_path,
        registry_path=registry_path,
        embedder=embedder,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or update the EEGAgent FAISS knowledge index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reset the index and re-embed every file in RAG/docs.",
    )
    args = parser.parse_args()
    if args.rebuild:
        rebuild_index()
    else:
        update_index()
