"""
RAG Helper for ZenRube Embeddings

This module provides convenient RAG (Retrieval-Augmented Generation) utilities
for experts to perform semantic search and context retrieval.

These utilities are safe to call from ZenRube MCP tools to perform semantic search,
RAG, and context retrieval.

Author: ZenRube Core Engineer
"""

import logging
from typing import List, Dict, Any, Optional

from .client import embed_text
from .index import search

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(prompt: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant text chunks from the embeddings index for a given prompt.

    This function embeds the prompt and searches the specified namespace for
    semantically similar content. Useful for RAG applications where experts
    need contextual information.

    Args:
        prompt: The query prompt to find relevant chunks for
        namespace: The namespace to search in (e.g., "systems_architect", "docs")
        top_k: Maximum number of chunks to retrieve

    Returns:
        List of relevant chunks with text, metadata, and similarity scores.
        Each chunk dict contains:
        - text: The original text chunk
        - metadata: Associated metadata
        - score: Similarity score (0-1, higher is more similar)

    Raises:
        RuntimeError: If embeddings are not available or search fails
    """
    try:
        # Embed the prompt
        query_vector = embed_text(prompt)
        logger.debug(f"Embedded prompt for RAG search in namespace '{namespace}'")

        # Search the index
        results = search(query_vector, namespace=namespace, top_k=top_k)

        # Format results for RAG use
        chunks = []
        for result in results:
            chunks.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "score": result["score"]
            })

        logger.info(f"RAG retrieved {len(chunks)} relevant chunks from namespace '{namespace}' for prompt: {prompt[:50]}...")
        return chunks

    except Exception as e:
        logger.error(f"Failed to retrieve relevant chunks: {e}")
        raise RuntimeError(f"RAG retrieval failed: {e}")


def format_chunks_for_context(chunks: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
    """
    Format retrieved chunks into a context string suitable for LLM prompts.

    Args:
        chunks: List of chunks from retrieve_relevant_chunks
        max_chars: Optional maximum characters to include (truncates if exceeded)

    Returns:
        Formatted context string
    """
    if not chunks:
        return ""

    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_text = f"[{i}] {chunk['text']}"
        if max_chars and total_chars + len(chunk_text) > max_chars:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n\n".join(context_parts)

    if max_chars and total_chars > max_chars:
        context = context[:max_chars] + "..."

    return context


def build_rag_prompt(base_prompt: str, context_chunks: List[Dict[str, Any]],
                     context_label: str = "Context") -> str:
    """
    Build a complete RAG prompt by combining base prompt with retrieved context.

    Args:
        base_prompt: The original prompt/question
        context_chunks: Retrieved relevant chunks
        context_label: Label for the context section

    Returns:
        Complete RAG prompt with context
    """
    if not context_chunks:
        return base_prompt

    context = format_chunks_for_context(context_chunks)
    if not context:
        return base_prompt

    rag_prompt = f"""{context_label}:
{context}

Question: {base_prompt}

Answer:"""

    return rag_prompt