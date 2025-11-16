"""
Embeddings Index for ZenRube

This module provides an in-memory vector index with cosine similarity search.
Stores embeddings in JSON format with atomic writes.

Author: ZenRube Core Engineer
"""

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import numpy for optimized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.info("NumPy not available, using pure Python implementations")

# Index file path
INDEX_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_FILE = INDEX_DIR / "embeddings_index.json"

# Default index structure
DEFAULT_INDEX = {
    "version": 1,
    "next_id": 1,
    "items": []
}


@dataclass
class EmbeddingRecord:
    """
    Represents a single embedding record in the index.
    """
    id: str
    text: str
    vector: List[float]
    namespace: str = "default"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingsIndex:
    """
    In-memory vector index with persistence to JSON.
    """

    def __init__(self):
        self.version = 1
        self.next_id = 1
        self.items: List[Dict[str, Any]] = []
        self._id_to_item: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Load index from JSON file. Creates new index if file doesn't exist.
        """
        try:
            if not INDEX_FILE.exists():
                logger.info(f"Embeddings index file not found at {INDEX_FILE}, creating new index")
                self._create_new_index()
                return

            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.version = data.get("version", 1)
            self.next_id = data.get("next_id", 1)
            self.items = data.get("items", [])

            # Build lookup dict
            self._id_to_item = {item["id"]: item for item in self.items}

            logger.info(f"Loaded embeddings index with {len(self.items)} items")
            self._loaded = True

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted embeddings index file: {e}, recreating")
            self._create_new_index()
        except Exception as e:
            logger.error(f"Failed to load embeddings index: {e}, recreating")
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new empty index."""
        self.version = 1
        self.next_id = 1
        self.items = []
        self._id_to_item = {}
        self._loaded = True
        self.save()

    def save(self) -> None:
        """
        Save index to JSON file atomically.
        """
        try:
            INDEX_DIR.mkdir(exist_ok=True)

            data = {
                "version": self.version,
                "next_id": self.next_id,
                "items": self.items
            }

            # Atomic write using temp file
            with tempfile.NamedTemporaryFile(mode='w', dir=INDEX_DIR, suffix='.tmp', delete=False) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                temp_path = f.name

            # Atomic rename
            os.rename(temp_path, INDEX_FILE)

            logger.debug(f"Saved embeddings index with {len(self.items)} items")

        except Exception as e:
            logger.error(f"Failed to save embeddings index: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)

    def add_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Add items to the index.

        Args:
            items: List of item dicts with keys: text, vector, namespace?, metadata?

        Returns:
            List of assigned IDs
        """
        if not self._loaded:
            self.load()

        assigned_ids = []

        for item in items:
            # Assign ID
            item_id = str(self.next_id)
            self.next_id += 1

            # Create full item
            full_item = {
                "id": item_id,
                "text": item["text"],
                "vector": item["vector"],
                "namespace": item.get("namespace", "default"),
                "metadata": item.get("metadata", {})
            }

            self.items.append(full_item)
            self._id_to_item[item_id] = full_item
            assigned_ids.append(item_id)

        if assigned_ids:
            self.save()
            logger.debug(f"Added {len(assigned_ids)} items to embeddings index")

        return assigned_ids

    def search(self, query_vector: List[float], namespace: Optional[str] = None,
               top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: Query embedding vector
            namespace: Optional namespace filter
            top_k: Number of results to return

        Returns:
            List of results with id, text, score, namespace, metadata
        """
        if not self._loaded:
            self.load()

        if not self.items:
            return []

        # Filter by namespace if specified
        candidates = self.items
        if namespace:
            candidates = [item for item in self.items if item.get("namespace") == namespace]

        if not candidates:
            return []

        # Calculate similarities
        results = []
        for item in candidates:
            try:
                score = cosine_similarity(query_vector, item["vector"])
                if not math.isnan(score):
                    results.append({
                        "id": item["id"],
                        "text": item["text"],
                        "score": score,
                        "namespace": item.get("namespace", "default"),
                        "metadata": item.get("metadata", {})
                    })
            except (ValueError, ZeroDivisionError):
                logger.debug(f"Skipping item {item['id']} due to vector similarity error")
                continue

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a, b: Vectors to compare

    Returns:
        Similarity score between 0 and 1

    Raises:
        ValueError: If vectors have different lengths or are empty
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    if len(a) == 0:
        raise ValueError("Empty vectors")

    if HAS_NUMPY:
        # Use numpy for better performance
        a_np = np.array(a)
        b_np = np.array(b)
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
    else:
        # Pure Python implementation
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# Global index instance
_index_instance: Optional[EmbeddingsIndex] = None


def get_embeddings_index() -> EmbeddingsIndex:
    """
    Get the global embeddings index instance.

    Returns:
        EmbeddingsIndex instance
    """
    global _index_instance
    if _index_instance is None:
        _index_instance = EmbeddingsIndex()
    return _index_instance


def load_index() -> EmbeddingsIndex:
    """
    Load and return the embeddings index.

    Returns:
        The loaded index instance
    """
    index = get_embeddings_index()
    if not index._loaded:
        index.load()
    return index


def save_index() -> None:
    """
    Save the embeddings index.
    """
    get_embeddings_index().save()


def add_items(items: List[Dict[str, Any]]) -> List[str]:
    """
    Add items to the index.

    Args:
        items: List of item dicts

    Returns:
        List of assigned IDs
    """
    return get_embeddings_index().add_items(items)


def search(query_vector: List[float], namespace: Optional[str] = None,
           top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Search the embeddings index.

    Args:
        query_vector: Query vector
        namespace: Optional namespace filter
        top_k: Number of results

    Returns:
        Search results
    """
    return get_embeddings_index().search(query_vector, namespace, top_k)