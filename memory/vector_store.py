"""
Vector Embedding Store using ChromaDB with local ONNX all-MiniLM-L6-v2 embeddings.
Provides local-only semantic search, cosine distance ranking, and batch re-indexing.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from core.models import MemoryItem


class VectorStore:
    """Manages persistent semantic embeddings for memories."""

    def __init__(self, persist_directory: str, collection_name: str = "time_capsule_memories"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            # PersistentClient stores vectors locally on disk in persist_directory
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Cosine distance for normalized similarity
            )
            self.is_available = True
        except Exception as e:
            print(f"[VectorStore] Warning: Could not initialize ChromaDB ({e}). Falling back to mock vector store.")
            self.client = None
            self.collection = None
            self.is_available = False
            self._mock_records: Dict[str, Dict[str, Any]] = {}

    def add_memory_embedding(self, memory: MemoryItem):
        self.add_batch_memory_embeddings([memory])

    def add_batch_memory_embeddings(self, memories: List[MemoryItem]):
        if not memories:
            return

        documents = []
        ids = []
        metadatas = []

        for m in memories:
            # Combine summary and content for rich semantic representation
            text = f"{m.summary}\n{m.content}".strip()
            documents.append(text)
            ids.append(m.id)
            metadatas.append({
                "category": m.category,
                "verification_status": m.verification_status.value,
                "date_occurred": m.date_occurred or "unknown",
                "emotional_tone": m.emotional_tone,
                "source_id": m.source_id or ""
            })

        if self.is_available and self.collection is not None:
            self.collection.upsert(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        else:
            for i, m_id in enumerate(ids):
                self._mock_records[m_id] = {
                    "document": documents[i],
                    "metadata": metadatas[i]
                }

    def query_similar(
        self,
        query_text: str,
        n_results: int = 10,
        filter_status: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Queries nearest semantic memories.
        Returns List of tuples: (memory_id, similarity_score, metadata)
        Similarity score is normalized from 0.0 (unrelated) to 1.0 (identical).
        """
        if not self.is_available or self.collection is None:
            # Fallback keyword overlap ranking for mock mode
            results = []
            q_terms = set(query_text.lower().split())
            for m_id, item in self._mock_records.items():
                if filter_status and item["metadata"].get("verification_status") != filter_status:
                    continue
                d_terms = set(item["document"].lower().split())
                overlap = len(q_terms.intersection(d_terms))
                score = overlap / max(len(q_terms), 1)
                results.append((m_id, score, item["metadata"]))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:n_results]

        where_filter = None
        if filter_status:
            where_filter = {"verification_status": filter_status}

        try:
            query_kwargs = {
                "query_texts": [query_text],
                "n_results": n_results
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            response = self.collection.query(**query_kwargs)

            results: List[Tuple[str, float, Dict[str, Any]]] = []
            if response and response["ids"] and len(response["ids"]) > 0:
                ids = response["ids"][0]
                distances = response["distances"][0] if response.get("distances") else [0.0] * len(ids)
                metas = response["metadatas"][0] if response.get("metadatas") else [{}] * len(ids)

                for mem_id, dist, meta in zip(ids, distances, metas):
                    # Chroma cosine distance is in [0, 2]; convert to similarity in [0, 1]
                    similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                    results.append((mem_id, similarity, meta))
            return results
        except Exception as e:
            print(f"[VectorStore] Query error: {e}")
            return []

    def delete_memory_embedding(self, memory_id: str):
        if self.is_available and self.collection is not None:
            try:
                self.collection.delete(ids=[memory_id])
            except Exception as e:
                print(f"[VectorStore] Delete error for {memory_id}: {e}")
        else:
            self._mock_records.pop(memory_id, None)

    def reindex_all(self, memories: List[MemoryItem]):
        """Wipes collection and rebuilds all embeddings, e.g. after model upgrade."""
        if self.is_available and self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self._mock_records.clear()
        self.add_batch_memory_embeddings(memories)
