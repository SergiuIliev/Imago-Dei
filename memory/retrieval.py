"""
Hybrid Retrieval Engine for Personal AI Time Capsule.
Merges BM25 full-text keyword retrieval (FTS5) and dense semantic vector similarity
using Reciprocal Rank Fusion (RRF), with source attribution and evidence thresholding.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from core.models import MemoryItem, MediaSource, TranscriptSegment, VerificationStatus
from memory.database import DatabaseManager
from memory.vector_store import VectorStore


class RetrievedEvidence:
    """Encapsulates a retrieved memory item alongside its source attribution and relevance score."""

    def __init__(
        self,
        memory: MemoryItem,
        score: float,
        retrieval_method: str, # "keyword", "semantic", "hybrid_rrf"
        media_source: Optional[MediaSource] = None,
        transcript_segment: Optional[TranscriptSegment] = None
    ):
        self.memory = memory
        self.score = score
        self.retrieval_method = retrieval_method
        self.media_source = media_source
        self.transcript_segment = transcript_segment

    def format_citation(self) -> str:
        """Generates a transparent citation string detailing provenance and verification status."""
        status_label = f"[{self.memory.verification_status.value.upper()}]"
        date_str = self.memory.date_occurred or "Date Unknown"
        time_range = ""
        if self.memory.start_seconds is not None and self.memory.end_seconds is not None:
            m_start, s_start = divmod(int(self.memory.start_seconds), 60)
            m_end, s_end = divmod(int(self.memory.end_seconds), 60)
            time_range = f" ({m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d})"

        source_info = f"Source ID: {self.memory.source_id[:8]}" if self.memory.source_id else "Standalone Archive Entry"
        return f"{status_label} {date_str}{time_range} | {source_info} | Confidence: {self.memory.confidence:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "citation": self.format_citation(),
            "media_source": self.media_source.to_dict() if self.media_source else None,
            "transcript_segment": self.transcript_segment.to_dict() if self.transcript_segment else None
        }


class HybridRetrievalProvider:
    """Orchestrates multi-modal retrieval across keyword FTS5 and vector indices."""

    def __init__(
        self,
        db: DatabaseManager,
        vector_store: VectorStore,
        rrf_k: int = 60,
        fts_weight: float = 0.5,
        vector_weight: float = 0.5,
        min_evidence_threshold: float = 0.005
    ):
        self.db = db
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight
        self.min_evidence_threshold = min_evidence_threshold

    def _matches_era(self, date_occurred: Optional[str], era_filter: Optional[str]) -> bool:
        if not era_filter:
            return True
        if not date_occurred:
            return True
        # Direct substring match
        if era_filter.lower() in date_occurred.lower():
            return True
        # Year range extraction (e.g. "2014-2018", "2023-Present")
        import re
        era_years = re.findall(r'\b(19\d\d|20\d\d)\b', era_filter)
        date_years = re.findall(r'\b(19\d\d|20\d\d)\b', date_occurred)
        if era_years and date_years:
            d_year = int(date_years[0])
            start_yr = int(era_years[0])
            end_yr = int(era_years[1]) if len(era_years) > 1 else (9999 if "present" in era_filter.lower() else start_yr)
            return start_yr <= d_year <= end_yr
        return True

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        status_filter: Optional[str] = None,
        era_filter: Optional[str] = None
    ) -> Tuple[List[RetrievedEvidence], bool]:
        """
        Executes hybrid retrieval using Reciprocal Rank Fusion.
        Returns:
            Tuple of (List[RetrievedEvidence], has_sufficient_evidence: bool)
        """
        # 1. BM25 Keyword Search
        keyword_results = self.db.search_keyword(query, limit=limit * 3)

        # 2. Dense Semantic Vector Search
        vector_results = self.vector_store.query_similar(query, n_results=limit * 3, filter_status=status_filter)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_score(d) = sum( weight / (k + rank) )
        rrf_scores: Dict[str, float] = {}
        memory_map: Dict[str, MemoryItem] = {}

        # Process Keyword ranks
        for rank, (mem_item, raw_score) in enumerate(keyword_results):
            if status_filter and mem_item.verification_status.value != status_filter:
                continue
            if not self._matches_era(mem_item.date_occurred, era_filter):
                continue
            memory_map[mem_item.id] = mem_item
            rrf_scores[mem_item.id] = rrf_scores.get(mem_item.id, 0.0) + (self.fts_weight / (self.rrf_k + rank + 1))

        # Process Vector ranks
        for rank, (mem_id, similarity, meta) in enumerate(vector_results):
            # Only consider vector matches if similarity >= 0.65 or if already matched by keyword
            if similarity < 0.65 and mem_id not in memory_map:
                continue

            if mem_id not in memory_map:
                item = self.db.get_memory(mem_id)
                if not item:
                    continue
                if not self._matches_era(item.date_occurred, era_filter):
                    continue
                memory_map[mem_id] = item
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + (self.vector_weight / (self.rrf_k + rank + 1))

        if not rrf_scores:
            return ([], False)

        # Sort candidate memories by RRF score descending
        ranked_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        max_score = ranked_candidates[0][1] if ranked_candidates else 0.0
        has_sufficient_evidence = bool(ranked_candidates) and (max_score >= self.min_evidence_threshold)

        evidence_list: List[RetrievedEvidence] = []
        for mem_id, score in ranked_candidates:
            mem = memory_map[mem_id]
            media_src = None
            if mem.source_id:
                media_src = self.db.get_media_by_id(mem.source_id)

            evidence_list.append(RetrievedEvidence(
                memory=mem,
                score=score,
                retrieval_method="hybrid_rrf",
                media_source=media_src
            ))

        return (evidence_list, has_sufficient_evidence)
