"""
Hybrid Retrieval Engine for Personal AI Time Capsule.
Merges BM25 full-text keyword retrieval (FTS5) and dense semantic vector similarity
using Reciprocal Rank Fusion (RRF), with source attribution and evidence thresholding.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from core.models import MemoryItem, MediaSource, TranscriptSegment, VerificationStatus
from memory.database import DatabaseManager
from memory.vector_store import VectorStore


STOP_WORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t',
    'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t',
    'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how',
    'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is',
    'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
    'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should',
    'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they',
    'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
    'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s',
    'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re',
    'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
    # Common conversational verbs and auxiliaries
    'go', 'goes', 'going', 'went', 'gone',
    'get', 'gets', 'getting', 'got', 'gotten',
    'take', 'takes', 'taking', 'took', 'taken',
    'make', 'makes', 'making', 'made',
    'see', 'sees', 'seeing', 'saw', 'seen',
    'know', 'knows', 'knew', 'known',
    'think', 'thinks', 'thought',
    'come', 'comes', 'coming', 'came',
    'tell', 'tells', 'telling', 'told',
    'say', 'says', 'saying', 'said'
}


def extract_content_words(text: str) -> Set[str]:
    """Extracts non-stopword alphanumeric content tokens from text."""
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return {t for t in tokens if len(t) > 1 and t not in STOP_WORDS}


def has_lexical_content_overlap(query: str, target_text: str) -> bool:
    """
    Checks if there is non-empty lexical overlap between key query content words
    and the target text, accounting for simple prefixes / stems (e.g. failure/fail).
    Excludes standalone 4-digit years from being the sole overlapping token,
    so that a query about a specific event in a year must match an actual content word,
    not just the year number.
    """
    q_words = extract_content_words(query)
    if not q_words:
        return True

    t_words = extract_content_words(target_text)
    if not t_words:
        return False

    overlapping = set()
    for qw in q_words:
        for tw in t_words:
            if qw == tw:
                overlapping.add(qw)
                break
            elif len(qw) >= 4 and len(tw) >= 4 and (qw.startswith(tw) or tw.startswith(qw)):
                overlapping.add(qw)
                break

    non_year_overlap = {w for w in overlapping if not (w.isdigit() and len(w) == 4)}
    return len(non_year_overlap) > 0 or (len(overlapping) > 0 and len(q_words) == 1)


class RetrievedEvidence:
    """Encapsulates a retrieved memory item alongside its source attribution and relevance score."""

    def __init__(
        self,
        memory: MemoryItem,
        score: float,
        retrieval_method: str, # "keyword", "semantic", "hybrid_rrf", "web_search"
        media_source: Optional[MediaSource] = None,
        transcript_segment: Optional[TranscriptSegment] = None,
        audio_clip_url: Optional[str] = None,
        citation_override: Optional[str] = None
    ):
        self.memory = memory
        self.score = score
        self.retrieval_method = retrieval_method
        self.media_source = media_source
        self.transcript_segment = transcript_segment
        self.citation_override = citation_override

        if audio_clip_url:
            self.audio_clip_url = audio_clip_url
        elif (
            self.memory.source_id
            and self.memory.start_seconds is not None
            and self.memory.end_seconds is not None
        ):
            self.audio_clip_url = f"/api/media/{self.memory.source_id}/clip?start={self.memory.start_seconds}&end={self.memory.end_seconds}"
        else:
            self.audio_clip_url = None

    def format_citation(self) -> str:
        """Generates a transparent citation string detailing provenance and verification status."""
        if getattr(self, "citation_override", None):
            return self.citation_override
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
        audio_clip_url = self.audio_clip_url
        if (
            not audio_clip_url
            and self.memory.source_id
            and self.memory.start_seconds is not None
            and self.memory.end_seconds is not None
        ):
            audio_clip_url = f"/api/media/{self.memory.source_id}/clip?start={self.memory.start_seconds}&end={self.memory.end_seconds}"

        d = {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "citation": self.format_citation(),
            "media_source": self.media_source.to_dict() if self.media_source else None,
            "transcript_segment": self.transcript_segment.to_dict() if self.transcript_segment else None,
            "audio_clip_url": audio_clip_url
        }
        if getattr(self, "citation_override", None):
            d["citation_override"] = self.citation_override
        return d


class HybridRetrievalProvider:
    """Orchestrates multi-modal retrieval across keyword FTS5 and vector indices."""

    def __init__(
        self,
        db: DatabaseManager,
        vector_store: VectorStore,
        rrf_k: int = 60,
        fts_weight: float = 0.5,
        vector_weight: float = 0.5,
        min_evidence_threshold: float = 0.012,
        min_semantic_similarity: float = 0.70,
        jev_engine: Optional[Any] = None,
        internet_engine: Optional[Any] = None,
        rag_mode: str = "airgapped"
    ):
        self.db = db
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.fts_weight = fts_weight
        self.vector_weight = vector_weight
        self.min_evidence_threshold = min_evidence_threshold
        self.min_semantic_similarity = min_semantic_similarity
        self.jev_engine = jev_engine
        self.internet_engine = internet_engine
        self.rag_mode = rag_mode

    def _matches_era(self, date_occurred: Optional[str], era_filter: Optional[str]) -> bool:
        if not era_filter:
            return True
        filter_clean = era_filter.strip().lower()
        if filter_clean in ("all", "all_eras", "all-eras", "panoramic", "omniscient", "none"):
            return True
        if not date_occurred:
            return False

        date_clean = date_occurred.strip().lower()

        # Extract 4-digit years from date_occurred
        date_years = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', date_clean)]

        # Specific handling for "present" / "present day"
        if filter_clean in ("present", "present day", "current", "now"):
            if date_years:
                return date_years[0] >= 2026
            return "present" in date_clean

        # Direct substring match (e.g. "2023-04-22" matches "2023")
        if filter_clean in date_clean:
            return True

        # Extract 4-digit years from era_filter
        era_years = [int(y) for y in re.findall(r'\b(19\d\d|20\d\d)\b', filter_clean)]

        if era_years and date_years:
            d_year = date_years[0]
            if len(era_years) == 1:
                if "present" in filter_clean:
                    return d_year >= era_years[0]
                else:
                    return d_year == era_years[0]
            else:
                return era_years[0] <= d_year <= era_years[1]

        if "present" in filter_clean and date_years:
            return date_years[0] >= 2026

        return False

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        status_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
        rag_mode: Optional[str] = None
    ) -> Tuple[List[RetrievedEvidence], bool]:
        """
        Executes hybrid retrieval using Reciprocal Rank Fusion.
        Optionally augments local archive memories with live external web knowledge
        when rag_mode is 'jev_controlled' or 'open_internet'.
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
        keyword_matched_ids: Set[str] = set()
        vector_matched_ids: Set[str] = set()
        vector_similarities: Dict[str, float] = {}

        # Process Keyword ranks
        for rank, (mem_item, raw_score) in enumerate(keyword_results):
            if status_filter and mem_item.verification_status.value != status_filter:
                continue
            if not self._matches_era(mem_item.date_occurred, era_filter):
                continue
            keyword_matched_ids.add(mem_item.id)
            memory_map[mem_item.id] = mem_item
            rrf_scores[mem_item.id] = rrf_scores.get(mem_item.id, 0.0) + (self.fts_weight / (self.rrf_k + rank + 1))

        # Process Vector ranks
        for rank, (mem_id, similarity, meta) in enumerate(vector_results):
            vector_similarities[mem_id] = similarity
            # Retain vector matches only if similarity >= 0.65
            if similarity < 0.65:
                continue

            if mem_id not in memory_map:
                item = self.db.get_memory(mem_id)
                if not item:
                    continue
                if status_filter and item.verification_status.value != status_filter:
                    continue
                if not self._matches_era(item.date_occurred, era_filter):
                    continue
                memory_map[mem_id] = item
            else:
                item = memory_map[mem_id]
                if not self._matches_era(item.date_occurred, era_filter):
                    continue

            vector_matched_ids.add(mem_id)
            rrf_scores[mem_id] = rrf_scores.get(mem_id, 0.0) + (self.vector_weight / (self.rrf_k + rank + 1))

        ranked_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit] if rrf_scores else []
        if not ranked_candidates:
            effective_rag_mode = rag_mode or self.rag_mode or "airgapped"
            if effective_rag_mode != "airgapped" and self.internet_engine:
                try:
                    web_evidence, _ = self.internet_engine.retrieve_web_evidence(
                        query=query,
                        rag_mode=effective_rag_mode,
                        limit=limit
                    )
                    if web_evidence:
                        return (web_evidence, True)
                except Exception:
                    pass
            return ([], False)

        top_id, top_score = ranked_candidates[0]
        top_mem = memory_map[top_id]
        top_vector_sim = vector_similarities.get(top_id, 0.0)

        # Agreement: candidate matched in both keyword and vector search
        agrees_in_both = (top_id in keyword_matched_ids) and (top_id in vector_matched_ids)

        # Check lexical content overlap between query and top candidate text
        candidate_text = f"{top_mem.summary or ''} {top_mem.content or ''} {' '.join(top_mem.tags or [])}"
        has_overlap = has_lexical_content_overlap(query, candidate_text)

        # Honest Uncertainty Gatekeeper:
        # Require that for has_sufficient_evidence to be True:
        # - Either the top dense semantic vector similarity is >= min_semantic_similarity (0.70), OR
        # - Both keyword (BM25) and semantic search agree on top candidate with composite RRF score >= min_evidence_threshold (0.012), AND
        # - Lexical overlap on key query content words is non-empty.
        is_high_semantic = top_vector_sim >= self.min_semantic_similarity
        is_hybrid_agreement = agrees_in_both and (top_score >= self.min_evidence_threshold)

        has_sufficient_evidence = bool(ranked_candidates) and (is_high_semantic or is_hybrid_agreement) and has_overlap

        # Step 3.3: Fast Noul check on top retrieved chunk if Jev engine is available
        if has_sufficient_evidence and self.jev_engine:
            try:
                is_avail = not hasattr(self.jev_engine, "is_available") or self.jev_engine.is_available()
                if is_avail:
                    top_content = f"{top_mem.summary or ''} {top_mem.content or ''}".strip()
                    noul_res = self.jev_engine.Noul(
                        instructions="Does this memory excerpt contain actual relevance or evidence to help answer the user query?",
                        context=f"Query: {query}\nMemory Excerpt: {top_content}"
                    )
                    noul_val = getattr(noul_res, "noul", noul_res.get("noul", 1.0))
                    if noul_val < 0.5:
                        has_sufficient_evidence = False
            except Exception:
                # Graceful fallback to deterministic threshold
                pass

        evidence_list: List[RetrievedEvidence] = []
        for mem_id, score in ranked_candidates:
            mem = memory_map[mem_id]
            media_src = None
            transcript_seg = None
            if mem.source_id:
                media_src = self.db.get_media_by_id(mem.source_id)
                transcript_seg = self.db.get_transcript_segment_for_range(
                    mem.source_id, mem.start_seconds, mem.end_seconds
                )

            evidence_list.append(RetrievedEvidence(
                memory=mem,
                score=score,
                retrieval_method="hybrid_rrf",
                media_source=media_src,
                transcript_segment=transcript_seg
            ))

        # Augment with live external web retrieval if requested and configured
        effective_rag_mode = rag_mode or self.rag_mode or "airgapped"
        if effective_rag_mode != "airgapped" and self.internet_engine:
            try:
                web_evidence, _ = self.internet_engine.retrieve_web_evidence(
                    query=query,
                    rag_mode=effective_rag_mode,
                    limit=limit
                )
                if web_evidence:
                    # Append web evidence behind local autobiographical memories
                    evidence_list = evidence_list + web_evidence
                    if not has_sufficient_evidence:
                        has_sufficient_evidence = True
            except Exception:
                pass

        return (evidence_list, has_sufficient_evidence)
