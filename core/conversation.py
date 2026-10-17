"""
Conversation Engine for Personal AI Time Capsule.
Coordinates hybrid retrieval, temporal belief lookup, chronological persona selection,
and grounded LLM generation with honest uncertainty and source citations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from core.models import (
    MemoryItem,
    TemporalBelief,
    PersonaSnapshot,
    ExpressiveDeliveryParams,
    EmotionCategory
)
from core.personality import PersonalityProfile
from memory.database import DatabaseManager
from memory.retrieval import HybridRetrievalProvider, RetrievedEvidence
from providers.llm.base import LLMProvider


@dataclass
class ConversationResponse:
    text: str
    era_id: Optional[str]
    evidence_used: List[RetrievedEvidence]
    temporal_beliefs_used: List[TemporalBelief]
    has_sufficient_evidence: bool
    is_extrapolation: bool
    recommended_delivery: ExpressiveDeliveryParams
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "era_id": self.era_id,
            "evidence_used": [e.to_dict() for e in self.evidence_used],
            "temporal_beliefs_used": [b.to_dict() for b in self.temporal_beliefs_used],
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "is_extrapolation": self.is_extrapolation,
            "recommended_delivery": self.recommended_delivery.to_dict(),
            "metadata": self.metadata
        }


class ConversationEngine:
    """High-level conversational orchestrator grounding replies in authentic memories."""

    def __init__(
        self,
        retrieval_provider: HybridRetrievalProvider,
        db: DatabaseManager,
        llm_provider: LLMProvider,
        personality_profile: Optional[PersonalityProfile] = None
    ):
        self.retrieval = retrieval_provider
        self.db = db
        self.llm = llm_provider
        self.personality = personality_profile or PersonalityProfile()

    def process_turn(
        self,
        query: str,
        era_id: Optional[str] = None,
        verification_filter: Optional[str] = None,
        force_simulation: bool = False
    ) -> ConversationResponse:
        """
        Processes a single conversational inquiry from the user.
        Retrieves matching evidence, matches chronological persona, queries LLM,
        and constructs delivery instructions.
        """
        # 1. Resolve chronological snapshot
        active_era: Optional[PersonaSnapshot] = None
        if era_id:
            active_era = self.personality.get_snapshot(era_id)
        if not active_era:
            # Default to present day
            active_era = self.personality.get_snapshot("present")

        # 2. Retrieve grounded memory evidence
        evidence_list, has_evidence = self.retrieval.retrieve(
            query=query,
            limit=5,
            status_filter=verification_filter,
            era_filter=active_era.time_period if active_era else None
        )

        # 3. Retrieve relevant temporal beliefs
        # Search topic keywords in beliefs
        matched_beliefs: List[TemporalBelief] = []
        for word in query.split():
            if len(word) > 3:
                matched_beliefs.extend(self.db.get_beliefs_for_topic(word))

        # Deduplicate matched beliefs
        unique_beliefs = {b.id: b for b in matched_beliefs}
        beliefs_list = list(unique_beliefs.values())

        # 4. Format evidence texts for prompt
        evidence_texts = []
        for ev in evidence_list:
            citation = ev.format_citation()
            content = ev.memory.content
            evidence_texts.append(f"[{citation}]\n\"{content}\"")

        beliefs_summary = ""
        if beliefs_list:
            beliefs_summary = "HISTORICAL BELIEFS EVOLUTION ON THIS TOPIC:\n"
            for b in beliefs_list:
                status_str = "Active" if b.active else f"Superseded (Reason: {b.reason_for_change or 'Not specified'})"
                beliefs_summary += f"- [{b.era_label}] {b.topic}: \"{b.belief_statement}\" (Status: {status_str})\n"

        # 5. Build system prompt
        system_prompt = self.personality.build_system_prompt(
            era=active_era,
            evidence_texts=evidence_texts,
            has_sufficient_evidence=has_evidence or force_simulation,
            temporal_beliefs_summary=beliefs_summary
        )

        user_prompt = f"User asks: \"{query}\""
        if force_simulation or not has_evidence:
            if not has_evidence:
                user_prompt += "\n[Note: Archive search returned no direct match. State honest uncertainty clearly.]"
            else:
                user_prompt += f"\n[Request: Simulate how you would reason about this from the perspective of {active_era.era_name if active_era else 'your past self'}. Prefix response with [SIMULATED PERSPECTIVE].]"

        # 6. Call LLM
        model_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.35 if has_evidence else 0.2
        )

        # 7. Determine expressive voice delivery parameters
        base_delivery = active_era.default_delivery if active_era else ExpressiveDeliveryParams()
        # Modulate emotion based on tone of response
        response_lower = model_output.lower()
        recommended_delivery = ExpressiveDeliveryParams(
            emotion=base_delivery.emotion,
            intensity=base_delivery.intensity,
            speech_rate=base_delivery.speech_rate,
            pitch_shift=base_delivery.pitch_shift,
            pause_duration_multiplier=base_delivery.pause_duration_multiplier
        )

        if not has_evidence:
            recommended_delivery.emotion = EmotionCategory.REFLECTIVE
            recommended_delivery.speech_rate = 0.92
            recommended_delivery.pause_duration_multiplier = 1.25
        elif "laugh" in response_lower or "funny" in response_lower:
            recommended_delivery.emotion = EmotionCategory.HUMOROUS
            recommended_delivery.speech_rate = 1.05
        elif "curious" in response_lower or "wonder" in response_lower:
            recommended_delivery.emotion = EmotionCategory.CURIOUS

        is_extrapolation = force_simulation or ("[SIMULATED PERSPECTIVE]" in model_output)

        return ConversationResponse(
            text=model_output,
            era_id=active_era.era_id if active_era else None,
            evidence_used=evidence_list,
            temporal_beliefs_used=beliefs_list,
            has_sufficient_evidence=has_evidence,
            is_extrapolation=is_extrapolation,
            recommended_delivery=recommended_delivery,
            metadata={
                "model_info": self.llm.get_model_info(),
                "query": query
            }
        )
