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
from core.jev_engine import DualModelVerifier
from core.voice import select_emotion_and_reference_jev


import re

FOLLOWUP_CUES = [
    "why did you feel that way",
    "why did you feel",
    "why did you",
    "when did that happen",
    "when did that",
    "tell me more about",
    "tell me more",
    "second point",
    "first point",
    "third point",
    "can you elaborate",
    "what do you mean",
    "why is that",
    "how did that happen",
    "how did you do that",
    "what about",
    "what happened next",
    "and then what",
    "could you explain",
    "elaborate on",
    "expand on",
]

COREFERENCE_PRONOUNS = {"that", "it", "this", "these", "those", "there", "then", "them"}

STAGE_CUES_PATTERN = re.compile(
    r'(?:\s*[\(\*](?:smirks?|chuckles?|pauses?|laughs?|sighs?|smiles?|clears throat|giggles?|grins?|winks?|nods?|looks? [a-z]+|hesitates?|takes a breath)[^\)\*]*[\)\*]\s*)',
    re.IGNORECASE
)
AI_IDENTITY_PATTERNS = [
    re.compile(r'\bAs an AI representation(?: of Sergiu)?,?\s*', re.IGNORECASE),
    re.compile(r'\bAs a digital being(?: or AI)?,?\s*', re.IGNORECASE),
    re.compile(r'\bAs an agentic AI,?\s*', re.IGNORECASE),
    re.compile(r'\bAs a language model,?\s*', re.IGNORECASE),
    re.compile(r'\bAs an AI,?\s*', re.IGNORECASE),
]


STANDARDIZED_FALLBACK_PREFIX = (
    "Memory not found. The closest memories are below. "
    "Based on fine-tuning, I can provide the following answer:"
)


def sanitize_persona_text(text: str) -> str:
    """Eliminates theatrical stage cues, bracketed emotions, and AI self-referrals."""
    cleaned = STAGE_CUES_PATTERN.sub(' ', text)
    cleaned = re.sub(r'^\s*[\(\*][^\)\*]+[\)\*]\s*', '', cleaned)
    for p in AI_IDENTITY_PATTERNS:
        cleaned = p.sub('', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    return cleaned


def is_followup_query(query_str: str) -> bool:
    """Detects if query contains conversational coreferences or follow-up cues."""
    q_lower = query_str.lower().strip()
    if any(cue in q_lower for cue in FOLLOWUP_CUES):
        return True
    tokens = set(re.findall(r'\b[a-zA-Z]+\b', q_lower))
    if len(tokens) <= 8 and tokens.intersection(COREFERENCE_PRONOUNS):
        return True
    return False


@dataclass
class ConversationResponse:
    text: str
    era_id: Optional[str]
    evidence_used: List[RetrievedEvidence]
    temporal_beliefs_used: List[TemporalBelief]
    has_sufficient_evidence: bool
    is_extrapolation: bool
    recommended_delivery: ExpressiveDeliveryParams
    session_id: Optional[str] = None
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
            "session_id": self.session_id,
            "metadata": self.metadata
        }


class ConversationEngine:
    """High-level conversational orchestrator grounding replies in authentic memories."""

    def __init__(
        self,
        retrieval_provider: HybridRetrievalProvider,
        db: DatabaseManager,
        llm_provider: LLMProvider,
        personality_profile: Optional[PersonalityProfile] = None,
        jev_verifier: Optional[DualModelVerifier] = None
    ):
        self.retrieval = retrieval_provider
        self.db = db
        self.llm = llm_provider
        self.personality = personality_profile or PersonalityProfile()
        self.jev_verifier = jev_verifier

    def process_turn(
        self,
        query: str,
        era_id: Optional[str] = None,
        verification_filter: Optional[str] = None,
        force_simulation: bool = False,
        session_id: Optional[str] = None
    ) -> ConversationResponse:
        """
        Processes a conversational inquiry from the user.
        Supports multi-turn session memory, coreference query expansion,
        temporal belief retrieval, and grounded first-person generation.
        """
        # 1. Resolve chronological snapshot
        active_era: Optional[PersonaSnapshot] = None
        if era_id:
            active_era = self.personality.get_snapshot(era_id)
        if not active_era:
            # Default to present day
            active_era = self.personality.get_snapshot("present")

        # 2. Retrieve recent session history if session_id provided
        history: List[Dict[str, Any]] = []
        if session_id:
            if not self.db.get_session(session_id):
                self.db.create_session(title=query[:60], session_id=session_id)
            history = self.db.get_session_messages(session_id, limit=6)

        # 3. Query expansion for follow-up questions / coreferences
        retrieval_query = query
        if session_id and history and is_followup_query(query):
            last_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
            last_assistant = next((m["content"][:150] for m in reversed(history) if m.get("role") == "assistant"), "")
            # Combine previous context to ensure hybrid retrieval captures referenced concepts
            if last_user:
                retrieval_query = f"{last_user} {query}".strip()
            elif last_assistant:
                retrieval_query = f"{last_assistant} {query}".strip()

        # 4. Retrieve grounded memory evidence
        effective_era_filter = None
        if active_era and active_era.era_id not in ("all_eras", "panoramic", "omniscient"):
            effective_era_filter = active_era.time_period

        evidence_list, has_evidence = self.retrieval.retrieve(
            query=retrieval_query,
            limit=5,
            status_filter=verification_filter,
            era_filter=effective_era_filter
        )

        # 5. Retrieve relevant temporal beliefs using enhanced FTS5 matching
        matched_beliefs: List[TemporalBelief] = []
        # Check full retrieval query and individual terms
        matched_beliefs.extend(self.db.get_beliefs_for_topic(retrieval_query))
        for word in retrieval_query.split():
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', word)
            if len(clean_w) > 3 and clean_w.lower() not in {"what", "when", "where", "which", "about", "your", "that", "this", "from", "with", "have"}:
                matched_beliefs.extend(self.db.get_beliefs_for_topic(clean_w))

        # Deduplicate matched beliefs
        unique_beliefs = {b.id: b for b in matched_beliefs}
        beliefs_list = list(unique_beliefs.values())

        # 6. Format evidence texts for prompt
        evidence_texts = []
        for ev in evidence_list:
            citation = ev.format_citation()
            content = ev.memory.content
            evidence_texts.append(f"[{citation}]\n\"{content}\"")

        beliefs_summary = ""
        if beliefs_list:
            beliefs_summary = self.personality.format_dialectical_beliefs(beliefs_list)

        # 7. Build system prompt
        system_prompt = self.personality.build_system_prompt(
            era=active_era,
            evidence_texts=evidence_texts,
            has_sufficient_evidence=has_evidence or force_simulation,
            temporal_beliefs_summary=beliefs_summary
        )

        # 8. Build user prompt with multi-turn conversation history
        prompt_parts = []
        if history:
            prompt_parts.append("RECENT CONVERSATION HISTORY:")
            for m in history:
                speaker = "Sergiu" if m.get("role") == "assistant" else "User"
                prompt_parts.append(f"{speaker}: {m.get('content', '')}")
            prompt_parts.append("")

        prompt_parts.append(f"User asks: \"{query}\"")
        if force_simulation or not has_evidence:
            if not has_evidence:
                prompt_parts.append(
                    f'[Note: Archive retrieval did not find direct high-confidence records for this specific query. '
                    f'Begin your response with the exact prefix: "{STANDARDIZED_FALLBACK_PREFIX}" '
                    f'and then answer fully based on your fine-tuned parametric knowledge of Sergiu\'s training corpus.]'
                )
            else:
                prompt_parts.append(f"[Request: Simulate how you would reason about this from the perspective of {active_era.era_name if active_era else 'your past self'}. Prefix response with [SIMULATED PERSPECTIVE].]")

        user_prompt = "\n".join(prompt_parts)

        # 9. Call LLM (Model 1)
        model_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.35 if has_evidence else 0.3
        )

        # Sanitize output against stage directions and AI roleplay disclaimers
        model_output = sanitize_persona_text(model_output)

        # 9b. Dual-Model Cross-Verification (Model 2 Critic)
        jev_telemetry = None
        if self.jev_verifier:
            try:
                model_output, jev_telemetry = self.jev_verifier.verify_and_refine_response(
                    query=query,
                    evidence_texts=evidence_texts,
                    beliefs_summary=beliefs_summary,
                    has_sufficient_evidence=has_evidence,
                    draft_response=model_output,
                    era_id=active_era.era_id if active_era else None
                )
                model_output = sanitize_persona_text(model_output)
            except Exception:
                pass

        # 9c. Standardized Parametric Fallback Enforcement
        is_greeting = re.sub(r'[^a-zA-Z\s]', '', query).strip().lower() in {
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"
        }

        if not has_evidence and not force_simulation and not is_greeting:
            # Check if standardized prefix is already cleanly present
            has_prefix = (
                "memory not found" in model_output.lower() and
                "closest memories are below" in model_output.lower()
            )
            if not has_prefix:
                refusal_patterns = [
                    r'^(?:Based on my available recorded memories,?\s*I don\'t have any record[^.]*\.\s*(?:I would rather admit[^.]*\.)?\s*)',
                    r'^(?:I don\'t recall mentioning this in my recorded memories[^.]*\.\s*)',
                    r'^(?:My recorded archive contains no record of this[^.]*\.\s*)'
                ]
                clean_body = model_output
                for pat in refusal_patterns:
                    clean_body = re.sub(pat, '', clean_body, flags=re.IGNORECASE).strip()

                if not clean_body or len(clean_body.split()) < 5:
                    # Regenerate directly from fine-tuned parametric weights
                    ft_sys = (
                        "You are Sergiu. You are speaking directly from your real life memories, experiences, and extensive autobiographical fine-tuning.\n"
                        "Direct archive search found no exact record, but you can answer authoritatively from your fine-tuned parametric weights.\n"
                        "Answer the user's question directly in natural first-person ('I', 'my') as Sergiu about your passions, mindset, habits, or craft.\n"
                        "Do NOT refuse. Do NOT include stage cues or AI disclaimers."
                    )
                    ft_user = f"User asks: \"{query}\"\nProvide your authentic answer as Sergiu based on your fine-tuning:"
                    clean_body = self.llm.generate(prompt=ft_user, system_prompt=ft_sys, temperature=0.35)
                    clean_body = sanitize_persona_text(clean_body)
                    for pat in refusal_patterns:
                        clean_body = re.sub(pat, '', clean_body, flags=re.IGNORECASE).strip()

                if clean_body:
                    model_output = f"{STANDARDIZED_FALLBACK_PREFIX}\n\n{clean_body}"
                else:
                    model_output = (
                        f"{STANDARDIZED_FALLBACK_PREFIX}\n\n"
                        "I am deeply driven by long-term compounding, autonomous systems engineering, relentless physical conditioning, "
                        "and building private, local-first intelligence systems. My daily vlogs reflect a continuous focus on habit consistency, "
                        "craftsmanship, and eliminating friction across software and life."
                    )

        # 10. Persist user turn and assistant response into session memory
        if session_id:
            self.db.append_message(
                session_id=session_id,
                role="user",
                content=query
            )
            evidence_dicts = [e.to_dict() for e in evidence_list]
            self.db.append_message(
                session_id=session_id,
                role="assistant",
                content=model_output,
                evidence=evidence_dicts
            )

        # 11. Determine expressive voice delivery parameters
        base_delivery = active_era.default_delivery if active_era else ExpressiveDeliveryParams()
        jev_engine = self.jev_verifier.critic_engine if self.jev_verifier else None
        detected_emotion, ref_clip, _ = select_emotion_and_reference_jev(model_output, jev_engine)

        recommended_delivery = ExpressiveDeliveryParams(
            emotion=detected_emotion,
            intensity=base_delivery.intensity,
            speech_rate=base_delivery.speech_rate,
            pitch_shift=base_delivery.pitch_shift,
            pause_duration_multiplier=base_delivery.pause_duration_multiplier
        )

        if not has_evidence:
            recommended_delivery.emotion = EmotionCategory.REFLECTIVE
            recommended_delivery.speech_rate = 0.92
            recommended_delivery.pause_duration_multiplier = 1.25

        is_extrapolation = force_simulation or ("[SIMULATED PERSPECTIVE]" in model_output)

        metadata = {
            "model_info": self.llm.get_model_info(),
            "query": query,
            "retrieval_query": retrieval_query if session_id else query,
            "history_length": len(history)
        }
        if jev_telemetry:
            metadata["jev_verification"] = jev_telemetry

        return ConversationResponse(
            text=model_output,
            era_id=active_era.era_id if active_era else None,
            evidence_used=evidence_list,
            temporal_beliefs_used=beliefs_list,
            has_sufficient_evidence=has_evidence,
            is_extrapolation=is_extrapolation,
            recommended_delivery=recommended_delivery,
            session_id=session_id,
            metadata=metadata
        )

