"""
Unit and Integration Tests for Persona Grounding, Anti-Roleplay Directives,
and Multi-Turn Conversational Session Memory.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import (
    MemoryItem,
    TemporalBelief,
    VerificationStatus,
    PersonaSnapshot,
    ExpressiveDeliveryParams,
    EmotionCategory,
    ChatSession,
    ChatMessage
)
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from memory.retrieval import HybridRetrievalProvider
from core.conversation import ConversationEngine, sanitize_persona_text, is_followup_query
from core.personality import PersonalityProfile
from providers.llm.mock_provider import MockLLMProvider


class TestPersonaAndSessionMemory(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_session.db")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")

        self.db = DatabaseManager(self.db_path)
        self.vector_store = VectorStore(self.chroma_dir)
        self.retrieval = HybridRetrievalProvider(self.db, self.vector_store)
        self.mock_llm = MockLLMProvider()
        self.personality = PersonalityProfile()
        self.engine = ConversationEngine(self.retrieval, self.db, self.mock_llm, self.personality)

        from core.models import MediaSource

        source_id = "3791e8c7-4c75-446a-82b3-641eebc98d3f"
        media = MediaSource(
            id=source_id,
            file_path="/tmp/vlog.mp4",
            file_hash="test_hash_3791",
            file_size_bytes=1024,
            duration_seconds=60.0
        )
        self.db.upsert_media(media)

        # Seed sample memories
        self.m1 = MemoryItem(
            id="mem-compounding-1",
            content="I want to look in the future and ask what could happen if it all goes well. What is possible in 20 years and how much better I can be at what I do if I continue to improve each day.",
            summary="20-year compounding vision",
            date_occurred="2026-09-06",
            source_id=source_id,
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        )
        self.m2 = MemoryItem(
            id="mem-failure-1",
            content="The way I deal with failure is that I won't succeed if I don't fail first. Setbacks and pressure are the forge of progress.",
            summary="Failure as forge of progress",
            date_occurred="2026-09-06",
            source_id=source_id,
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        )
        self.db.add_memory(self.m1)
        self.db.add_memory(self.m2)
        self.vector_store.add_batch_memory_embeddings([self.m1, self.m2])

        # Seed temporal beliefs
        self.b_early = TemporalBelief(
            id="belief-fail-early",
            topic="Failure and Resilience",
            belief_statement="Avoid failure through cautious planning.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            active=False,
            reason_for_change="Slowed bold learning."
        )
        self.b_present = TemporalBelief(
            id="belief-fail-present",
            topic="Failure and Resilience",
            belief_statement="I won't succeed if I don't fail first. Setbacks are the forge of progress.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            active=True
        )
        self.db.add_belief(self.b_early)
        self.db.add_belief(self.b_present)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_system_prompt_anti_roleplay_invariants(self):
        """Verifies system prompt strictly instructs LLM to speak directly as Sergiu without acting cues."""
        prompt = self.personality.build_system_prompt(
            era=self.personality.get_snapshot("present"),
            evidence_texts=["Some evidence"],
            has_sufficient_evidence=True
        )

        # Must explicitly assert first-person identity
        self.assertIn("You are Sergiu", prompt)
        self.assertIn("Always speak directly as Sergiu in natural first-person", prompt)

        # Must forbid AI self-referrals
        self.assertIn("NEVER refer to yourself as \"a digital being\"", prompt)
        self.assertIn("an AI representation", prompt)

        # Must strictly forbid theatrical acting directions
        self.assertIn("ABSOLUTELY NO THEATRICAL STAGE DIRECTIONS", prompt)
        self.assertIn("(smirks)", prompt)
        self.assertIn("(chuckles)", prompt)
        self.assertIn("(pauses for a moment)", prompt)

    def test_sanitize_persona_text_eliminates_stage_directions(self):
        """Verifies that parenthetical/asterisk acting cues and AI identity preambles are removed."""
        raw_responses = [
            ("(smirks) Well, looking 20 years into the future...", "Well, looking 20 years into the future..."),
            ("(chuckles) That was an interesting experience.", "That was an interesting experience."),
            ("I decided to try again. (pauses for a moment) It took three days.", "I decided to try again. It took three days."),
            ("*smiles* Yes, Constantin Brâncuși's quote resonated deeply with me.", "Yes, Constantin Brâncuși's quote resonated deeply with me."),
            ("*sighs* It wasn't easy at first.", "It wasn't easy at first."),
            ("As a digital being, I remember the 2026 project.", "I remember the 2026 project."),
            ("As an AI representation of Sergiu, I value simplicity.", "I value simplicity.")
        ]
        for raw, expected in raw_responses:
            cleaned = sanitize_persona_text(raw)
            self.assertEqual(cleaned, expected)

    def test_followup_query_detection(self):
        """Verifies coreference and follow-up cues are accurately detected."""
        self.assertTrue(is_followup_query("Why did you feel that way?"))
        self.assertTrue(is_followup_query("When did that happen?"))
        self.assertTrue(is_followup_query("Tell me more about the second point"))
        self.assertTrue(is_followup_query("Can you elaborate?"))
        self.assertTrue(is_followup_query("Why is that?"))
        self.assertTrue(is_followup_query("And then what happened?"))
        self.assertTrue(is_followup_query("How did that work?"))

        # Non follow-up standalone queries
        self.assertFalse(is_followup_query("What were you working on regarding smart home in 2026?"))
        self.assertFalse(is_followup_query("How do you practice French vocabulary?"))

    def test_database_chat_sessions_crud(self):
        """Verifies database methods for creating sessions, appending messages, and retrieving history."""
        session_id = self.db.create_session(title="Vision Discussion")
        self.assertIsNotNone(session_id)

        # Retrieve session
        sess = self.db.get_session(session_id)
        self.assertIsNotNone(sess)
        self.assertEqual(sess["title"], "Vision Discussion")

        # Append messages
        m1_id = self.db.append_message(
            session_id=session_id,
            role="user",
            content="What is your 20-year compounding vision?"
        )
        m2_id = self.db.append_message(
            session_id=session_id,
            role="assistant",
            content="Looking 20 years out, I ask what happens if it all goes well.",
            evidence=[{"source_id": "3791e8c7"}]
        )
        self.assertIsNotNone(m1_id)
        self.assertIsNotNone(m2_id)

        # Retrieve messages in chronological order
        msgs = self.db.get_session_messages(session_id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "What is your 20-year compounding vision?")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["evidence"][0]["source_id"], "3791e8c7")

        # Verify session listing
        sessions = self.db.list_sessions()
        self.assertGreaterEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], session_id)

    def test_multi_turn_conversation_persistence_and_coreference_expansion(self):
        """Verifies multi-turn dialogue persistence across 2 sequential turns with coreference resolution."""
        session_id = "test-session-multi-turn-01"

        # Turn 1: Inquire about 20-year compounding vision
        self.mock_llm.canned_response = "Around 2026-09-06, I want to look in the future and ask what could happen if it all goes well."
        turn1 = self.engine.process_turn(
            query="What is your 20-year compounding vision?",
            session_id=session_id
        )
        self.assertTrue(turn1.has_sufficient_evidence)
        self.assertEqual(turn1.session_id, session_id)

        # Verify DB has 2 messages (user + assistant)
        history_after_turn1 = self.db.get_session_messages(session_id)
        self.assertEqual(len(history_after_turn1), 2)
        self.assertEqual(history_after_turn1[0]["role"], "user")
        self.assertEqual(history_after_turn1[1]["role"], "assistant")

        # Turn 2: Follow-up question with coreference "Why did you feel that way?"
        # The engine must expand the query with Turn 1 context so retrieval finds the grounding evidence
        self.mock_llm.canned_response = "(smirks) I felt that way because continuous daily improvement compounds over time."
        turn2 = self.engine.process_turn(
            query="Why did you feel that way?",
            session_id=session_id
        )

        # Stage direction '(smirks)' must be sanitized away
        self.assertNotIn("(smirks)", turn2.text)
        self.assertIn("I felt that way because continuous daily improvement", turn2.text)

        # Retrieval query should have been expanded to include previous context
        self.assertIn("What is your 20-year compounding vision?", turn2.metadata.get("retrieval_query", ""))
        self.assertTrue(turn2.has_sufficient_evidence)

        # Verify DB now has 4 messages in chronological order
        history_after_turn2 = self.db.get_session_messages(session_id)
        self.assertEqual(len(history_after_turn2), 4)
        self.assertEqual(history_after_turn2[2]["role"], "user")
        self.assertEqual(history_after_turn2[2]["content"], "Why did you feel that way?")
        self.assertEqual(history_after_turn2[3]["role"], "assistant")
        self.assertNotIn("(smirks)", history_after_turn2[3]["content"])

    def test_temporal_beliefs_fts5_and_topic_normalization(self):
        """Verifies enhanced FTS5 matching on topic and belief statements."""
        # Query topic with lowercase and punctuation
        beliefs_fail = self.db.get_beliefs_for_topic("failure & resilience")
        self.assertEqual(len(beliefs_fail), 2)
        self.assertEqual(beliefs_fail[0].era_label, "Early 20s (2014-2018)")
        self.assertEqual(beliefs_fail[1].era_label, "Present Day (2023-Present)")

        # Query by keyword inside belief statement
        beliefs_forge = self.db.get_beliefs_for_topic("forge of progress")
        self.assertGreaterEqual(len(beliefs_forge), 1)
        self.assertTrue(any("forge" in b.belief_statement.lower() for b in beliefs_forge))

    def test_standardized_fallback_bridge(self):
        """Verifies that queries with low retrieval confidence trigger the standardized fallback prefix."""
        self.mock_llm.canned_response = "I am deeply driven by simplicity and continuous compounding."
        resp = self.engine.process_turn(
            query="What is your favorite cooking recipe from childhood?"
        )
        self.assertFalse(resp.has_sufficient_evidence)
        expected_prefix = (
            "Memory not found. The closest memories are below. "
            "Based on fine-tuning, I can provide the following answer:"
        )
        self.assertIn(expected_prefix, resp.text)
        self.assertIn("simplicity and continuous compounding", resp.text)


if __name__ == "__main__":
    unittest.main()
