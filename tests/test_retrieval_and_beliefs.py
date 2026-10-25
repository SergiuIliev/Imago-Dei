"""
Unit and Integration Tests for Retrieval, Historical Belief Preservation, and Persona Grounding.
Verifies temporal belief evolution, conflicting historical perspectives, and honest uncertainty.
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

from core.models import MemoryItem, TemporalBelief, VerificationStatus
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from memory.retrieval import HybridRetrievalProvider
from core.conversation import ConversationEngine
from core.personality import PersonalityProfile
from providers.llm.mock_provider import MockLLMProvider


class TestRetrievalAndBeliefs(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.chroma_dir = os.path.join(self.test_dir, "chroma")

        self.db = DatabaseManager(self.db_path)
        self.vector_store = VectorStore(self.chroma_dir)
        self.retrieval = HybridRetrievalProvider(self.db, self.vector_store)
        self.mock_llm = MockLLMProvider()
        self.personality = PersonalityProfile()
        self.engine = ConversationEngine(self.retrieval, self.db, self.mock_llm, self.personality)

        # Populate sample memories
        self.m1 = MemoryItem(
            id="mem-1",
            content="Early twenties experience with rapid prototyping in 2016.",
            summary="Rapid prototyping in 2016",
            date_occurred="2016-05-10",
            verification_status=VerificationStatus.SYNTHETIC_FIXTURE
        )
        self.m2 = MemoryItem(
            id="mem-2",
            content="Present day focus on autonomous AI agents and local verification in 2024.",
            summary="Autonomous agents in 2024",
            date_occurred="2024-02-15",
            verification_status=VerificationStatus.SYNTHETIC_FIXTURE
        )
        self.db.add_memory(self.m1)
        self.db.add_memory(self.m2)
        self.vector_store.add_batch_memory_embeddings([self.m1, self.m2])

        # Populate conflicting temporal beliefs on same topic
        self.belief_early = TemporalBelief(
            id="b-early",
            topic="Architecture",
            belief_statement="Monolithic single-tier systems are always best for speed.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="b-present",
            active=False,
            reason_for_change="Monoliths became brittle under complexity."
        )
        self.belief_present = TemporalBelief(
            id="b-present",
            topic="Architecture",
            belief_statement="Autonomous local modules with explicit interfaces provide maximum durability.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2023,
            approx_year_end=None,
            supersedes_id="b-early",
            active=True
        )
        self.db.add_belief(self.belief_early)
        self.db.add_belief(self.belief_present)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_historical_belief_preservation(self):
        """Verifies that earlier beliefs remain queryable and are not overwritten by newer ones."""
        beliefs = self.db.get_beliefs_for_topic("Architecture")
        self.assertEqual(len(beliefs), 2)

        early = next(b for b in beliefs if not b.active)
        present = next(b for b in beliefs if b.active)

        self.assertEqual(early.belief_statement, "Monolithic single-tier systems are always best for speed.")
        self.assertEqual(early.superseded_by_id, "b-present")
        self.assertEqual(present.supersedes_id, "b-early")
        self.assertEqual(early.reason_for_change, "Monoliths became brittle under complexity.")

    def test_hybrid_retrieval_relevance(self):
        """Verifies hybrid search returns relevant items and respects date filtering."""
        evidence, has_ev = self.retrieval.retrieve("rapid prototyping", limit=5, era_filter="2014-2018")
        self.assertTrue(has_ev)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].memory.id, "mem-1")

    def test_honest_uncertainty_handling(self):
        """Verifies that an unsupported inquiry returns honest uncertainty without fabricating facts."""
        turn = self.engine.process_turn("What was your favorite cafe in Paris during 2011?")
        self.assertFalse(turn.has_sufficient_evidence)
        self.assertIn("don't have any record", turn.text.lower())

    def test_provider_replacement(self):
        """Verifies LLMProvider can be substituted dynamically via configuration."""
        canned_llm = MockLLMProvider(canned_response="Alternative provider output.")
        new_engine = ConversationEngine(self.retrieval, self.db, canned_llm, self.personality)

        turn = new_engine.process_turn("Hello")
        self.assertEqual(turn.text, "Alternative provider output.")

    def test_honest_uncertainty_gatekeeper_out_of_domain(self):
        """Verifies that out-of-domain queries strictly evaluate to has_sufficient_evidence = False."""
        # Query with non-matching content
        evidence, has_ev = self.retrieval.retrieve("What was my favorite restaurant in Tokyo in 2016?")
        self.assertFalse(has_ev)

        # Another out-of-domain query
        evidence2, has_ev2 = self.retrieval.retrieve("Where did I go for vacation in Iceland in 2015?")
        self.assertFalse(has_ev2)

        # In-domain query with matching keyword and semantic support
        evidence3, has_ev3 = self.retrieval.retrieve("rapid prototyping in 2016")
        self.assertTrue(has_ev3)
        self.assertGreaterEqual(len(evidence3), 1)

    def test_audio_clip_url_binding_and_transcript_segment(self):
        """Verifies that audio_clip_url and transcript_segment are bound into citation evidence."""
        from core.models import MediaSource, TranscriptSegment

        source_id = "test-source-42"
        media = MediaSource(
            id=source_id,
            file_path="/tmp/test.wav",
            file_hash="fakehash123",
            file_size_bytes=1000,
            duration_seconds=120.0
        )
        self.db.upsert_media(media)

        seg = TranscriptSegment(
            id="seg-1",
            source_id=source_id,
            start_seconds=10.0,
            end_seconds=25.0,
            speaker="Sergiu",
            text="Testing audio clip binding."
        )
        self.db.insert_transcript_segments([seg])

        m_clip = MemoryItem(
            id="mem-clip",
            content="Testing audio clip binding in recorded memory.",
            summary="Audio clip memory",
            source_id=source_id,
            start_seconds=10.0,
            end_seconds=25.0,
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        )
        self.db.add_memory(m_clip)
        self.vector_store.add_memory_embedding(m_clip)

        ev_list, has_ev = self.retrieval.retrieve("Testing audio clip binding")
        self.assertTrue(has_ev)
        top = ev_list[0]
        self.assertEqual(top.memory.id, "mem-clip")

        d = top.to_dict()
        expected_url = f"/api/media/{source_id}/clip?start=10.0&end=25.0"
        self.assertEqual(d.get("audio_clip_url"), expected_url)
        self.assertIsNotNone(d.get("transcript_segment"))
        self.assertEqual(d["transcript_segment"]["text"], "Testing audio clip binding.")

    def test_dialectical_temporal_beliefs_schema(self):
        """Verifies supporting_evidence and counter_arguments fields persist and deserialize cleanly."""
        belief_dialectical = TemporalBelief(
            id="b-dialectical",
            topic="Dialectical Systems",
            belief_statement="Systems must be evaluated through opposing thesis and antithesis.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            active=True
        )
        belief_dialectical.add_supporting_evidence(
            quote="Corroborated proof of dialectical synthesis.",
            timestamp_str="2026-08-10",
            start_seconds=38.5,
            end_seconds=79.9,
            source_id="src-001"
        )
        belief_dialectical.add_counter_argument(
            quote="Isolated edge case caveat that must be balanced.",
            timestamp_str="2026-09-01",
            start_seconds=79.4,
            end_seconds=107.8,
            source_id="src-002"
        )

        self.db.add_belief(belief_dialectical)
        retrieved_list = self.db.get_beliefs_for_topic("Dialectical Systems")
        self.assertEqual(len(retrieved_list), 1)
        retrieved = retrieved_list[0]

        self.assertEqual(len(retrieved.supporting_evidence), 1)
        self.assertEqual(retrieved.supporting_evidence[0]["quote"], "Corroborated proof of dialectical synthesis.")
        self.assertEqual(retrieved.supporting_evidence[0]["timestamp_str"], "2026-08-10")
        self.assertEqual(retrieved.supporting_evidence[0]["start_seconds"], 38.5)

        self.assertEqual(len(retrieved.counter_arguments), 1)
        self.assertEqual(retrieved.counter_arguments[0]["quote"], "Isolated edge case caveat that must be balanced.")
        self.assertEqual(retrieved.counter_arguments[0]["timestamp_str"], "2026-09-01")

    def test_corroborated_evidence_superseding_invariant(self):
        """
        Enforces the Invariant:
        A belief MUST NOT be marked superseded by just a single isolated piece of evidence;
        it requires corroborated evidence (at least 2 corroborated pieces of evidence).
        If only 1 outlier piece exists, the belief remains active with an attached caveat/counterargument.
        """
        b = TemporalBelief(
            id="b-invariant-test",
            topic="Empirical Invariant",
            belief_statement="Primary hypothesis held firmly.",
            era_label="2024 Era",
            active=True
        )

        # 1. Attempt superseding with 1 isolated piece of evidence -> MUST FAIL to supersede
        single_evidence = [
            {"quote": "Single anomaly observed once.", "timestamp_str": "2025-01-01"}
        ]
        res1 = b.mark_superseded_with_rule(
            newer_belief_id="b-newer",
            reason="Observed single outlier anomaly.",
            transition_evidence=single_evidence
        )
        self.assertFalse(res1, "Must refuse to supersede belief with only 1 isolated evidence piece")
        self.assertTrue(b.active, "Belief must remain active")
        self.assertIsNone(b.superseded_by_id, "superseded_by_id must remain None")
        self.assertEqual(len(b.counter_arguments), 1, "Outlier must be attached to counter_arguments")
        self.assertIn("Isolated Outlier Caveat", b.counter_arguments[0]["quote"])

        # 2. Attempt superseding with >= 2 corroborated pieces of evidence -> MUST SUCCEED
        corroborated_evidence = [
            {"quote": "First corroborated proof of paradigm breakdown.", "timestamp_str": "2025-06-01"},
            {"quote": "Second independent confirmation under stress.", "timestamp_str": "2025-09-15"}
        ]
        res2 = b.mark_superseded_with_rule(
            newer_belief_id="b-newer",
            reason="Multiple corroborated observations proved old approach brittle.",
            transition_evidence=corroborated_evidence
        )
        self.assertTrue(res2, "Must succeed when corroborated evidence >= 2")
        self.assertFalse(b.active, "Belief must now be marked inactive / superseded")
        self.assertEqual(b.superseded_by_id, "b-newer")
        self.assertGreaterEqual(len(b.supporting_evidence), 2)

    def test_dialectical_system_prompt_formatting(self):
        """Verifies format_dialectical_beliefs outputs timestamps, quotes, and dialectical nuances."""
        b = TemporalBelief(
            id="b-prompt-test",
            topic="Simplicity",
            belief_statement="Simplicity is complexity resolved.",
            era_label="Present Day (2026)",
            active=True,
            supporting_evidence=[{
                "quote": "Constantin Brâncuși quote on simplicity.",
                "timestamp_str": "2026-08-10",
                "start_seconds": 38.53,
                "end_seconds": 79.95,
                "source_id": "b0b95788"
            }],
            counter_arguments=[{
                "quote": "Must first master underlying architectural complexity.",
                "timestamp_str": "2026-08-10",
                "start_seconds": 0.0,
                "end_seconds": 38.53,
                "source_id": "b0b95788"
            }]
        )
        formatted = self.personality.format_dialectical_beliefs([b])
        self.assertIn("DIALECTICAL TEMPORAL BELIEFS", formatted)
        self.assertIn("Simplicity is complexity resolved", formatted)
        self.assertIn("ACTIVE", formatted)
        self.assertIn("Constantin Brâncuși quote", formatted)
        self.assertIn("2026-08-10", formatted)
        self.assertIn("38.53s-79.95s", formatted)
        self.assertIn("Must first master underlying architectural complexity", formatted)


if __name__ == "__main__":
    unittest.main()
