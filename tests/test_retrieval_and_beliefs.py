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


if __name__ == "__main__":
    unittest.main()
