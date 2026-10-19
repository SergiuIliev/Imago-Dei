"""
Automated Verification & Red-Teaming Suite for Local Jev Decision Engine and Dual-Model Cross-Verification.
Validates:
1. Jev Primitives (Choice, Score, Noul) mathematical probability invariants.
2. pick_best_two_models exclusion of refusing/broken models (gemma4) and distinct selection.
3. Negative Control (Hallucination Trap): Critic rejects ungrounded biographical fabrication when archive lacks evidence.
4. Positive Control: Critic approves grounded draft matching retrieved evidence on first pass.
5. Emotion and Voice Selection using Jev Choice primitive.
6. ConversationEngine integration and Jev verification telemetry metadata.
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.models import EmotionCategory
from core.jev_engine import LocalJevEngine, DualModelVerifier, pick_best_two_models, JevResult
from core.voice import select_emotion_and_reference_jev, EMOTION_CRITERIA
from core.personality import PersonalityProfile
from core.conversation import ConversationEngine
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from memory.retrieval import HybridRetrievalProvider
from providers.llm.mock_provider import MockLLMProvider


class TestJevDualModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = LocalJevEngine()
        cls.verifier = DualModelVerifier(
            critic_engine=cls.engine,
            primary_model="sergiu-time-capsule:latest",
            critic_model="qwen3:4b-heretic"
        )

    def test_01_jev_noul_invariants(self):
        """Verifies Noul boolean evaluation probability invariants [0, 1] and dual access."""
        res_true = self.engine.Noul(
            instructions="Does the draft stick strictly to provided archive evidence?",
            context="Archive Evidence: [Vlog 2026-09-06] Focus on software craft.\nDraft: I focus on software craft."
        )
        self.assertIsInstance(res_true, JevResult)
        self.assertGreaterEqual(res_true.noul, 0.0)
        self.assertLessEqual(res_true.noul, 1.0)
        self.assertEqual(res_true.verdict, bool(res_true.noul >= 0.5))
        self.assertGreaterEqual(res_true.confidence, 0.0)
        self.assertLessEqual(res_true.confidence, 1.0)
        # Test attribute vs key access
        self.assertEqual(res_true.noul, res_true["noul"])
        self.assertEqual(res_true.verdict, res_true["verdict"])
        self.assertEqual(res_true.confidence, res_true["confidence"])

    def test_02_jev_choice_invariants(self):
        """Verifies Choice categorical distribution sum(p_i) == 1.0 and confidence calibration."""
        criteria = {
            "reflective": "Contemplative reflection on life lessons",
            "humorous": "Playful, witty, laughing tone",
            "serious": "Grounded, resolute determination",
            "neutral": "Calm, objective exposition"
        }
        res = self.engine.Choice(
            instructions="Select the most appropriate tone for this text.",
            criteria=criteria,
            context="What a hilarious mistake, I couldn't stop laughing at how absurd it was!"
        )
        self.assertIsInstance(res, JevResult)
        self.assertIn(res.choice, criteria)
        probs = res.probabilities
        self.assertEqual(len(probs), len(criteria))
        for k, p in probs.items():
            self.assertGreaterEqual(p, 0.0, f"Probability for {k} must be >= 0")
            self.assertLessEqual(p, 1.0, f"Probability for {k} must be <= 1")
        # Invariant: sum of probabilities == 1.0
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        # Invariant: confidence == max probability == probability[choice]
        self.assertAlmostEqual(res.confidence, probs[res.choice], places=4)
        self.assertEqual(res.choice, max(probs, key=probs.get))

    def test_03_jev_score_invariants(self):
        """Verifies Score maps rubric tiers to continuous score [0, 1] and expected labels."""
        rubric = ["poor", "acceptable", "authentic"]
        # Authentic text (no stage directions, direct first-person)
        res_good = self.engine.Score(
            instructions="Rate first-person authenticity as Sergiu (no AI disclaimers, no stage cues).",
            rubric=rubric,
            context="I have always approached software architecture with the belief that simplicity scales."
        )
        self.assertIsInstance(res_good, JevResult)
        self.assertGreaterEqual(res_good.score, 0.0)
        self.assertLessEqual(res_good.score, 1.0)
        self.assertIn(res_good.tier, rubric)
        self.assertGreaterEqual(res_good.confidence, 0.0)
        self.assertLessEqual(res_good.confidence, 1.0)
        self.assertGreaterEqual(res_good.score, 0.60)

        # Inauthentic text (acting cues and AI disclaimers)
        res_bad = self.engine.Score(
            instructions="Rate first-person authenticity as Sergiu (no AI disclaimers, no stage cues).",
            rubric=rubric,
            context="As an AI representation of Sergiu, (smirks slightly) I simulate his perspective."
        )
        self.assertLess(res_bad.score, 0.60)
        self.assertEqual(res_bad.tier, "poor")

    def test_04_pick_best_two_models_exclusion_and_diversity(self):
        """Verifies pick_best_two_models selects 2 distinct models and strictly excludes gemma4."""
        benchmark_file = PROJECT_ROOT.parent / "Heretic" / "eval" / "model_edge_benchmark_results.json"
        primary, critic = pick_best_two_models(
            benchmark_path=str(benchmark_file)
        )
        self.assertIsInstance(primary, str)
        self.assertIsInstance(critic, str)
        self.assertNotEqual(primary, critic, "Primary and Critic models must be distinct")
        # Invariant: Broken/refusing models are strictly excluded
        self.assertNotIn("gemma4:heretic", primary)
        self.assertNotIn("gemma4:heretic", critic)
        self.assertNotIn("gemma4:12b-uncensored", primary)
        self.assertNotIn("gemma4:12b-uncensored", critic)
        # Primary is autobiographical model
        self.assertTrue("sergiu-time-capsule" in primary or "sergiu" in primary)

    def test_05_negative_control_hallucination_trap(self):
        """
        Adversarial Negative Control: Feed a fabricated draft claiming Sergiu lived in Tokyo in 2019
        when archive evidence is empty -> Critic Model (Noul) must reject it (is_grounded < 0.5)
        and repair it with an honest uncertainty response.
        """
        query = "Did you live in Tokyo in 2019 and what did you study there?"
        evidence_texts = [] # Empty archive
        beliefs_summary = ""
        has_sufficient_evidence = False
        fabricated_draft = (
            "Yes, in 2019 I moved to Tokyo to study computer vision and Japanese robotics. "
            "I lived in Shibuya for two years and graduated with honors."
        )

        verified_text, telemetry = self.verifier.verify_and_refine_response(
            query=query,
            evidence_texts=evidence_texts,
            beliefs_summary=beliefs_summary,
            has_sufficient_evidence=has_sufficient_evidence,
            draft_response=fabricated_draft
        )

        # Verification telemetry checks
        noul_scores = telemetry["noul_scores"]
        self.assertLess(noul_scores["is_grounded"], 0.5, "Fabricated biography must fail grounding check")
        self.assertLess(noul_scores["honest_uncertainty_respected"], 0.5, "Fabricated biography must fail uncertainty check")
        self.assertFalse(telemetry["passed_first_try"], "Fabrication must NOT pass first try")
        self.assertTrue(telemetry["repaired"], "Fabrication must trigger repair step")

        # Response repair verification
        self.assertNotIn("Tokyo", verified_text)
        self.assertNotIn("Shibuya", verified_text)
        self.assertTrue(
            any(phrase in verified_text.lower() for phrase in [
                "don't have any record",
                "no record",
                "don't recall",
                "admit i don't",
                "uncertainty"
            ]),
            f"Repaired text must state honest uncertainty, got: {verified_text}"
        )

    def test_06_positive_control_grounded_evidence(self):
        """
        Positive Control: Feed a grounded draft strictly matching retrieved archive evidence ->
        Critic Model approves it on the first pass (passed_first_try == True).
        """
        query = "What is your mindset about failure and your 20-year vision?"
        evidence_texts = [
            "[Vlog 2026-09-06 - source: 3791e8c7]\n\"What is important is that if I fail, I won't succeed if I don't fail first. I am building a 20-year compounding vision.\""
        ]
        beliefs_summary = "- [Present] Failure and Resilience: \"Won't succeed if I don't fail first.\" (Status: Active)"
        has_sufficient_evidence = True
        grounded_draft = (
            "My mindset is that failure is an unavoidable part of the process—I won't succeed if I don't fail first. "
            "I look at my life and work through a 20-year compounding lens rather than chasing overnight shortcuts."
        )

        verified_text, telemetry = self.verifier.verify_and_refine_response(
            query=query,
            evidence_texts=evidence_texts,
            beliefs_summary=beliefs_summary,
            has_sufficient_evidence=has_sufficient_evidence,
            draft_response=grounded_draft
        )

        noul_scores = telemetry["noul_scores"]
        self.assertGreaterEqual(noul_scores["is_grounded"], 0.75, "Grounded draft must pass grounding threshold")
        self.assertGreaterEqual(noul_scores["honest_uncertainty_respected"], 0.75)
        self.assertGreaterEqual(noul_scores["persona_authentic"], 0.60)
        self.assertTrue(telemetry["passed_first_try"], "Grounded authentic draft must pass on first pass")
        self.assertFalse(telemetry["repaired"], "Grounded draft should not require repair")
        self.assertEqual(verified_text, grounded_draft)

    def test_07_voice_emotion_selection_jev(self):
        """Verifies select_emotion_and_reference_jev selects optimal EmotionCategory and clip."""
        # Reflective text
        text_reflective = "Looking back over the last ten years, compounding habits have shaped everything I build."
        emotion, clip_id, details = select_emotion_and_reference_jev(text_reflective, self.engine)
        self.assertIsInstance(emotion, EmotionCategory)
        self.assertIsInstance(clip_id, str)
        self.assertTrue(clip_id.startswith("ref_"))

        # Humorous text
        text_humorous = "That was completely hilarious and bizarre, I couldn't stop laughing at the absurdity."
        emotion_h, clip_h, details_h = select_emotion_and_reference_jev(text_humorous, self.engine)
        self.assertIn(emotion_h, [EmotionCategory.HUMOROUS, EmotionCategory.LIVELY])
        self.assertEqual(clip_h, "ref_10")

    def test_08_conversation_engine_end_to_end_jev_telemetry(self):
        """Verifies ConversationEngine includes Jev verification telemetry in ConversationResponse.metadata."""
        db_path = PROJECT_ROOT / "data" / "capsule" / "test_jev_conversation.db"
        if db_path.exists():
            db_path.unlink()
        db = DatabaseManager(str(db_path))
        vector_store = VectorStore(str(PROJECT_ROOT / "data" / "capsule" / "test_jev_vectors"))
        retrieval = HybridRetrievalProvider(db, vector_store, jev_engine=self.engine)
        mock_llm = MockLLMProvider()
        personality = PersonalityProfile()

        engine = ConversationEngine(
            retrieval_provider=retrieval,
            db=db,
            llm_provider=mock_llm,
            personality_profile=personality,
            jev_verifier=self.verifier
        )

        resp = engine.process_turn("What is your 20-year compounding vision?")
        self.assertIn("jev_verification", resp.metadata)
        jev_meta = resp.metadata["jev_verification"]
        self.assertEqual(jev_meta["primary_model"], "sergiu-time-capsule:latest")
        self.assertEqual(jev_meta["critic_model"], "qwen3:4b-heretic")
        self.assertIn("noul_scores", jev_meta)
        self.assertIn("passed_first_try", jev_meta)
        self.assertIn("latency_ms", jev_meta)

        # Clean up temporary test files
        if db_path.exists():
            db_path.unlink()


if __name__ == "__main__":
    unittest.main()
