"""
Seed Temporal Beliefs for Personal AI Time Capsule.
Extracts and seeds structured TemporalBelief entries across Sergiu's core life themes:
- 20-Year Compounding Vision (2026-09-06, source 3791e8c7)
- Failure and Resilience ("won't succeed if I don't fail first", 2026-09-06)
- Software & Infrastructure Craft (reliable home automation, clean architecture, Brâncuși simplicity)
- French Learning & Consistency (2026-09-02, source 63a53acf)
- Physical Resilience & Daily Disciplines

Preserves chronological belief evolution (early vs mid vs present) and attaches
verified memory IDs from the 113 autobiographical records.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig, ensure_directories
from core.models import TemporalBelief, VerificationStatus
from memory.database import DatabaseManager


def find_matching_memory_ids(db: DatabaseManager, keywords: List[str], source_prefix: str = "") -> List[str]:
    """Finds memory IDs in archive matching search terms and optional source ID prefix."""
    matched = []
    with db.get_connection() as conn:
        for kw in keywords:
            sql = "SELECT id FROM memories WHERE content LIKE ? AND verification_status = ?"
            params: List[Any] = [f"%{kw}%", VerificationStatus.PERSONAL_VERIFIED.value]
            if source_prefix:
                sql += " AND source_id LIKE ?"
                params.append(f"%{source_prefix}%")
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                if r[0] not in matched:
                    matched.append(r[0])
    return matched


def seed_temporal_beliefs(db_path: str = ""):
    config = CapsuleConfig.load_or_create()
    target_db = db_path or config.storage.sqlite_db_path
    db = DatabaseManager(target_db)

    print(f"[Seed Beliefs] Connecting to database: {target_db}")

    # Gather matching verified memory IDs from DB
    vision_ids = find_matching_memory_ids(db, ["20 years", "improve each day", "what could happen if it all goes well"], source_prefix="3791e8c7")
    failure_ids = find_matching_memory_ids(db, ["won't succeed if I don't fail first", "forge of progress", "fire and pressure"], source_prefix="3791e8c7")
    craft_ids = find_matching_memory_ids(db, ["Bruncush", "simplicity is complexity resolved", "no part is the best part"])
    french_ids = find_matching_memory_ids(db, ["French exam", "practice my French", "vocabulary and grammar"], source_prefix="63a53acf")
    french_immersion_ids = find_matching_memory_ids(db, ["vocabulary and grammar", "loop and it wasn't stopping"], source_prefix="657f5f91")
    french_all = list(dict.fromkeys(french_ids + french_immersion_ids))
    physical_ids = find_matching_memory_ids(db, ["iron routine for sustainable travel", "sprints where I build my body up", "correcting my posture"])

    beliefs_definitions = [
        # 1. 20-Year Compounding Vision
        TemporalBelief(
            id="belief-vision-20year",
            topic="20-Year Compounding Vision",
            belief_statement="Look 20 years into the future with intentional optimism: ask what could happen if it all goes well, and compound daily incremental improvements in craft, health, and life balance.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=vision_ids or ["4337239a-7680-40a3-b1c2-54713a4709ab", "7e83482d-fcc6-4caf-875f-09a743faaec1", "f13a6d0e-3442-4ebb-9889-3b07c9f65bd4"],
            active=True
        ),

        # 2. Failure and Resilience (Evolving from early risk-avoidance to present forge of progress)
        TemporalBelief(
            id="belief-failure-early",
            topic="Failure and Resilience",
            belief_statement="Failure should be prevented and avoided through cautious upfront scoping and safe incremental steps.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-failure-present",
            active=False,
            reason_for_change="Excessive risk avoidance slowed learning; setbacks proved essential for high-leverage growth."
        ),
        TemporalBelief(
            id="belief-failure-present",
            topic="Failure and Resilience",
            belief_statement="I won't succeed if I don't fail first. Setbacks, demotivation, and temporary weakness are the forge of progress where fire and pressure shape capability and resilience.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=failure_ids or ["4337239a-7680-40a3-b1c2-54713a4709ab", "82db2a82-cac3-439c-93f2-908503ce7479", "ee121c55-9128-41ef-b4ac-894a0f65b3f3"],
            supersedes_id="belief-failure-early",
            active=True
        ),

        # 3. Software & Infrastructure Craft (Evolving through three distinct eras)
        TemporalBelief(
            id="belief-craft-early",
            topic="Software & Infrastructure Craft",
            belief_statement="Monolithic single-file architectures and rapid scripting are best because they optimize for speed of initial iteration.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-craft-mid",
            active=False,
            reason_for_change="Single-file monoliths grew brittle and unmaintainable as system complexity increased."
        ),
        TemporalBelief(
            id="belief-craft-mid",
            topic="Software & Infrastructure Craft",
            belief_statement="Systems should be decomposed into decoupled microservices with network APIs to maximize independence.",
            era_label="Mid-to-Late 20s (2019-2022)",
            approx_year_start=2019,
            approx_year_end=2022,
            evidence_memory_ids=[],
            supersedes_id="belief-craft-early",
            superseded_by_id="belief-craft-present",
            active=False,
            reason_for_change="Distributed microservice sprawl introduced network latency, operational overhead, and distributed debugging pain."
        ),
        TemporalBelief(
            id="belief-craft-present",
            topic="Software & Infrastructure Craft",
            belief_statement="Simplicity is complexity resolved (Constantin Brâncuși / Brancusi / Bruncush doctrine; Elon Musk 'no part is the best part'). High-reliability local-first architecture, standardized .69 static IPs, resilient triple-instance Home Assistant, and hardware-in-the-loop validation without unnecessary layers.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=craft_ids or ["0b8de00a-03eb-414b-97c6-4dfdc88bc97a", "12d463c7-0b28-471e-9e04-25d17b6f5d62", "7f4949c2-2029-46af-9d34-977a47edc800"],
            supersedes_id="belief-craft-mid",
            active=True
        ),

        # 4. French Learning & Consistency
        TemporalBelief(
            id="belief-french-present",
            topic="French Learning & Consistency",
            belief_statement="Disciplined daily immersion and conversational practice create true language fluency. Maintain consistency even when vocabulary loops through the mind at night, and celebrate milestones with meaningful rewards (like baking pumpkin pie after passing exams).",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=french_all or ["f4f2ca8f-b9a9-4f12-9db0-0682461daa3d", "4466cb7c-30dd-4bd5-a5d7-affb3cd3cbf4", "1bee97f1-9165-483f-8fe8-d6c5e7839c09", "9cdb8d48-e16b-47c3-be30-aea7de33c1f7"],
            active=True
        ),

        # 5. Physical Resilience & Daily Disciplines
        TemporalBelief(
            id="belief-physical-early",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Physical fitness can be maintained through occasional gym sessions and ad-hoc workouts whenever time permits.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-physical-present",
            active=False,
            reason_for_change="Inconsistent training led to energy troughs and poor posture mechanics during long software engineering blocks."
        ),
        TemporalBelief(
            id="belief-physical-present",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Establish iron routines for sustainable travel and daily life: sprint cycles of intense physical conditioning alternating with deliberate recovery, daily posture correction, shoulder exercises, and incremental bodily care.",
            era_label="Present Day (2023-Present)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=physical_ids or ["9ae12163-3e95-445b-baf5-7e8c5dee8045", "d0fb47f4-c32d-4ca4-acf8-beccc82938e4", "0365008c-114c-4b4b-8252-cd40ef325b19", "7e83482d-fcc6-4caf-875f-09a743faaec1"],
            supersedes_id="belief-physical-early",
            active=True
        )
    ]

    count = 0
    for belief in beliefs_definitions:
        db.add_belief(belief)
        count += 1
        print(f"  + Seeded belief [{belief.era_label}] {belief.topic}: {belief.belief_statement[:60]}...")

    stats = db.get_archive_stats()
    print(f"\n[Seed Beliefs] Successfully seeded {count} temporal beliefs. Total beliefs in DB: {stats['temporal_beliefs']}")
    return count


if __name__ == "__main__":
    db_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    seed_temporal_beliefs(db_arg)
