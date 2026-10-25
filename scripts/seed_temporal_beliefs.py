"""
Seed Temporal Beliefs for Personal AI Time Capsule.
Extracts and seeds structured TemporalBelief entries across Sergiu's core life themes:
- Software & Infrastructure Craft (Simplicity is complexity resolved / Brâncuși / .69 IPs)
- Failure and Resilience ("won't succeed if I don't fail first" / Forge of progress)
- 20-Year Compounding Vision (intentional optimism and daily micro-compounding)
- Physical Resilience & Daily Disciplines (iron routines and posture alignment)
- Language Mastery & Consistency (French immersion and conversational milestones)
- Data Privacy & Local Sovereignty (Rule 6: 100% on-device local execution)

Preserves chronological multi-year belief progressions across 2023, 2024, 2025, and 2026,
linking superseding IDs, reasons for change, and verified memory IDs.
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
    craft_ids = find_matching_memory_ids(db, ["Brâncuși", "Bruncush", "Brancusi", "simplicity is complexity resolved", "no part is the best part"])
    french_ids = find_matching_memory_ids(db, ["French exam", "practice my French", "vocabulary and grammar"], source_prefix="63a53acf")
    french_immersion_ids = find_matching_memory_ids(db, ["vocabulary and grammar", "loop and it wasn't stopping"], source_prefix="657f5f91")
    french_all = list(dict.fromkeys(french_ids + french_immersion_ids))
    physical_ids = find_matching_memory_ids(db, ["iron routine for sustainable travel", "sprints where I build my body up", "correcting my posture"])

    beliefs_definitions = [
        # =========================================================================
        # 1. SOFTWARE & INFRASTRUCTURE CRAFT (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
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
            superseded_by_id="belief-craft-2023",
            active=False,
            reason_for_change="Distributed microservice sprawl introduced network latency, operational overhead, and distributed debugging pain."
        ),
        TemporalBelief(
            id="belief-craft-2023",
            topic="Software & Infrastructure Craft",
            belief_statement="Cloud convenience and managed smart home APIs are acceptable initial tradeoffs to get systems connected quickly, even if local control is incomplete.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            supersedes_id="belief-craft-mid",
            superseded_by_id="belief-craft-2024",
            active=False,
            reason_for_change="Cloud API outages, vendor subscription changes, and latency made cloud-dependent automation unreliable for daily living."
        ),
        TemporalBelief(
            id="belief-craft-2024",
            topic="Software & Infrastructure Craft",
            belief_statement="Total local control is mandatory: decouple from cloud vendors, adopt Home Assistant as the primary orchestration hub, and connect devices over private Zigbee and Tailscale mesh.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-craft-2023",
            superseded_by_id="belief-craft-2025",
            active=False,
            reason_for_change="Ad-hoc IP addressing and disparate configs between locations caused configuration drift and fragile debugging."
        ),
        TemporalBelief(
            id="belief-craft-2025",
            topic="Software & Infrastructure Craft",
            belief_statement="Standardize network topology with deterministic static .69 IPs, Cudy P5 5G cellular fail-safe routing, and native MSS clamping across Romania Home, Swiss, and LakeOn instances.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-craft-2024",
            superseded_by_id="belief-craft-present",
            active=False,
            reason_for_change="Architecture required philosophical consolidation: elimination of all redundant proxy layers and deep hardware-in-the-loop closure."
        ),
        TemporalBelief(
            id="belief-craft-present",
            topic="Software & Infrastructure Craft",
            belief_statement="Simplicity is complexity resolved (Constantin Brâncuși doctrine; Elon Musk 'no part is the best part'). High-reliability local-first architecture, standardized .69 static IPs, resilient triple-instance Home Assistant, and hardware-in-the-loop validation without unnecessary layers.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=craft_ids or ["0b8de00a-03eb-414b-97c6-4dfdc88bc97a", "12d463c7-0b28-471e-9e04-25d17b6f5d62", "7f4949c2-2029-46af-9d34-977a47edc800"],
            supersedes_id="belief-craft-2025",
            active=True
        ),

        # =========================================================================
        # 2. FAILURE AND RESILIENCE (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-failure-early",
            topic="Failure and Resilience",
            belief_statement="Failure should be prevented and avoided through cautious upfront scoping and safe incremental steps.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-failure-2023",
            active=False,
            reason_for_change="Excessive risk avoidance slowed learning; setbacks proved essential for high-leverage growth."
        ),
        TemporalBelief(
            id="belief-failure-2023",
            topic="Failure and Resilience",
            belief_statement="Failure and friction are frustrating disruptions to be minimized through cautious planning and defensive execution.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            supersedes_id="belief-failure-early",
            superseded_by_id="belief-failure-2024",
            active=False,
            reason_for_change="Defensive planning often led to analysis paralysis without delivering real-world empirical validation."
        ),
        TemporalBelief(
            id="belief-failure-2024",
            topic="Failure and Resilience",
            belief_statement="Lean into empirical trial and rapid prototyping: small failures encountered early in hardware and code are valuable diagnostic signals.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-failure-2023",
            superseded_by_id="belief-failure-2025",
            active=False,
            reason_for_change="Tolerating failure was not enough; systems and habits needed intentional adversarial stress testing."
        ),
        TemporalBelief(
            id="belief-failure-2025",
            topic="Failure and Resilience",
            belief_statement="Adversarial red-teaming: deliberately stress systems, network links, and personal stamina under harsh real-world conditions to expose invisible brittle points.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-failure-2024",
            superseded_by_id="belief-failure-present",
            active=False,
            reason_for_change="Evolved from seeing stress as a technical test to realizing that struggle and setback are the fundamental forge of personal capability."
        ),
        TemporalBelief(
            id="belief-failure-present",
            topic="Failure and Resilience",
            belief_statement="I won't succeed if I don't fail first. Setbacks, demotivation, and temporary weakness are the forge of progress where fire and pressure shape capability and resilience.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=failure_ids or ["4337239a-7680-40a3-b1c2-54713a4709ab", "82db2a82-cac3-439c-93f2-908503ce7479", "ee121c55-9128-41ef-b4ac-894a0f65b3f3"],
            supersedes_id="belief-failure-2025",
            active=True
        ),

        # =========================================================================
        # 3. 20-YEAR COMPOUNDING VISION (2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-vision-2023",
            topic="20-Year Compounding Vision",
            belief_statement="Focus on immediate 1-to-2 year horizon: solve current technical challenges, establish remote work flexibility, and build core foundational competencies.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            superseded_by_id="belief-vision-2024",
            active=False,
            reason_for_change="Short-term horizons led to reactive planning rather than compounding strategic leverage."
        ),
        TemporalBelief(
            id="belief-vision-2024",
            topic="20-Year Compounding Vision",
            belief_statement="Adopt a 5-year strategic arc: invest heavily in lasting physical property, dual Switzerland-Romania base infrastructure, and autonomous automation platforms.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-vision-2023",
            superseded_by_id="belief-vision-2025",
            active=False,
            reason_for_change="Recognized that true compounding requires a generational multi-decade horizon rather than a 5-year cycle."
        ),
        TemporalBelief(
            id="belief-vision-2025",
            topic="20-Year Compounding Vision",
            belief_statement="Decadal compounding: create self-sustaining systems, sovereign energy, and deep mastery of local AI agents that endure for over ten years.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-vision-2024",
            superseded_by_id="belief-vision-20year",
            active=False,
            reason_for_change="Expanded to a full 20-year vision centered on intentional optimism and daily micro-compounding."
        ),
        TemporalBelief(
            id="belief-vision-20year",
            topic="20-Year Compounding Vision",
            belief_statement="Look 20 years into the future with intentional optimism: ask what could happen if it all goes well, and compound daily incremental improvements in craft, health, and life balance.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=vision_ids or ["4337239a-7680-40a3-b1c2-54713a4709ab", "7e83482d-fcc6-4caf-875f-09a743faaec1", "f13a6d0e-3442-4ebb-9889-3b07c9f65bd4"],
            supersedes_id="belief-vision-2025",
            active=True
        ),

        # =========================================================================
        # 4. PHYSICAL RESILIENCE & DAILY DISCIPLINES (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-physical-early",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Physical fitness can be maintained through occasional gym sessions and ad-hoc workouts whenever time permits.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-physical-2023",
            active=False,
            reason_for_change="Inconsistent training led to energy troughs and poor posture mechanics during long software engineering blocks."
        ),
        TemporalBelief(
            id="belief-physical-2023",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Workouts and physical training can be fitted in ad-hoc around heavy programming and travel schedules.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            supersedes_id="belief-physical-early",
            superseded_by_id="belief-physical-2024",
            active=False,
            reason_for_change="Ad-hoc fitness resulted in physical fatigue, posture deterioration, and inconsistent energy."
        ),
        TemporalBelief(
            id="belief-physical-2024",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Prioritize physical conditioning as an essential cognitive prerequisite: implement consistent morning routines and dedicated gym sessions.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-physical-2023",
            superseded_by_id="belief-physical-2025",
            active=False,
            reason_for_change="General workouts lacked specific postural corrections and shoulder rehabilitation needed for desk longevity."
        ),
        TemporalBelief(
            id="belief-physical-2025",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Targeted biomechanics: deliberate posture alignment, shoulder rehabilitation, sprint conditioning, and matching recovery to engineering output.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-physical-2024",
            superseded_by_id="belief-physical-present",
            active=False,
            reason_for_change="Travel transitions disrupted biomechanical routines; required an immutable iron routine adaptable to any location."
        ),
        TemporalBelief(
            id="belief-physical-present",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Establish iron routines for sustainable travel and daily life: sprint cycles of intense physical conditioning alternating with deliberate recovery, daily posture correction, shoulder exercises, and incremental bodily care.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=physical_ids or ["9ae12163-3e95-445b-baf5-7e8c5dee8045", "d0fb47f4-c32d-4ca4-acf8-beccc82938e4", "0365008c-114c-4b4b-8252-cd40ef325b19", "7e83482d-fcc6-4caf-875f-09a743faaec1"],
            supersedes_id="belief-physical-2025",
            active=True
        ),

        # =========================================================================
        # 5. LANGUAGE MASTERY & CONSISTENCY (FRENCH) (2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-french-2023",
            topic="French Learning & Consistency",
            belief_statement="French can be acquired through passive app practice, vocabulary drills, and casual exposure when residing in Switzerland.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            superseded_by_id="belief-french-2024",
            active=False,
            reason_for_change="Passive exposure failed to build conversational fluency or active verbal recall."
        ),
        TemporalBelief(
            id="belief-french-2024",
            topic="French Learning & Consistency",
            belief_statement="Structured grammar study and active conversational attempts are required to break through intermediate language plateaus.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-french-2023",
            superseded_by_id="belief-french-2025",
            active=False,
            reason_for_change="Structured study needed continuous high-intensity immersion to prevent syntax decay."
        ),
        TemporalBelief(
            id="belief-french-2025",
            topic="French Learning & Consistency",
            belief_statement="Intense daily immersion: rigorous French exam preparation, relentless vocabulary repetition until the language loops through thoughts, and celebrating milestones.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-french-2024",
            superseded_by_id="belief-french-present",
            active=False,
            reason_for_change="Exam preparation transitioned into integrated spontaneous fluency and self-rewarding habits."
        ),
        TemporalBelief(
            id="belief-french-present",
            topic="French Learning & Consistency",
            belief_statement="Disciplined daily immersion and conversational practice create true language fluency. Maintain consistency even when vocabulary loops through the mind at night, and celebrate milestones with meaningful rewards (like baking pumpkin pie after passing exams).",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=french_all or ["f4f2ca8f-b9a9-4f12-9db0-0682461daa3d", "4466cb7c-30dd-4bd5-a5d7-affb3cd3cbf4", "1bee97f1-9165-483f-8fe8-d6c5e7839c09", "9cdb8d48-e16b-47c3-be30-aea7de33c1f7"],
            supersedes_id="belief-french-2025",
            active=True
        ),

        # =========================================================================
        # 6. DATA PRIVACY & LOCAL SOVEREIGNTY (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-privacy-early",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Cloud hosting is completely fine for personal tools as long as basic HTTPS is enforced.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            evidence_memory_ids=[],
            superseded_by_id="belief-privacy-2023",
            active=False,
            reason_for_change="Cloud data leaks and corporate surveillance demonstrated the vulnerability of hosted personal data."
        ),
        TemporalBelief(
            id="belief-privacy-2023",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Commercial cloud LLMs and hosted tools are convenient for exploration; data sensitivity is manageable with basic settings.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            evidence_memory_ids=[],
            supersedes_id="belief-privacy-early",
            superseded_by_id="belief-privacy-2024",
            active=False,
            reason_for_change="Third-party data scraping, privacy policy shifts, and terms of service changes highlighted severe data sovereignty risks."
        ),
        TemporalBelief(
            id="belief-privacy-2024",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Personal memories, private journals, and core credentials must never touch external third-party cloud servers. Begin running Ollama and local models.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            evidence_memory_ids=[],
            supersedes_id="belief-privacy-2023",
            superseded_by_id="belief-privacy-2025",
            active=False,
            reason_for_change="Local models alone were insufficient; needed loopback network isolation, Pi-hole DNS sinkholing, and hardware-level enforcement."
        ),
        TemporalBelief(
            id="belief-privacy-2025",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Sovereign local infrastructure: Pi-hole DNS sinkholes, drop cloud telemetry (disable cmsd on Cudy P5), Tailscale encrypted mesh, and local-only SQLite archives.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            evidence_memory_ids=[],
            supersedes_id="belief-privacy-2024",
            superseded_by_id="belief-privacy-present",
            active=False,
            reason_for_change="Elevated to an absolute constitutional operational invariant across all AI agents and personal data systems."
        ),
        TemporalBelief(
            id="belief-privacy-present",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Personal memories and identity must run 100% locally with zero cloud leakage, loopback binding, cryptographic manifests, and strict Rule 6 data privacy invariants.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            evidence_memory_ids=[],
            supersedes_id="belief-privacy-2025",
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
