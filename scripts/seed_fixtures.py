"""
Utility script to seed the Personal AI Time Capsule archive with synthetic test fixtures.
Labels all fixtures clearly with VerificationStatus.SYNTHETIC_FIXTURE.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig, ensure_directories
from core.models import MemoryItem, TemporalBelief, VerificationStatus
from memory.database import DatabaseManager
from memory.vector_store import VectorStore


def seed_fixtures():
    config = CapsuleConfig.load_or_create()
    ensure_directories(config)

    db = DatabaseManager(config.storage.sqlite_db_path)
    vector_store = VectorStore(config.storage.chroma_dir)

    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "synthetic_fixtures.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Seed Memories
    memories = []
    for item in data.get("synthetic_memories", []):
        mem = MemoryItem(
            id=item["id"],
            content=item["content"],
            summary=item["summary"],
            category=item.get("category", "reflection"),
            date_occurred=item.get("date_occurred"),
            verification_status=VerificationStatus.SYNTHETIC_FIXTURE,
            confidence=item.get("confidence", 1.0),
            emotional_tone=item.get("emotional_tone", "neutral"),
            tags=item.get("tags", [])
        )
        db.add_memory(mem)
        memories.append(mem)

    # Seed Chroma embeddings
    vector_store.add_batch_memory_embeddings(memories)

    # Seed Temporal Beliefs
    for b_data in data.get("temporal_beliefs", []):
        belief = TemporalBelief(
            id=b_data["id"],
            topic=b_data["topic"],
            belief_statement=b_data["belief_statement"],
            era_label=b_data["era_label"],
            approx_year_start=b_data.get("approx_year_start"),
            approx_year_end=b_data.get("approx_year_end"),
            evidence_memory_ids=b_data.get("evidence_memory_ids", []),
            superseded_by_id=b_data.get("superseded_by_id"),
            supersedes_id=b_data.get("supersedes_id"),
            active=b_data.get("active", True),
            reason_for_change=b_data.get("reason_for_change")
        )
        db.add_belief(belief)

    stats = db.get_archive_stats()
    print(f"[Seed] Successfully seeded archive: {stats}")


if __name__ == "__main__":
    seed_fixtures()
