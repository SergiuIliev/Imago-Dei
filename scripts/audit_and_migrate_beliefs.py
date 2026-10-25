#!/usr/bin/env python3
"""
Beliefs & Memory Dialectical Auditor and Migrator.
Personal AI Time Capsule (Project 30).

Executes:
1. Complete audit & classification of all 1,231 memories in archive.db:
   - Maps categories: belief, event, career, relationship, lifestyle, reflection
   - Maps emotional tones: reflective, lively, serious, curious, humorous, melancholic, neutral
   - Cleans summaries: eliminates truncated cut-off words and aligns sentence boundaries
   - Synchronizes memories_fts virtual search table
2. Schema upgrade & migration for temporal_beliefs:
   - Adds supporting_evidence and counter_arguments columns
   - Removes redundant 'Architecture Design' beliefs (subsumed by Software & Infrastructure Craft)
   - Populates 34 consolidated, dialectically enriched beliefs with timestamped evidence
   - Enforces the Corroborated Evidence Invariant (>= 2 corroborated pieces required for supersession)
   - Synchronizes temporal_beliefs_fts virtual search table
"""

import sys
import re
import json
import sqlite3
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig
from memory.database import DatabaseManager
from scripts.seed_temporal_beliefs import seed_temporal_beliefs


def clean_summary_text(content: str, current_summary: str, is_synthetic: bool) -> str:
    """Produces a clean, complete sentence or word-boundary summary without trailing half-words."""
    if is_synthetic and current_summary and not current_summary.endswith("..."):
        return current_summary.strip()
    
    content = content.strip()
    if not content:
        return ""
    
    # Try finding sentence boundary between 25 and 110 characters
    match = re.match(r'^([^.?!]+[.?!])', content)
    if match and 25 <= len(match.group(1)) <= 110:
        return match.group(1).strip()
    
    # Otherwise cut at word boundary before 95 characters
    max_len = 95
    if len(content) <= max_len:
        return content
    
    truncated = content[:max_len]
    last_space = truncated.rfind(' ')
    if last_space > 30:
        return truncated[:last_space].strip() + "..."
    return truncated.strip() + "..."


def classify_memory_category(content: str, current_cat: str, is_synthetic: bool) -> str:
    """Maps memory content to standardized category enums."""
    if is_synthetic:
        return current_cat
    
    c = content.lower()
    # 1. Belief & Philosophy
    if any(k in c for k in [
        'brâncuși', 'brancusi', 'complexity resolved', 'no part is the best',
        'forge of progress', 'fire and pressure', "won't succeed if i don't fail",
        'what could happen if it all goes well', '20 years', 'twenty years',
        'intentional optimism', 'decadal compounding', 'data sovereignty',
        'sovereign intelligence', 'core philosophy', 'foundational belief'
    ]):
        return 'belief'
    
    # 2. Key Events & Milestones
    if any(k in c for k in [
        'french exam', 'solar eclipse', 'road trip across', 'passed the exam',
        'pumpkin pie', 'eclipse yesterday', 'taking the exam', 'acing the exam'
    ]):
        return 'event'
    
    # 3. Career & Systems Craft
    if any(k in c for k in [
        'software', 'programming', 'architecture', 'infrastructure', 'server',
        'microservice', 'git', 'deploy', 'fastapi', 'docker', 'database',
        'sqlite', 'agentic ai', 'hardware-in-the-loop', 'code module',
        '3d printer', 'engineering', 'web craft', 'developer'
    ]):
        return 'career'
    
    # 4. Relationships & Family
    if any(k in c for k in [
        'parents', 'mother', 'father', 'mom', 'dad', 'family',
        'partner', 'love letter', 'dear sergiu', 'moai', 'loved ones',
        'sandra', 'grand gestures of love'
    ]):
        return 'relationship'
    
    # 5. Lifestyle & Physical Disciplines
    if any(k in c for k in [
        'workout', 'routine', 'posture', 'shoulder', 'muscle', 'sprint',
        'sleep', 'recovery', 'travel', 'flight', 'dental', 'teeth',
        'running', 'gym', 'stretching', 'biomechanics', 'mitochondria'
    ]):
        return 'lifestyle'
    
    return 'reflection'


def classify_emotional_tone(content: str, is_synthetic: bool, current_tone: str) -> str:
    """Maps memory content to valid EmotionCategory values."""
    if is_synthetic:
        return current_tone
    
    c = content.lower()
    if any(k in c for k in ['excited', 'quite excited', 'super happy', 'proud', 'love you', 'joy', 'beautiful', 'ace it', 'thrilled', 'celebrate']):
        return 'lively'
    if any(k in c for k in ['tongue twister', 'haha', 'silly', 'chuckle', 'funny']):
        return 'humorous'
    if any(k in c for k in ['tears', 'nostalgic weakness', 'sad', 'melancholy', 'sorrow', 'longing']):
        return 'melancholic'
    if any(k in c for k in ['failure', 'setback', 'fire and pressure', 'pain', 'terrible', 'deteriorate', 'danger', 'strain', 'risk']):
        return 'serious'
    if any(k in c for k in ['curious', 'wondering', 'discovery', 'experiment', 'insatiable curiosity', 'explore', 'inquiring']):
        return 'curious'
    return 'reflective'


def audit_and_migrate():
    config = CapsuleConfig.load_or_create()
    db_path = config.storage.sqlite_db_path
    print(f"=== Personal AI Time Capsule: Dialectical Beliefs & Memory Auditor ===")
    print(f"Target Database: {db_path}")

    db = DatabaseManager(db_path)

    # 1. Clean up stray test memories if present
    with db.get_connection() as conn:
        conn.execute("DELETE FROM memories WHERE id LIKE 'test-verify%' OR date_occurred IS NULL")
        conn.execute("DELETE FROM memories_fts WHERE id LIKE 'test-verify%'")
        conn.commit()

    # 2. Audit and update all memories
    print("\n--- Auditing Memories Classification & Clean Summaries ---")
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, content, summary, category, emotional_tone, verification_status FROM memories")
        rows = cur.fetchall()
        total_memories = len(rows)
        print(f"Auditing {total_memories} memory items...")

        cat_counts = {}
        tone_counts = {}
        updated = 0

        for r in rows:
            mid = r["id"]
            content = r["content"]
            old_summary = r["summary"] or ""
            old_cat = r["category"] or ""
            old_tone = r["emotional_tone"] or ""
            v_status = r["verification_status"]
            is_syn = (v_status == "synthetic_fixture")

            new_cat = classify_memory_category(content, old_cat, is_syn)
            new_tone = classify_emotional_tone(content, is_syn, old_tone)
            new_summary = clean_summary_text(content, old_summary, is_syn)

            cat_counts[new_cat] = cat_counts.get(new_cat, 0) + 1
            tone_counts[new_tone] = tone_counts.get(new_tone, 0) + 1

            if new_cat != old_cat or new_tone != old_tone or new_summary != old_summary:
                cur.execute("""
                    UPDATE memories
                    SET category = ?, emotional_tone = ?, summary = ?
                    WHERE id = ?
                """, (new_cat, new_tone, new_summary, mid))

                # Update FTS
                cur.execute("""
                    UPDATE memories_fts
                    SET category = ?, summary = ?
                    WHERE id = ?
                """, (new_cat, new_summary, mid))
                updated += 1

        conn.commit()
        print(f"  Audit complete: {updated} memories updated with clean categories, tones, and summaries.")
        print(f"  Category Distribution: {cat_counts}")
        print(f"  Emotional Tone Distribution: {tone_counts}")

    # 3. Seed and Consolidate Temporal Beliefs
    print("\n--- Seeding Consolidated Dialectical Temporal Beliefs ---")
    belief_count = seed_temporal_beliefs(db_path)

    # 4. Final Verification
    stats = db.get_archive_stats()
    print("\n=== Final Archive Verification ===")
    print(f"  Total Media Files:     {stats['media_files']}")
    print(f"  Total Transcripts:     {stats['transcripts']}")
    print(f"  Total Memories:        {stats['total_memories']}")
    print(f"  Temporal Beliefs:      {stats['temporal_beliefs']}")

    # Verify beliefs have supporting evidence and counterarguments
    all_beliefs = db.list_all_beliefs()
    print(f"\nAudit of {len(all_beliefs)} Temporal Beliefs:")
    for b in all_beliefs:
        status_tag = "ACTIVE" if b.active else f"SUPERSEDED by {b.superseded_by_id}"
        print(f"  - [{b.era_label}] ({b.topic}) [{status_tag}]:")
        print(f"      Statement: {b.belief_statement[:70]}...")
        print(f"      Supporting Evidence: {len(b.supporting_evidence)} items")
        print(f"      Counterarguments:    {len(b.counter_arguments)} items")

    print("\nAudit and dialectical migration completed successfully!")


if __name__ == "__main__":
    audit_and_migrate()
