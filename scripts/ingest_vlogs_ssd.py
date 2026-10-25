#!/usr/bin/env python3
"""
Dedicated Batch Ingestion Runner for Project 30 (Personal AI Time Capsule).
Ingests authentic personal vlog recordings from connected external SSD:
/Volumes/SSD/Photography/_Daily Speak Project

Persists all extracted metadata, container creation timestamps, GPS coordinates,
granular transcripts with domain lexicon normalization, narrative memories,
Chroma vector embeddings, and 16kHz audio clips locally on-device so the
Time Capsule functions completely offline even when the external SSD is disconnected.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from providers.asr.whisper_provider import PyAVWhisperASRProvider
from ingestion.pipeline import MediaIngestionPipeline


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/SSD/Photography/_Daily Speak Project"
    
    print("\n" + "=" * 70)
    print(" PROJECT 30: PERSONAL AI TIME CAPSULE — AUTONOMOUS INGESTION")
    print("=" * 70)
    print(f" Source Directory:    {target_dir}")
    print(f" Timestamp:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(target_dir):
        print(f"\n[ERROR] Target directory not found: {target_dir}")
        print("Please ensure the external SSD is mounted at /Volumes/SSD.")
        sys.exit(1)

    # Initialize subsystem
    config = CapsuleConfig.load_or_create()
    print(f" Local SQLite DB:     {config.storage.sqlite_db_path}")
    print(f" Local Vector Store:  {config.storage.chroma_dir}")
    print(f" Local Audio Cache:   {config.storage.audio_output_dir}")
    print("-" * 70)

    db = DatabaseManager(config.storage.sqlite_db_path)
    vs = VectorStore(config.storage.chroma_dir)
    asr = PyAVWhisperASRProvider(model_size=config.asr.model_size)
    pipeline = MediaIngestionPipeline(config, db, vs, asr)

    # Initial stats
    initial_stats = db.get_archive_stats()
    print(f" Database state before run: {initial_stats['media_files']} media files, "
          f"{initial_stats['transcripts']} transcripts, {initial_stats['total_memories']} memories")

    # Discover files
    files = pipeline.discover_media_files(target_dir)
    total_files = len(files)
    print(f"\nDiscovered {total_files} media files in {target_dir}.\n")

    if total_files == 0:
        print("No supported media files found. Exiting.")
        return

    processed_count = 0
    skipped_count = 0
    quarantined_count = 0
    failed_count = 0
    total_transcripts_added = 0
    total_memories_added = 0
    start_time = time.time()

    for i, file_path in enumerate(files, 1):
        f_name = file_path.name
        f_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"[{i:03d}/{total_files:03d}] {f_name} ({f_size_mb:.1f} MB)... ", end="", flush=True)

        t_file_start = time.time()
        res = pipeline.process_file(str(file_path))
        t_file_elapsed = time.time() - t_file_start

        status = res.get("status")
        if status == "duplicate_skipped":
            skipped_count += 1
            print(f"⏩ [SKIP: Already indexed in DB]")
        elif status == "quarantined_corrupt_stub":
            quarantined_count += 1
            print(f"⚠️ [QUARANTINED: Corrupt stub <4KB]")
        elif status == "success":
            processed_count += 1
            segs = res.get("transcript_segments", 0)
            mems = res.get("memories_created", 0)
            rec_date = res.get("recorded_at") or "unknown date"
            v_status = res.get("verification_status", "personal_verified")
            total_transcripts_added += segs
            total_memories_added += mems
            status_tag = "SYNTHETIC_FIXTURE" if "synthetic" in v_status else "INDEXED"
            print(f"✅ [{status_tag}] {segs} utterances, {mems} memories, recorded {rec_date[:10]} ({t_file_elapsed:.1f}s)")
        else:
            failed_count += 1
            err = res.get("error", "Unknown error")
            print(f"❌ [FAILED] {err}")

    total_elapsed = time.time() - start_time

    # Checkpoint WAL and verify unlock
    try:
        with db.get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        db_unlocked = True
    except Exception as e:
        print(f"[Warning] WAL checkpoint issue: {e}")
        db_unlocked = False

    final_stats = db.get_archive_stats()

    print("\n" + "=" * 70)
    print(" INGESTION RUN COMPLETED")
    print("=" * 70)
    print(f" Total files examined:       {total_files}")
    print(f" Newly processed & indexed:  {processed_count}")
    print(f" Previously indexed (skip):  {skipped_count}")
    print(f" Quarantined stubs (<4KB):   {quarantined_count}")
    print(f" Failed files:               {failed_count}")
    print(f" Total run duration:         {total_elapsed/60.0:.2f} minutes")
    print(f" Transcripts added this run: {total_transcripts_added}")
    print(f" Memories added this run:    {total_memories_added}")
    print(f" Database Checkpoint/Unlock: {'CONFIRMED (TRUNCATED WAL)' if db_unlocked else 'PENDING'}")
    print("-" * 70)
    print(f" Final Archive DB Stats:")
    print(f"  - Indexed Media Sources:   {final_stats['media_files']}")
    print(f"  - Granular Transcripts:    {final_stats['transcripts']}")
    print(f"  - Total Memories in Store: {final_stats['total_memories']} ({final_stats['verified_memories']} verified, {final_stats['synthetic_fixtures']} synthetic)")
    print(f"  - Temporal Beliefs:        {final_stats['temporal_beliefs']}")
    print("=" * 70)

    # Show a few sample memories from the newly ingested vlogs
    print("\nSample Newly Ingested Memories from Authentic Vlogs:")
    sample_mems = db.list_memories(limit=5)
    for m in sample_mems:
        print(f"\n • [{m.verification_status.value.upper()}] Date: {m.date_occurred} (Span: {m.start_seconds:.1f}s - {m.end_seconds:.1f}s)")
        print(f"   Summary: {m.summary}")
        print(f"   Tags: {m.tags}")
        print(f"   Excerpt: \"{m.content[:140]}...\"")
    print("\nAll memories and demuxed audio clips are safely preserved on local disk.")


if __name__ == "__main__":
    main()
