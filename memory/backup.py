"""
Backup, Export, and Disaster Recovery Module for Personal AI Time Capsule.
Creates verified timestamped archives with SHA-256 manifests, integrity checks,
and portable JSON/Markdown exports.
"""

from __future__ import annotations
import os
import tarfile
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.config import CapsuleConfig
from memory.database import DatabaseManager


def calculate_sha256(file_path: str) -> str:
    """Calculates SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class BackupManager:
    """Manages full archive backups, portable exports, and verified restoration."""

    def __init__(self, config: CapsuleConfig, db: DatabaseManager):
        self.config = config
        self.db = db
        Path(self.config.storage.exports_dir).mkdir(parents=True, exist_ok=True)

    def create_full_backup(self, label: str = "snapshot") -> Dict[str, Any]:
        """
        Creates a timestamped .tar.gz archive containing:
        - archive.db (SQLite database)
        - vector_store/ (Chroma vector files)
        - SHA256 manifest
        """
        # 1. Flush SQLite WAL to disk
        try:
            with self.db.get_connection() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception as e:
            print(f"[Backup] WAL checkpoint notice: {e}")

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"capsule_backup_{label}_{timestamp}.tar.gz"
        backup_path = Path(self.config.storage.exports_dir) / backup_filename

        staging_dir = Path(self.config.storage.exports_dir) / f"staging_{timestamp}_{label}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage SQLite DB
            db_path = Path(self.config.storage.sqlite_db_path)
            if db_path.exists():
                shutil.copy2(db_path, staging_dir / "archive.db")

            # Stage Vector Store
            chroma_dir = Path(self.config.storage.chroma_dir)
            if chroma_dir.exists():
                staged_chroma = staging_dir / "vector_store"
                shutil.copytree(chroma_dir, staged_chroma)

            # Build Manifest from staged files
            manifest = {
                "created_at": datetime.utcnow().isoformat(),
                "version": 1,
                "backup_filename": backup_filename,
                "files": {}
            }

            for root, _, files in os.walk(staging_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = os.path.relpath(full_p, staging_dir)
                    manifest["files"][rel_p] = {
                        "size_bytes": os.path.getsize(full_p),
                        "sha256": calculate_sha256(str(full_p))
                    }

            # Write manifest into staging
            with open(staging_dir / "manifest.json", "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

            # Create tar.gz from staging
            with tarfile.open(backup_path, "w:gz") as tar:
                for root, _, files in os.walk(staging_dir):
                    for f in files:
                        full_p = Path(root) / f
                        rel_p = os.path.relpath(full_p, staging_dir)
                        tar.add(str(full_p), arcname=rel_p)

            archive_size = os.path.getsize(backup_path)
            archive_hash = calculate_sha256(str(backup_path))

            return {
                "backup_file": str(backup_path),
                "size_bytes": archive_size,
                "sha256": archive_hash,
                "file_count": len(manifest["files"]),
                "timestamp": timestamp
            }
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """Verifies the structural integrity and SHA-256 checksums of an archive."""
        p = Path(backup_path)
        if not p.exists():
            return {"valid": False, "error": f"Backup file {backup_path} not found"}

        with tarfile.open(p, "r:gz") as tar:
            # Extract manifest
            try:
                manifest_member = tar.getmember("manifest.json")
                f = tar.extractfile(manifest_member)
                if not f:
                    return {"valid": False, "error": "Could not read manifest.json from archive"}
                manifest = json.loads(f.read().decode("utf-8"))
            except KeyError:
                return {"valid": False, "error": "Archive is missing manifest.json"}

            # Verify files in manifest
            corrupted = []
            for arc_name, info in manifest.get("files", {}).items():
                try:
                    member = tar.getmember(arc_name)
                    member_file = tar.extractfile(member)
                    if not member_file:
                        corrupted.append(f"{arc_name}: unreadable")
                        continue
                    content = member_file.read()
                    actual_hash = hashlib.sha256(content).hexdigest()
                    if actual_hash != info["sha256"]:
                        corrupted.append(f"{arc_name}: hash mismatch")
                except KeyError:
                    corrupted.append(f"{arc_name}: missing from archive")

        if corrupted:
            return {"valid": False, "corrupted_files": corrupted}
        return {"valid": True, "file_count": len(manifest.get("files", {}))}

    def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restores database and vector store from a verified backup archive."""
        verification = self.verify_backup(backup_path)
        if not verification.get("valid"):
            raise ValueError(f"Backup verification failed: {verification}")

        p = Path(backup_path)
        temp_restore_dir = p.parent / f"restore_temp_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        temp_restore_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(p, "r:gz") as tar:
                tar.extractall(path=temp_restore_dir)

            # Move archive.db into place
            restored_db = temp_restore_dir / "archive.db"
            if restored_db.exists():
                target_db = Path(self.config.storage.sqlite_db_path)
                # Create safety copy of existing DB
                if target_db.exists():
                    shutil.copy2(target_db, f"{target_db}.bak_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
                # Remove WAL and SHM sidecars to prevent replaying stale WAL transactions
                wal_file = target_db.with_name(f"{target_db.name}-wal")
                shm_file = target_db.with_name(f"{target_db.name}-shm")
                if wal_file.exists():
                    os.remove(wal_file)
                if shm_file.exists():
                    os.remove(shm_file)

                shutil.copy2(restored_db, target_db)

            # Restore vector store
            restored_vectors = temp_restore_dir / "vector_store"
            if restored_vectors.exists():
                target_vectors = Path(self.config.storage.chroma_dir)
                if target_vectors.exists():
                    shutil.rmtree(target_vectors)
                shutil.copytree(restored_vectors, target_vectors)

            return {"restored": True, "source": backup_path}
        finally:
            if temp_restore_dir.exists():
                shutil.rmtree(temp_restore_dir)

    def export_portable_json(self, export_path: Optional[str] = None) -> str:
        """Exports all memories, temporal beliefs, and corrections to human-readable JSON."""
        if not export_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_path = str(Path(self.config.storage.exports_dir) / f"capsule_export_{timestamp}.json")

        memories = [m.to_dict() for m in self.db.list_memories(limit=10000)]
        beliefs = [b.to_dict() for b in self.db.list_all_beliefs()]
        stats = self.db.get_archive_stats()

        export_data = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "archive_version": 1,
                "stats": stats
            },
            "temporal_beliefs": beliefs,
            "memories": memories
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        return export_path
