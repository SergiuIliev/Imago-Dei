#!/usr/bin/env python3
"""
Public Secure Double Audit Tool for Personal AI Time Capsule.
Zero Info-Leak Enforcement Engine (Layer 1 Static Audit + Layer 2 Runtime Audit).

Usage:
    python scripts/double_audit.py [OPTIONS]
    python scripts/double_audit.py --strict
    python scripts/double_audit.py --check-git
    python scripts/double_audit.py --export-report markdown
    python scripts/double_audit.py --fix
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

# Project Root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# AUDIT PATTERNS & REGEX SUITE
# ==============================================================================

# Regex definitions for Secret Tokens & Keys
SECRET_PATTERNS: Dict[str, re.Pattern] = {
    "openai_api_key": re.compile(r"(?<![a-zA-Z0-9_-])(?:sk-[a-zA-Z0-9]{20,}|sk-proj-[a-zA-Z0-9_-]{20,})"),
    "anthropic_api_key": re.compile(r"(?<![a-zA-Z0-9_-])sk-ant-[a-zA-Z0-9_-]{20,}"),
    "huggingface_token": re.compile(r"(?<![a-zA-Z0-9_-])hf_[a-zA-Z0-9]{30,}"),
    "github_pat": re.compile(r"(?<![a-zA-Z0-9_-])(?:ghp_[a-zA-Z0-9]{36,}|github_pat_[a-zA-Z0-9_]{50,})"),
    "cloudflare_tunnel_jwt": re.compile(r"(?<![a-zA-Z0-9_-])eyJhIjoi[a-zA-Z0-9_\-\.\=]{40,}"),
    "private_key_pem": re.compile(r"-----BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PGP|ENCRYPTED|PRIVATE)?\s*KEY-----"),
    "generic_assigned_secret": re.compile(
        r'(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*["\']([a-zA-Z0-9_\-\.]{24,})["\']'
    ),
}

# Regex definitions for Personally Identifiable Information (PII)
PII_PATTERNS: Dict[str, re.Pattern] = {
    "personal_email_sj": re.compile(r"(?i)[a-zA-Z0-9_.+-]+@sj-webdesign\.com"),
    "personal_email_sergiu": re.compile(r"(?i)\b(?:sergiu\.iliev|sergiuiliev)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "phone_romanian": re.compile(r"(?:\+40|0040)[-.\s]?(?:7\d{2})[-.\s]?\d{3}[-.\s]?\d{3}\b"),
    "phone_swiss": re.compile(r"(?:\+41|0041)[-.\s]?(?:7\d)[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b"),
    "wifi_password": re.compile(r'(?i)(?:wifi_password|wpa_passphrase|wifi_pass)\s*[:=]\s*["\']([^"\']{6,})["\']'),
}

# Regex definitions for Hardcoded Local User Home Paths
HOME_PATH_PATTERNS: Dict[str, re.Pattern] = {
    "macos_user_home": re.compile(r"/(?:Users)/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*"),
    "windows_user_home": re.compile(r"[A-Za-z]:[\\/]Users[\\/][a-zA-Z0-9_.-]+(?:[\\/][a-zA-Z0-9_.-]+)*"),
    "linux_user_home": re.compile(r"/home/(?!(?:runner|circleci|travis|appuser|node|git|default))[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*"),
}

# Safe placeholders allowed in examples, templates, or test assertions
SAFE_PLACEHOLDERS: Set[str] = {
    "placeholder",
    "dummy",
    "replace_me",
    "your-api-key",
    "your_secret_key",
    "sk-proj-placeholder",
    "hf_placeholder",
    "eyjhijoiplaceholder",
    "00000000-0000-0000-0000-000000000000",
}

# Binary file extensions and directories ignored by static scanner
IGNORED_DIRS: Set[str] = {
    ".git",
    "venv",
    ".venv",
    "venv-webui",
    "venv-voice",
    "venv-mlx",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".DS_Store",
    ".idea",
    ".vscode",
    "Toggle Time Capsule.app",
    "Start Time Capsule.app",
    "Stop Time Capsule.app",
    "Time Capsule.app",
    "data",
    "logs",
    "exports",
    "backups",
}

BINARY_EXTENSIONS: Set[str] = {
    ".pyc", ".db", ".sqlite", ".sqlite3", ".icns", ".png", ".jpg", ".jpeg",
    ".dylib", ".so", ".applet", ".rsrc", ".bin", ".pt", ".safetensors",
    ".tar.gz", ".zip", ".wav", ".mp3", ".mp4", ".mov", ".mkv", ".webm",
    ".onnx", ".gguf"
}


@dataclass
class AuditFinding:
    category: str  # "SECRET", "PII", "HOME_PATH", "TUNNEL_CRED"
    file_path: str
    line_number: int
    matched_pattern: str
    redacted_snippet: str
    is_blocking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    total_files_scanned: int = 0
    total_findings: int = 0
    blocking_findings: int = 0
    findings: List[AuditFinding] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files_scanned": self.total_files_scanned,
            "total_findings": self.total_findings,
            "blocking_findings": self.blocking_findings,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings]
        }


# ==============================================================================
# LAYER 2: RUNTIME & SEMANTIC LEAKAGE AUDIT ENGINE
# ==============================================================================

def sanitize_text(text: str) -> str:
    """
    Masks or redacts accidental PII, credentials, API keys, and hardcoded home paths.
    Safe for production logs, external LLM prompts, and public exports.
    """
    if not isinstance(text, str) or not text:
        return text

    sanitized = text

    # 1. Redact Private Key blocks
    sanitized = re.sub(
        r"-----BEGIN\s+(?:[A-Z0-9_-]+ )?KEY-----[\s\S]*?-----END\s+(?:[A-Z0-9_-]+ )?KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        sanitized
    )

    # 2. Redact Known Secret Formats
    for name, pattern in SECRET_PATTERNS.items():
        if name == "private_key_pem":
            continue
        sanitized = pattern.sub(f"[REDACTED_{name.upper()}]", sanitized)

    # 3. Redact PII
    for name, pattern in PII_PATTERNS.items():
        sanitized = pattern.sub(f"[REDACTED_{name.upper()}]", sanitized)

    # 4. Redact Hardcoded Home Paths
    for name, pattern in HOME_PATH_PATTERNS.items():
        sanitized = pattern.sub("[PORTABLE_USER_PATH]", sanitized)

    return sanitized


def audit_rag_context(context_text: str) -> Tuple[bool, List[str]]:
    """
    Audits retrieved RAG context text for secret credentials, forbidden PII, or raw home paths.
    Returns (is_clean, violations).
    """
    if not isinstance(context_text, str) or not context_text:
        return True, []

    violations: List[str] = []

    # Check secrets
    for name, pat in SECRET_PATTERNS.items():
        matches = pat.findall(context_text)
        for m in matches:
            m_str = m if isinstance(m, str) else str(m)
            if not any(ph in m_str.lower() for ph in SAFE_PLACEHOLDERS):
                violations.append(f"Secret detected ({name}): {m_str[:6]}...{m_str[-4:] if len(m_str)>10 else ''}")

    # Check PII
    for name, pat in PII_PATTERNS.items():
        matches = pat.findall(context_text)
        for m in matches:
            m_str = m if isinstance(m, str) else str(m)
            violations.append(f"PII detected ({name}): {m_str}")

    # Check user home paths
    for name, pat in HOME_PATH_PATTERNS.items():
        matches = pat.findall(context_text)
        for m in matches:
            if not any(ph in m.lower() for ph in SAFE_PLACEHOLDERS):
                violations.append(f"Hardcoded home path detected ({name}): {m}")

    is_clean = len(violations) == 0
    return is_clean, violations


def audit_outgoing_payload(payload: Any) -> Tuple[bool, List[str]]:
    """
    Deep audits an outgoing payload dictionary or list for potential info leaks.
    Traverses nested structures recursively.
    Returns (is_clean, violations).
    """
    violations: List[str] = []

    def _traverse(val: Any, path: str):
        if isinstance(val, str):
            clean, viols = audit_rag_context(val)
            for v in viols:
                violations.append(f"At '{path}': {v}")
        elif isinstance(val, dict):
            for k, v in val.items():
                _traverse(k, f"{path}.<key>")
                _traverse(v, f"{path}.{k}")
        elif isinstance(val, (list, tuple, set)):
            for i, item in enumerate(val):
                _traverse(item, f"{path}[{i}]")

    _traverse(payload, "root")
    return (len(violations) == 0, violations)


# ==============================================================================
# LAYER 1: STATIC CODEBASE AUDIT ENGINE
# ==============================================================================

class DoubleAuditor:
    """
    Orchestrates static audits across tracked files, codebase directories,
    and Git diffs, enforcing zero info leakage.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        strict: bool = False,
        scan_git_only: bool = False
    ):
        self.root_dir = (root_dir or PROJECT_ROOT).resolve()
        self.strict = strict
        self.scan_git_only = scan_git_only

    def is_binary_or_ignored(self, path: Path) -> bool:
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True
        for part in path.parts:
            if part in IGNORED_DIRS:
                return True
        # Check for binary null byte in head
        try:
            with open(path, "rb") as f:
                chunk = f.read(4096)
                if b"\x00" in chunk:
                    return True
        except Exception:
            return True
        return False

    def get_target_files(self) -> List[Path]:
        """Collects files to audit, respecting git tracking or direct filesystem scan."""
        files: List[Path] = []

        if self.scan_git_only:
            try:
                res = subprocess.run(
                    ["git", "ls-files"],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in res.stdout.strip().splitlines():
                    p = self.root_dir / line.strip()
                    if p.is_file() and not self.is_binary_or_ignored(p):
                        files.append(p)
                return sorted(files)
            except Exception:
                pass

        # Fallback / full filesystem scan
        for p in self.root_dir.rglob("*"):
            if p.is_file() and not self.is_binary_or_ignored(p):
                files.append(p)

        return sorted(files)

    def scan_file_content(self, file_path: Path, content: str) -> List[AuditFinding]:
        """Audits content of a single text file line by line."""
        findings: List[AuditFinding] = []
        rel_path = str(file_path.relative_to(self.root_dir) if file_path.is_relative_to(self.root_dir) else file_path)

        # Allow audit script itself and its unit test suite to define regexes and synthetic canaries
        if "double_audit" in rel_path:
            return []

        is_test = rel_path.startswith("tests/") or rel_path.endswith("_test.py") or rel_path.startswith("tests\\")

        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            # Inline suppression check
            if "# audit:ignore" in line or "# pragma: no-leak" in line:
                continue

            # 1. Check Secrets
            for name, pat in SECRET_PATTERNS.items():
                for match in pat.finditer(line):
                    match_val = match.group(0)
                    if any(ph in match_val.lower() for ph in SAFE_PLACEHOLDERS):
                        continue
                    if is_test and (
                        "r\"" in line or "r'" in line or "test_" in line or
                        "self.assert" in line or "dirty" in line or "mock" in line.lower() or
                        "sk-1234" in match_val
                    ):
                        continue
                    findings.append(AuditFinding(
                        category="SECRET",
                        file_path=rel_path,
                        line_number=idx,
                        matched_pattern=name,
                        redacted_snippet=f"{match_val[:4]}...{match_val[-4:] if len(match_val)>8 else ''}",
                        is_blocking=True
                    ))

            # 2. Check PII
            for name, pat in PII_PATTERNS.items():
                for match in pat.finditer(line):
                    match_val = match.group(0)
                    if any(ph in match_val.lower() for ph in SAFE_PLACEHOLDERS):
                        continue
                    if is_test and (
                        "r\"" in line or "r'" in line or "test_" in line or
                        "self.assert" in line or "dirty" in line or "mock" in line.lower() or
                        "sk-1234" in match_val
                    ):
                        continue
                    findings.append(AuditFinding(
                        category="PII",
                        file_path=rel_path,
                        line_number=idx,
                        matched_pattern=name,
                        redacted_snippet=match_val[:3] + "***@" if "@" in match_val else match_val[:4] + "***",
                        is_blocking=True
                    ))

            # 3. Check Hardcoded Home Paths
            for name, pat in HOME_PATH_PATTERNS.items():
                for match in pat.finditer(line):
                    match_val = match.group(0)
                    if any(ph in match_val.lower() for ph in SAFE_PLACEHOLDERS):
                        continue
                    if is_test:
                        continue
                    # Ignore documentation references to generic examples if marked
                    findings.append(AuditFinding(
                        category="HOME_PATH",
                        file_path=rel_path,
                        line_number=idx,
                        matched_pattern=name,
                        redacted_snippet=match_val,
                        is_blocking=self.strict
                    ))

        return findings

    def run_audit(self) -> AuditReport:
        """Executes Layer 1 static audit across all target files."""
        report = AuditReport()
        files = self.get_target_files()
        report.total_files_scanned = len(files)

        for p in files:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                file_findings = self.scan_file_content(p, content)
                for f_item in file_findings:
                    report.findings.append(f_item)
                    report.total_findings += 1
                    if f_item.is_blocking:
                        report.blocking_findings += 1
            except Exception as e:
                print(f"[DoubleAudit] Warning: Could not read {p}: {e}")

        report.passed = (report.blocking_findings == 0) if self.strict else (
            not any(f.category in ("SECRET", "PII") for f in report.findings)
        )
        return report

    def audit_git_changes(self) -> List[AuditFinding]:
        """Audits uncommitted git changes and staged diffs for accidental leaks."""
        findings: List[AuditFinding] = []
        try:
            # Check untracked files
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True
            )
            for line in status_res.stdout.splitlines():
                if line.startswith("??"):
                    untracked_path = self.root_dir / line[3:].strip()
                    if untracked_path.name in (".env", "tunnel_token.txt", "secret_key.txt") or untracked_path.suffix in (".db", ".sqlite"):
                        findings.append(AuditFinding(
                            category="SECRET",
                            file_path=str(untracked_path.relative_to(self.root_dir)),
                            line_number=0,
                            matched_pattern="untracked_sensitive_file",
                            redacted_snippet=f"Untracked sensitive file {untracked_path.name}",
                            is_blocking=True
                        ))

            # Check diff of tracked files
            diff_res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True
            )
            current_diff_file = ""
            for line in diff_res.stdout.splitlines():
                if line.startswith("diff --git"):
                    current_diff_file = line.split()[-1]
                    continue
                if "double_audit" in current_diff_file:
                    continue
                is_test_file = "test" in current_diff_file.lower()
                if line.startswith("+") and not line.startswith("+++"):
                    added_line = line[1:]
                    if "# audit:ignore" in added_line or "# pragma: no-leak" in added_line:
                        continue
                    if any(ph in added_line.lower() for ph in SAFE_PLACEHOLDERS):
                        continue
                    if is_test_file and ("self.assert" in added_line or "sk-1234" in added_line or "dirty" in added_line):
                        continue
                    for name, pat in SECRET_PATTERNS.items():
                        if pat.search(added_line):
                            findings.append(AuditFinding(
                                category="SECRET",
                                file_path=f"git:diff:{current_diff_file}",
                                line_number=0,
                                matched_pattern=name,
                                redacted_snippet="Added secret detected in git diff",
                                is_blocking=True
                            ))
                    for name, pat in PII_PATTERNS.items():
                        if pat.search(added_line):
                            findings.append(AuditFinding(
                                category="PII",
                                file_path=f"git:diff:{current_diff_file}",
                                line_number=0,
                                matched_pattern=name,
                                redacted_snippet="Added PII detected in git diff",
                                is_blocking=True
                            ))
        except Exception as e:
            print(f"[DoubleAudit] Git check warning: {e}")

        return findings

    def fix_hardcoded_paths(self) -> int:
        """
        Replaces detected hardcoded user paths with portable environment-relative paths.
        Returns the number of files modified.
        """
        files = self.get_target_files()
        modified_count = 0

        # Patterns to replace
        replace_pairs = [
            (re.compile(r"/Users/Sergiu/AI/Time_Capsule/data/capsule"), "data/capsule"),
            (re.compile(r"/Users/Sergiu/AI/Time_Capsule/vlogs"), "vlogs"),
            (re.compile(r"/Users/Sergiu/AI/Time_Capsule/exports"), "exports"),
            (re.compile(r"/Users/Sergiu/AI/Time_Capsule"), "."),
            (re.compile(r"/Users/Sergiu/AI/models"), "models"),
            (re.compile(r"/Users/Sergiu/AI/config"), "config"),
            (re.compile(r"/Users/Sergiu/AI/data"), "data"),
            (re.compile(r"/Users/Sergiu/AI/logs"), "logs"),
            (re.compile(r"/Users/Sergiu/AI/bin"), "bin"),
            (re.compile(r"/Users/Sergiu/AI/research"), "research"),
            (re.compile(r"/Users/Sergiu/AI/venv-webui/bin/python3?"), "python3"),
            (re.compile(r"/Users/Sergiu/AI/venv-webui"), "venv"),
            (re.compile(r"/Users/Sergiu/AI/venv-voice/bin/python"), "python3"),
            (re.compile(r"/Users/Sergiu/AI/venv-voice/bin/ffmpeg"), "ffmpeg"),
            (re.compile(r"/Users/Sergiu/\.local/bin/ollama"), "ollama"),
            (re.compile(r"/Users/Sergiu/\.local/bin/python3\.11"), "python3"),
            (re.compile(r"/Users/Sergiu/\.cache/chroma"), "~/.cache/chroma"),
            (re.compile(r"/Users/Sergiu/\.cache/huggingface"), "~/.cache/huggingface"),
            (re.compile(r"/Users/Sergiu/AI"), "~/AI"),
            (re.compile(r"/Users/Sergiu/Movies/Vlogs"), "~/Movies/Vlogs"),
        ]

        for p in files:
            # Skip double_audit script itself and git configs
            if p == Path(__file__).resolve() or ".git" in p.parts:
                continue

            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()

                new_content = content
                for pat, repl in replace_pairs:
                    new_content = pat.sub(repl, new_content)

                if new_content != content:
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    modified_count += 1
                    print(f"[DoubleAudit --fix] Sanitized paths in: {p.relative_to(self.root_dir)}")
            except Exception as e:
                print(f"[DoubleAudit --fix] Error processing {p}: {e}")

        return modified_count


# ==============================================================================
# CLI INTERFACE & REPORT FORMATTERS
# ==============================================================================

def format_markdown_report(report: AuditReport) -> str:
    md = []
    md.append("# 🛡️ Personal AI Time Capsule - Double Audit Report")
    md.append("")
    md.append(f"- **Status**: {'✅ PASSED (0 Leaks)' if report.passed else '❌ FAILED (Leaks Detected)'}")
    md.append(f"- **Files Scanned**: {report.total_files_scanned}")
    md.append(f"- **Total Findings**: {report.total_findings}")
    md.append(f"- **Blocking Issues**: {report.blocking_findings}")
    md.append("")

    if not report.findings:
        md.append("✨ **No secret credentials, PII leaks, or forbidden paths were found in the audited files.**")
    else:
        md.append("| Category | File | Line | Rule | Snippet | Blocking |")
        md.append("| :--- | :--- | :---: | :--- | :--- | :---: |")
        for f in report.findings:
            block_icon = "🛑 YES" if f.is_blocking else "⚠️ NO"
            md.append(f"| `{f.category}` | `{f.file_path}` | {f.line_number} | `{f.matched_pattern}` | `{f.redacted_snippet}` | {block_icon} |")

    md.append("")
    md.append("---")
    md.append("*Generated by `scripts/double_audit.py` (Zero Info-Leak Double Audit Engine)*")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Personal AI Time Capsule - Double Audit Zero Info-Leak Enforcement"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict mode (fail with exit code 1 on any warning/error)"
    )
    parser.add_argument(
        "--check-git",
        action="store_true",
        help="Scan git uncommitted changes, untracked files, and diff"
    )
    parser.add_argument(
        "--export-report",
        choices=["json", "markdown"],
        help="Export audit report in specified format"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="",
        help="Output file path for report export (defaults to stdout or standard name)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically replace detected hardcoded user paths with portable relative paths"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(PROJECT_ROOT),
        help="Root path to audit (defaults to PROJECT_ROOT)"
    )

    args = parser.parse_args()

    auditor = DoubleAuditor(
        root_dir=Path(args.path),
        strict=args.strict,
        scan_git_only=False
    )

    # Optional Fix step first
    if args.fix:
        print("[DoubleAudit] Running path sanitization (--fix)...")
        modified = auditor.fix_hardcoded_paths()
        print(f"[DoubleAudit] Path sanitization complete: {modified} files updated.\n")

    # Run Static Audit
    print(f"[DoubleAudit] Starting Static Audit (Layer 1) across {auditor.root_dir}...")
    report = auditor.run_audit()

    # Optional Git diff check
    if args.check_git:
        print("[DoubleAudit] Scanning Git status and uncommitted diffs...")
        git_findings = auditor.audit_git_changes()
        for gf in git_findings:
            report.findings.append(gf)
            report.total_findings += 1
            if gf.is_blocking:
                report.blocking_findings += 1
        if git_findings and args.strict:
            report.passed = False

    # Summary
    print("\n========================================================")
    print(" Double Audit Zero Info-Leak Summary")
    print("========================================================")
    print(f" Files Scanned   : {report.total_files_scanned}")
    print(f" Total Findings  : {report.total_findings}")
    print(f" Blocking Issues : {report.blocking_findings}")
    print(f" Final Verdict   : {'PASSED ✅' if report.passed else 'FAILED ❌'}")
    print("========================================================\n")

    if report.findings:
        for f in report.findings:
            icon = "🛑" if f.is_blocking else "⚠️"
            print(f" {icon} [{f.category}] {f.file_path}:{f.line_number} -> {f.matched_pattern} ({f.redacted_snippet})")
        print("")

    # Export report if requested
    if args.export_report == "json":
        out_content = json.dumps(report.to_dict(), indent=2)
        if args.report_file:
            with open(args.report_file, "w", encoding="utf-8") as f:
                f.write(out_content)
            print(f"[DoubleAudit] JSON report exported to {args.report_file}")
        else:
            print(out_content)
    elif args.export_report == "markdown":
        out_content = format_markdown_report(report)
        if args.report_file:
            with open(args.report_file, "w", encoding="utf-8") as f:
                f.write(out_content)
            print(f"[DoubleAudit] Markdown report exported to {args.report_file}")
        else:
            print(out_content)

    if not report.passed and args.strict:
        sys.exit(1)
    elif not report.passed:
        # If not strict but secrets/PII were found, fail
        if any(f.category in ("SECRET", "PII") for f in report.findings):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
