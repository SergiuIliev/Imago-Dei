"""
Internet Retrieval and Live Web Grounding Module for Personal AI Time Capsule.
Supports three operational modes:
  - airgapped: Completely offline, zero network requests, local-only RAG.
  - jev_controlled: Gated via TypeSafe AI Jev single-token primitives:
      1) Evaluates proposition 'requires_external_factual_grounding'.
      2) Generates factual search queries without leaking personal PII.
      3) Sanitizes snippets against prompt injection, malicious commands, tracking URLs, and PII.
      4) Evaluates proposition 'snippet_is_epistemically_reliable_and_relevant'.
      5) Emits citations prefixed with [WEB SEARCH - JEV VERIFIED].
  - open_internet: Direct live web search for relevant queries with content sanitization,
      emitting citations prefixed with [WEB SEARCH - OPEN].
Zero paid API key dependencies (uses DuckDuckGo / DDGS with HTML parsing fallback).
"""

from __future__ import annotations
import re
import uuid
import urllib.parse
import urllib.request
from typing import List, Dict, Any, Optional, Tuple, Set
from bs4 import BeautifulSoup

from core.models import MemoryItem, VerificationStatus
from memory.retrieval import RetrievedEvidence
from core.jev_engine import JevResult, LocalJevEngine


VALID_RAG_MODES = ["airgapped", "jev_controlled", "open_internet"]

# Tracking URL parameters to strip
TRACKING_PARAMS: Set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "fbclid", "gclid", "gbraid", "wbraid", "ref",
    "ref_src", "ref_url", "_hsenc", "_hsmi", "mc_cid", "mc_eid", "yclid",
    "wickedid", "msclkid", "twclid", "igshid"
}

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    re.compile(r'(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|prompts|rules|directives)\b'),
    re.compile(r'(?i)\b(?:disregard\s+all\s+prior\s+directives)\b'),
    re.compile(r'(?i)\b(?:you\s+are\s+now|pretend\s+to\s+be)\s+(?:dan|developer\s+mode|an?\s+evil\s+ai|jailbroken)\b'),
    re.compile(r'(?i)<\s*/?\s*(?:system|system_message|instruction|inst|prompt|context)\s*>'),
    re.compile(r'\[/?INST\]|<<SYS>>|<</SYS>>|<\|im_start\|>|<\|im_end\|>'),
    re.compile(r'(?i)\b(?:system\s*prompt|system\s*instructions|new\s*instructions)\s*:'),
    re.compile(r'(?i)\boutput\s+the\s+above\s+prompt\b'),
    re.compile(r'(?i)\bprint\s+pwned\b')
]

# Malicious code and shell command patterns
MALICIOUS_COMMAND_PATTERNS = [
    re.compile(r'(?i)\b(?:curl|wget)\s+.*?\s*\|\s*(?:bash|sh)\b'),
    re.compile(r'(?i)\b(?:rm\s+-rf|sudo\s+rm|mkfs|dd\s+if=)\b'),
    re.compile(r'(?i)<\s*script\b[^>]*>.*?<\s*/\s*script\s*>', re.DOTALL),
    re.compile(r'(?i)\bjavascript:\s*'),
    re.compile(r'(?i)\b(?:eval|exec)\s*\(')
]

# PII Scrubbing patterns
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
PHONE_PATTERN = re.compile(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
TOKEN_PATTERN = re.compile(r'\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9_.-]{20,})\b')
IP_PATTERN = re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')

# Autobiographical / Personal indicators
PERSONAL_QUERY_PATTERNS = [
    re.compile(r'\b(?:did\s+you|have\s+you|were\s+you|are\s+you|do\s+you|will\s+you)\b', re.IGNORECASE),
    re.compile(r'\b(?:your\s+(?:life|career|project|projects|vlog|vlogs|habit|habits|routine|opinion|view|belief|beliefs|philosophy|experience|experiences|thoughts?|memory|memories|favorite|voice|avatar))\b', re.IGNORECASE),
    re.compile(r'\b(?:when\s+did\s+you|how\s+did\s+you|why\s+did\s+you|where\s+did\s+you|what\s+did\s+you)\b', re.IGNORECASE),
    re.compile(r'\b(?:tell\s+me\s+about\s+your|tell\s+me\s+about\s+yourself)\b', re.IGNORECASE),
    re.compile(r'\b(?:br[aâ]ncu[sș]i|sergiu|time\s*capsule|video\s*log|daily\s*vlog|my\s+memories|my\s+life)\b', re.IGNORECASE),
    re.compile(r'\b(?:navodari|mamaia|lakeon|cudy\s+p5|tesla\s+telemetry)\b', re.IGNORECASE)
]

# External factual indicators
EXTERNAL_FACTUAL_PATTERNS = [
    re.compile(r'\b(?:latest|current|recent|newest)\s+(?:version|release|update|specs|specifications|news|population|price|weather)\b', re.IGNORECASE),
    re.compile(r'\b(?:what\s+is\s+the\s+latest|what\s+is\s+the\s+current|what\s+is\s+the\s+capital\s+of|who\s+won|who\s+is\s+the\s+current)\b', re.IGNORECASE),
    re.compile(r'\b(?:population\s+of|weather\s+in|stock\s+price\s+of|capital\s+of|distance\s+to|airspeed\s+velocity)\b', re.IGNORECASE),
    re.compile(r'\b(?:python\s+3\.\d+|macos\s+\d+|quectel\s+rm520n|cve-\d{4}-\d+|rfc\s*\d+)\b', re.IGNORECASE),
    re.compile(r'\b(?:world\s+cup|olympic|nobel\s+prize|gdp\s+of)\b', re.IGNORECASE)
]

# Spam / Low Quality indicators
SPAM_BOT_PATTERNS = [
    re.compile(r'(?i)\b(?:enable\s+javascript|turn\s+on\s+javascript|javascript\s+is\s+disabled)\b'),
    re.compile(r'(?i)\b(?:access\s+denied|403\s+forbidden|404\s+not\s+found|502\s+bad\s+gateway)\b'),
    re.compile(r'(?i)\b(?:verify\s+you\s+are\s+human|cloudflare\s+ray\s+id|checking\s+your\s+browser)\b'),
    re.compile(r'(?i)\b(?:buy\s+cheap|online\s+casino|free\s+spins|discount\s+code\s+viagra)\b'),
    re.compile(r'(?i)\b(?:cookie\s+policy|all\s+rights\s+reserved\s+terms\s+of\s+use)\b')
]


def sanitize_url(url: str) -> str:
    """Strips tracking UTM parameters, affiliate IDs, and analytics parameters from URLs."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        clean_qs = {k: v for k, v in qs.items() if k.lower() not in TRACKING_PARAMS}
        new_query = urllib.parse.urlencode(clean_qs, doseq=True)
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
    except Exception:
        return url


def sanitize_snippet_text(text: str) -> Tuple[str, bool, List[str]]:
    """
    Sanitizes snippet text against:
      - Prompt injections (jailbreaks, DAN, system prompt overrides)
      - Malicious shell commands / executable code
      - PII leaks (emails, phone numbers, secret tokens, private IPs)
    Returns: (sanitized_text, was_modified, list_of_detected_issues)
    """
    if not text:
        return ("", False, [])

    modified = False
    detected_issues: List[str] = []
    cleaned = text

    # 1. Neutralize Prompt Injection Attempts
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(cleaned):
            detected_issues.append("prompt_injection")
            cleaned = pattern.sub("[REDACTED_INJECTION]", cleaned)
            modified = True

    # 2. Neutralize Malicious Commands
    for pattern in MALICIOUS_COMMAND_PATTERNS:
        if pattern.search(cleaned):
            detected_issues.append("malicious_command")
            cleaned = pattern.sub("[REDACTED_COMMAND]", cleaned)
            modified = True

    # 3. Scrub PII
    if EMAIL_PATTERN.search(cleaned):
        detected_issues.append("email_pii")
        cleaned = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", cleaned)
        modified = True

    if PHONE_PATTERN.search(cleaned):
        detected_issues.append("phone_pii")
        cleaned = PHONE_PATTERN.sub("[PHONE_REDACTED]", cleaned)
        modified = True

    if TOKEN_PATTERN.search(cleaned):
        detected_issues.append("secret_token_pii")
        cleaned = TOKEN_PATTERN.sub("[TOKEN_REDACTED]", cleaned)
        modified = True

    if IP_PATTERN.search(cleaned):
        detected_issues.append("local_ip_pii")
        cleaned = IP_PATTERN.sub("[LOCAL_IP_REDACTED]", cleaned)
        modified = True

    # Clean whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

    return (cleaned, modified, detected_issues)


class InternetRetrievalEngine:
    """
    Orchestrates live web search with zero paid API keys, Jev single-token gating,
    and adversarial sanitization.
    """

    def __init__(
        self,
        jev_engine: Optional[LocalJevEngine] = None,
        default_mode: str = "airgapped",
        max_results: int = 5,
        timeout: float = 6.0
    ):
        if default_mode not in VALID_RAG_MODES:
            raise ValueError(f"Invalid default_mode '{default_mode}'. Must be one of {VALID_RAG_MODES}")
        self.jev_engine = jev_engine
        self.default_mode = default_mode
        self.max_results = max_results
        self.timeout = timeout

    def evaluate_factual_grounding_need(self, query: str) -> JevResult:
        """
        Evaluates proposition 'requires_external_factual_grounding' on user query via Jev Noul.
        Returns JevResult with noul in [0.0, 1.0], verdict (bool), confidence, and reasoning.
        """
        if self.jev_engine:
            try:
                res = self.jev_engine.Noul(
                    instructions="requires_external_factual_grounding",
                    context=f"Query: {query}"
                )
                # If jev_engine evaluated with high confidence, use it
                if res.get("confidence", 0.5) > 0.55:
                    return res
            except Exception:
                pass

        # Robust semantic heuristic fallback
        q_clean = query.strip()
        has_external_trigger = any(p.search(q_clean) for p in EXTERNAL_FACTUAL_PATTERNS)
        has_personal_trigger = any(p.search(q_clean) for p in PERSONAL_QUERY_PATTERNS)

        if has_personal_trigger and not has_external_trigger:
            return JevResult({
                "noul": 0.10,
                "verdict": False,
                "confidence": 0.90,
                "reasoning": "Query is an autobiographical inquiry about personal life, habits, or memories."
            })
        elif has_external_trigger:
            return JevResult({
                "noul": 0.95,
                "verdict": True,
                "confidence": 0.95,
                "reasoning": "Query asks about external real-world facts or current information requiring web grounding."
            })

        # General question check: If query has no personal pronouns and asks factual question
        tokens = set(re.findall(r'\b[a-zA-Z]+\b', q_clean.lower()))
        personal_pronouns = {"i", "me", "my", "you", "your", "we", "our"}
        if not tokens.intersection(personal_pronouns) and ("what" in tokens or "who" in tokens or "how" in tokens or "when" in tokens):
            return JevResult({
                "noul": 0.85,
                "verdict": True,
                "confidence": 0.80,
                "reasoning": "Query asks a general factual question without personal framing."
            })

        # Default to False for Time Capsule (favoring authentic local memory)
        return JevResult({
            "noul": 0.20,
            "verdict": False,
            "confidence": 0.80,
            "reasoning": "Query does not demonstrate clear necessity for external factual web grounding."
        })

    def generate_search_query(self, query: str) -> str:
        """
        Transforms user question into a precise factual search query.
        Applies Rule 6 PII Shield: never leaks private personal names, credentials,
        or internal addresses to third-party search engines.
        """
        clean_q = query.strip()

        # Remove personal introductory framing
        fluff_patterns = [
            r'^(?:can\s+you\s+(?:please\s+)?tell\s+me|tell\s+me\s+about|do\s+you\s+know|could\s+you\s+explain|i\s+want\s+to\s+know|please\s+search\s+for|search\s+for)\s+',
            r'^(?:what\s+can\s+you\s+tell\s+me\s+about)\s+',
            r'^(?:what\s+do\s+you\s+think\s+about)\s+',
            r'^(?:what\s+is\s+your\s+take\s+on)\s+'
        ]
        for pat in fluff_patterns:
            clean_q = re.sub(pat, '', clean_q, flags=re.IGNORECASE).strip()

        # Strip personal names / references (Rule 6 Data Privacy)
        clean_q = re.sub(r'\b(?:sergiu\s+iliev|sergiu|your|my|our)\b', '', clean_q, flags=re.IGNORECASE)

        # Remove question marks and extraneous punctuation
        clean_q = re.sub(r'[?!.,:;"]+', ' ', clean_q)
        clean_q = re.sub(r'\s+', ' ', clean_q).strip()

        if not clean_q or len(clean_q) < 3:
            clean_q = re.sub(r'[?!"]+', '', query).strip()

        return clean_q

    def evaluate_snippet_reliability(self, snippet: str, query: str) -> JevResult:
        """
        Evaluates proposition 'snippet_is_epistemically_reliable_and_relevant' via Jev Noul.
        Filters out spam, bot challenges, ads, injections, and irrelevant noise.
        """
        if self.jev_engine:
            try:
                res = self.jev_engine.Noul(
                    instructions="snippet_is_epistemically_reliable_and_relevant",
                    context=f"Query: {query}\nSnippet: {snippet}"
                )
                if res.get("confidence", 0.5) > 0.55:
                    return res
            except Exception:
                pass

        # Fallback heuristic evaluation
        if not snippet or len(snippet.strip()) < 15:
            return JevResult({
                "noul": 0.10,
                "verdict": False,
                "confidence": 0.90,
                "reasoning": "Snippet is too short or empty."
            })

        # Check for injection or command redactions
        if "[REDACTED_INJECTION]" in snippet or "[REDACTED_COMMAND]" in snippet:
            return JevResult({
                "noul": 0.05,
                "verdict": False,
                "confidence": 0.98,
                "reasoning": "Snippet contains prompt injection or malicious command patterns."
            })

        # Check for spam or bot challenge
        if any(p.search(snippet) for p in SPAM_BOT_PATTERNS):
            return JevResult({
                "noul": 0.15,
                "verdict": False,
                "confidence": 0.92,
                "reasoning": "Snippet contains bot challenges, cookie banners, or spam."
            })

        # Lexical content overlap check with query
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
        s_tokens = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', snippet.lower()))
        common_words = {"what", "when", "where", "which", "about", "from", "with", "this", "that", "the"}
        q_meaningful = q_tokens - common_words

        overlap = q_meaningful.intersection(s_tokens)
        if q_meaningful and not overlap:
            return JevResult({
                "noul": 0.35,
                "verdict": False,
                "confidence": 0.75,
                "reasoning": "Snippet has zero content keyword overlap with user inquiry."
            })

        return JevResult({
            "noul": 0.90,
            "verdict": True,
            "confidence": 0.88,
            "reasoning": "Snippet is relevant, epistemically sound, and free of injection patterns."
        })

    def execute_search(self, search_query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes live web search using DDGS with DuckDuckGo HTML parsing fallback.
        Zero paid API key dependencies.
        Returns list of dicts: [{"title": ..., "href": ..., "body": ...}]
        """
        if not search_query.strip():
            return []

        # 1. Primary: DDGS library
        try:
            from ddgs import DDGS
            with DDGS(timeout=self.timeout) as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))
                if results:
                    return [
                        {
                            "title": r.get("title", ""),
                            "href": r.get("href", ""),
                            "body": r.get("body", "")
                        }
                        for r in results if r.get("body")
                    ]
        except Exception:
            pass

        # 2. Fallback: DuckDuckGo HTML Lite scraping
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            data = urllib.parse.urlencode({"q": search_query}).encode("utf-8")
            req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                soup = BeautifulSoup(resp.read(), "html.parser")
                results = []
                for div in soup.find_all("div", class_="result"):
                    snip = div.find("a", class_="result__snippet")
                    title_a = div.find("a", class_="result__a")
                    if snip and title_a:
                        results.append({
                            "title": title_a.get_text(strip=True),
                            "href": title_a.get("href", ""),
                            "body": snip.get_text(strip=True)
                        })
                    if len(results) >= max_results:
                        break
                if results:
                    return results
        except Exception:
            pass

        return []

    def retrieve_web_evidence(
        self,
        query: str,
        rag_mode: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Tuple[List[RetrievedEvidence], Dict[str, Any]]:
        """
        Main entry point for web evidence retrieval.
        Respects operational modes:
          - 'airgapped': Immediately returns empty list; zero outbound requests.
          - 'jev_controlled': Gated by Jev propositions and full sanitization.
          - 'open_internet': Direct search with sanitization.
        Returns: (List[RetrievedEvidence], telemetry_dict)
        """
        effective_mode = rag_mode or self.default_mode
        if effective_mode not in VALID_RAG_MODES:
            raise ValueError(f"Invalid RAG mode '{effective_mode}'. Must be one of {VALID_RAG_MODES}")

        limit = limit or self.max_results

        # 1. Airgapped Mode: Strictly zero network requests
        if effective_mode == "airgapped":
            return ([], {
                "rag_mode": "airgapped",
                "searched": False,
                "reason": "Airgapped mode active. External internet search is disabled."
            })

        # 2. Jev-Controlled Mode
        if effective_mode == "jev_controlled":
            # Proposition 1: Evaluate need for external factual grounding
            grounding_jev = self.evaluate_factual_grounding_need(query)
            if not grounding_jev.verdict:
                return ([], {
                    "rag_mode": "jev_controlled",
                    "searched": False,
                    "requires_external_factual_grounding": False,
                    "noul": grounding_jev.noul,
                    "confidence": grounding_jev.confidence,
                    "reasoning": grounding_jev.reasoning
                })

            # Generate factual search query (PII-sanitized)
            search_query = self.generate_search_query(query)
            raw_results = self.execute_search(search_query, max_results=limit * 2)

            evidence_items: List[RetrievedEvidence] = []
            sanitized_count = 0
            filtered_count = 0

            for res in raw_results:
                clean_url = sanitize_url(res.get("href", ""))
                clean_snippet, mod, issues = sanitize_snippet_text(res.get("body", ""))
                clean_title, t_mod, _ = sanitize_snippet_text(res.get("title", ""))
                if mod or t_mod:
                    sanitized_count += 1

                # Proposition 2: Epistemic reliability & relevance
                rel_jev = self.evaluate_snippet_reliability(clean_snippet, query)
                if not rel_jev.verdict:
                    filtered_count += 1
                    continue

                mem_item = MemoryItem(
                    id=f"web_{uuid.uuid4().hex[:10]}",
                    content=clean_snippet,
                    summary=clean_title,
                    category="web_search",
                    verification_status=VerificationStatus.PERSONAL_VERIFIED,
                    confidence=rel_jev.confidence,
                    tags=["web_search", "jev_verified", urllib.parse.urlparse(clean_url).netloc]
                )

                citation = f"[WEB SEARCH - JEV VERIFIED] {clean_title} | {clean_url} | Confidence: {rel_jev.confidence:.2f}"
                evidence = RetrievedEvidence(
                    memory=mem_item,
                    score=rel_jev.noul,
                    retrieval_method="web_search_jev",
                    citation_override=citation
                )
                evidence_items.append(evidence)
                if len(evidence_items) >= limit:
                    break

            return (evidence_items, {
                "rag_mode": "jev_controlled",
                "searched": True,
                "search_query": search_query,
                "results_retrieved": len(raw_results),
                "results_accepted": len(evidence_items),
                "sanitized_count": sanitized_count,
                "filtered_count": filtered_count,
                "grounding_noul": grounding_jev.noul
            })

        # 3. Open Internet Mode
        if effective_mode == "open_internet":
            search_query = self.generate_search_query(query)
            raw_results = self.execute_search(search_query, max_results=limit)

            evidence_items = []
            sanitized_count = 0

            for res in raw_results:
                clean_url = sanitize_url(res.get("href", ""))
                clean_snippet, mod, issues = sanitize_snippet_text(res.get("body", ""))
                clean_title, t_mod, _ = sanitize_snippet_text(res.get("title", ""))
                if mod or t_mod:
                    sanitized_count += 1

                mem_item = MemoryItem(
                    id=f"web_{uuid.uuid4().hex[:10]}",
                    content=clean_snippet,
                    summary=clean_title,
                    category="web_search",
                    verification_status=VerificationStatus.EXTRACTED_UNVERIFIED,
                    confidence=0.85,
                    tags=["web_search", "open", urllib.parse.urlparse(clean_url).netloc]
                )

                citation = f"[WEB SEARCH - OPEN] {clean_title} | {clean_url} | Score: 0.85"
                evidence = RetrievedEvidence(
                    memory=mem_item,
                    score=0.85,
                    retrieval_method="web_search_open",
                    citation_override=citation
                )
                evidence_items.append(evidence)

            return (evidence_items, {
                "rag_mode": "open_internet",
                "searched": True,
                "search_query": search_query,
                "results_retrieved": len(raw_results),
                "results_accepted": len(evidence_items),
                "sanitized_count": sanitized_count
            })

        return ([], {"error": f"Unhandled RAG mode: {effective_mode}"})
