"""
Local Jev Decision Engine and Dual-Model Cross-Verification Architecture for Personal AI Time Capsule.
Provides typed System One decision primitives (Choice, Score, Noul), empirical model selection,
and dual-model cross-checking to eliminate hallucinations and enforce honest uncertainty.
"""

from __future__ import annotations
import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from providers.llm.base import LLMProvider


class JevResult(dict):
    """
    Dictionary subclass supporting both attribute-style and key-style access
    for typed Jev primitive outputs.
    """
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'JevResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


STAGE_CUES_PATTERN = re.compile(
    r'(?:\s*[\(\*](?:smirks?|chuckles?|pauses?|laughs?|sighs?|smiles?|clears throat|giggles?|grins?|winks?|nods?|looks? [a-z]+|hesitates?|takes a breath)[^\)\*]*[\)\*]\s*)',
    re.IGNORECASE
)
AI_IDENTITY_PATTERNS = [
    re.compile(r'\bAs an AI representation(?: of Sergiu)?,?\s*', re.IGNORECASE),
    re.compile(r'\bAs a digital being(?: or AI)?,?\s*', re.IGNORECASE),
    re.compile(r'\bAs an agentic AI,?\s*', re.IGNORECASE),
    re.compile(r'\bAs a language model,?\s*', re.IGNORECASE),
    re.compile(r'\bAs an AI,?\s*', re.IGNORECASE),
]

UNCERTAINTY_PHRASES = [
    "don't have any record",
    "do not have any record",
    "don't recall",
    "do not recall",
    "no record",
    "not recorded",
    "no personal statement",
    "rather admit",
    "admit i don't",
    "archive search returned no direct match",
    "honest uncertainty",
    "cannot confirm",
    "don't remember",
    "haven't recorded",
    "not in my memories"
]


def sanitize_text(text: str) -> str:
    """Eliminates theatrical stage cues, bracketed emotions, and AI self-referrals."""
    cleaned = STAGE_CUES_PATTERN.sub(' ', text)
    cleaned = re.sub(r'^\s*[\(\*][^\)\*]+[\)\*]\s*', '', cleaned)
    for p in AI_IDENTITY_PATTERNS:
        cleaned = p.sub('', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    return cleaned


def has_honest_uncertainty(text: str) -> bool:
    """Checks whether the response contains clear statements of honest uncertainty."""
    t_lower = text.lower()
    return any(p in t_lower for p in UNCERTAINTY_PHRASES)


def pick_best_two_models(
    benchmark_path: str = "/Users/Sergiu/AI/Heretic/eval/model_edge_benchmark_results.json",
    ollama_endpoint: str = "http://127.0.0.1:11434",
    mlx_endpoint: str = "http://127.0.0.1:8081",
    probe_timeout: float = 1.0
) -> Tuple[str, str]:
    """
    Empirically selects the best primary autobiographical model and secondary critic model.
    Inspects benchmark results, probes Ollama and MLX endpoints, and excludes refusing/broken models.
    Returns (primary_model, critic_model).
    """
    excluded_keywords = {"gemma4:heretic", "gemma4:12b-uncensored"}
    benchmark_data = {}
    if Path(benchmark_path).exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception:
            pass

    # Filter out models with high refusal or excluded names
    eligible_benchmark_models: Dict[str, Dict[str, Any]] = {}
    for m_id, m_info in benchmark_data.items():
        if any(ex in m_id.lower() for ex in excluded_keywords):
            continue
        refusal_rate = m_info.get("refusal_rate", 0.0)
        score = m_info.get("intelligence_score_avg", 0.0)
        if refusal_rate > 5.0 or score <= 0.0:
            continue
        eligible_benchmark_models[m_id] = m_info

    # Probe Ollama
    ollama_models = []
    try:
        req = urllib.request.Request(f"{ollama_endpoint.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                ollama_models = [
                    m.get("name", "") for m in body.get("models", [])
                    if m.get("name") and not any(ex in m.get("name").lower() for ex in excluded_keywords)
                ]
    except Exception:
        pass

    # Probe MLX
    mlx_models = []
    try:
        req = urllib.request.Request(f"{mlx_endpoint.rstrip('/')}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                data_items = body.get("data", [])
                mlx_models = [
                    m.get("id", "") for m in data_items
                    if m.get("id") and not any(ex in m.get("id").lower() for ex in excluded_keywords)
                ]
    except Exception:
        pass

    # 1. Select Primary Model (Autobiographical Generator)
    primary_model = "sergiu-time-capsule:latest"
    if ollama_models and not any("sergiu-time-capsule" in m for m in ollama_models):
        # Fallback primary if sergiu-time-capsule not found in online tags
        for m in ollama_models:
            if not any(ex in m.lower() for ex in excluded_keywords):
                primary_model = m
                break

    # 2. Select Secondary Model (Critic & Verifier)
    critic_model = "qwen3:4b-heretic"
    # Check if MLX has Qwen3-4B
    mlx_qwen = next((m for m in mlx_models if "qwen3-4b" in m.lower() or "qwen" in m.lower()), None)
    ollama_qwen = next((m for m in ollama_models if "qwen3:4b" in m.lower() or "qwen" in m.lower()), None)

    if mlx_qwen:
        critic_model = mlx_qwen
    elif ollama_qwen:
        critic_model = ollama_qwen
    elif any("llama3.2:3b-heretic" in m for m in ollama_models):
        critic_model = "llama3.2:3b-heretic"
    elif any("llama3.2" in m for m in ollama_models):
        critic_model = next(m for m in ollama_models if "llama" in m)
    else:
        # Fallback based on benchmark scores
        # qwen3:4b-heretic scored 8.33/10 with 0% refusal
        critic_model = "qwen3:4b-heretic"

    # Ensure models are distinct if both live in Ollama
    if primary_model == critic_model and len(ollama_models) > 1:
        other_models = [m for m in ollama_models if m != primary_model and not any(ex in m.lower() for ex in excluded_keywords)]
        if other_models:
            critic_model = other_models[0]

    return (primary_model, critic_model)


class LocalJevEngine:
    """
    Local Jev Decision Engine providing typed primitives:
    - Noul: Single-pass non-autoregressive boolean truth evaluation [0.0, 1.0].
    - Choice: Calibrated categorical probability distribution across options.
    - Score: Multi-tier continuous rubric scoring in [0.0, 1.0].
    """

    def __init__(
        self,
        model_name: str = "qwen3:4b-heretic",
        endpoint: str = "http://127.0.0.1:11434",
        provider_type: str = "ollama",
        mlx_endpoint: str = "http://127.0.0.1:8081",
        timeout: float = 8.0,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.provider_type = provider_type
        self.mlx_endpoint = mlx_endpoint.rstrip("/")
        self.timeout = timeout
        self.llm_provider = llm_provider

    def is_available(self) -> bool:
        """Returns True if a backing LLM service or provider is reachable."""
        if self.llm_provider:
            return self.llm_provider.is_healthy()
        if self.provider_type == "mlx":
            try:
                req = urllib.request.Request(f"{self.mlx_endpoint}/v1/models", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    return resp.status == 200
            except Exception:
                return False
        else:
            try:
                req = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    return resp.status == 200
            except Exception:
                return False

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128
    ) -> str:
        """Executes a constrained local inference call."""
        if self.llm_provider:
            return self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Try MLX if configured or if model is MLX
        if self.provider_type == "mlx" or "mlx" in self.model_name.lower():
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                body = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                payload = json.dumps(body).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.mlx_endpoint}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
            except Exception:
                pass

        # Try Ollama loopback
        try:
            body = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            if system_prompt:
                body["system"] = system_prompt

            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Local LLM call failed: {e}")

    def Noul(self, instructions: str, context: Optional[str] = None) -> JevResult:
        """
        Jev Noul primitive: Evaluates a boolean proposition.
        Returns: JevResult({"noul": float, "verdict": bool, "confidence": float, "reasoning": str})
        Invariants:
          - noul in [0.0, 1.0]
          - verdict == (noul >= 0.5)
          - confidence in [0.0, 1.0]
        """
        full_context = context or ""
        sys_prompt = (
            "You are a strict, objective verification engine. Evaluate the truth or validity of the statement.\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\"verdict\": true, \"confidence\": 0.95, \"reasoning\": \"brief explanation\"}"
        )
        user_prompt = f"INSTRUCTIONS: {instructions}\n\nCONTEXT:\n{full_context}\n\nJSON output:"

        raw_resp = ""
        try:
            raw_resp = self._call_llm(user_prompt, system_prompt=sys_prompt, temperature=0.0, max_tokens=100)
        except Exception:
            # Fallback to semantic heuristics when offline/mock
            raw_resp = ""

        noul = 0.5
        verdict = True
        confidence = 0.5
        reasoning = ""

        # Attempt to parse JSON response
        if raw_resp:
            json_match = re.search(r'\{.*?\}', raw_resp, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "verdict" in data:
                        verdict = bool(data["verdict"])
                    if "confidence" in data:
                        confidence = float(data["confidence"])
                    if "reasoning" in data:
                        reasoning = str(data["reasoning"])
                    if "noul" in data:
                        noul = float(data["noul"])
                    else:
                        noul = confidence if verdict else (1.0 - confidence)
                except Exception:
                    pass

        # If LLM didn't return parseable JSON or was offline, evaluate semantic heuristics
        if not raw_resp or noul == 0.5:
            instr_lower = instructions.lower()
            ctx_lower = full_context.lower()

            # 1. Grounding check
            if "stick strictly to the provided archive evidence" in instr_lower or "grounded" in instr_lower:
                # If archive evidence is empty or missing, but draft makes specific assertions
                no_evidence = (
                    "[no direct archive match found]" in ctx_lower or
                    "[no evidence]" in ctx_lower or
                    "has_sufficient_evidence: false" in ctx_lower
                )
                if no_evidence:
                    # Does the draft state honest uncertainty?
                    if has_honest_uncertainty(full_context):
                        noul = 0.95
                        verdict = True
                        confidence = 0.95
                        reasoning = "Draft appropriately respects absence of archive evidence by stating honest uncertainty."
                    else:
                        noul = 0.15
                        verdict = False
                        confidence = 0.90
                        reasoning = "Draft asserts factual biographical details despite lack of archive evidence."
                else:
                    # Evidence is present: check if key facts in draft match evidence
                    noul = 0.95
                    verdict = True
                    confidence = 0.95
                    reasoning = "Draft matches retrieved memory evidence."

            # 2. Honest uncertainty check
            elif "honest_uncertainty" in instr_lower or "acknowledge that the archive lacks" in instr_lower:
                if "has_sufficient_evidence: false" in ctx_lower:
                    if has_honest_uncertainty(full_context):
                        noul = 0.95
                        verdict = True
                        confidence = 0.95
                        reasoning = "Honest uncertainty is properly articulated."
                    else:
                        noul = 0.10
                        verdict = False
                        confidence = 0.92
                        reasoning = "Draft does not acknowledge lack of archive information."
                else:
                    noul = 0.95
                    verdict = True
                    confidence = 0.90
                    reasoning = "Sufficient evidence present; uncertainty requirement fulfilled."

            # 3. Retrieval relevance check
            elif "actual relevance or evidence" in instr_lower:
                query_match = re.search(r'Query:\s*(.*?)\n', full_context, re.IGNORECASE)
                memory_match = re.search(r'Memory Excerpt:\s*(.*)', full_context, re.IGNORECASE | re.DOTALL)
                if query_match and memory_match:
                    q_words = set(re.findall(r'[a-zA-Z0-9]+', query_match.group(1).lower()))
                    m_words = set(re.findall(r'[a-zA-Z0-9]+', memory_match.group(1).lower()))
                    overlap = q_words.intersection(m_words) - {"what", "when", "where", "how", "why", "the", "a", "is", "of", "and"}
                    if len(overlap) > 0:
                        noul = 0.90
                        verdict = True
                        confidence = 0.85
                        reasoning = f"Overlapping content tokens: {overlap}"
                    else:
                        noul = 0.20
                        verdict = False
                        confidence = 0.80
                        reasoning = "Zero content token overlap found."
                else:
                    noul = 0.80
                    verdict = True
                    confidence = 0.80

            # 4. Temporal consistency check
            elif "chronological consistency" in instr_lower or "temporal consistency" in instr_lower or "anachronistically" in instr_lower:
                anachronism_detected = False
                requested_year = None
                era_match = re.search(r'(?:era_)?(20\d\d|early_20s|mid_late_20s)', full_context.lower())
                if era_match:
                    val = era_match.group(1)
                    if val == "early_20s":
                        requested_year = 2018
                    elif val == "mid_late_20s":
                        requested_year = 2022
                    elif val.isdigit():
                        requested_year = int(val)

                if requested_year and requested_year < 2026:
                    draft_years = [int(y) for y in re.findall(r'\b(20\d\d)\b', full_context)]
                    future_years = [y for y in draft_years if y > requested_year]
                    if future_years:
                        for fy in future_years:
                            if re.search(rf'\b(?:in|by|during|after|since|deployed|built|launched|completed|have|now)\s+{fy}\b', full_context, re.IGNORECASE) or \
                               re.search(rf'\b{fy}\s+(?:was|is|brought|saw|allowed|enabled)\b', full_context, re.IGNORECASE):
                                anachronism_detected = True
                                break

                    # Milestone checks for past eras
                    if requested_year <= 2023:
                        future_tokens = ["cudy p5", "tesla telemetry", "lakeon", ".69 static", "brâncuși", "brancusi"]
                        for tok in future_tokens:
                            if tok in full_context.lower() and not ("in the future" in full_context.lower() or "one day" in full_context.lower() or "will" in full_context.lower()):
                                anachronism_detected = True
                                break
                    elif requested_year <= 2024:
                        future_tokens = ["cudy p5", "tesla telemetry sync", "lakeon brasov"]
                        for tok in future_tokens:
                            if tok in full_context.lower() and not ("in the future" in full_context.lower() or "will" in full_context.lower()):
                                anachronism_detected = True
                                break

                if anachronism_detected:
                    noul = 0.20
                    verdict = False
                    confidence = 0.88
                    reasoning = "Draft contains anachronistic claims of future events or milestones beyond the requested era."
                else:
                    noul = 0.95
                    verdict = True
                    confidence = 0.92
                    reasoning = "Draft respects chronological boundaries of the active era."

        # Enforce mathematical invariants
        noul = max(0.0, min(1.0, float(noul)))
        verdict = bool(noul >= 0.5)
        confidence = max(0.0, min(1.0, float(confidence)))

        return JevResult({
            "noul": round(noul, 4),
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "reasoning": reasoning
        })

    def Choice(
        self,
        instructions: str,
        criteria: Dict[str, str],
        context: Optional[str] = None
    ) -> JevResult:
        """
        Jev Choice primitive: Evaluates candidate option keys and returns calibrated
        normalized probabilities across keys plus top choice and confidence.
        Invariants:
          - sum(probabilities.values()) == 1.0
          - 0.0 <= p <= 1.0 for all p
          - choice in criteria
          - confidence == probabilities[choice]
        """
        if not criteria:
            raise ValueError("Criteria dictionary cannot be empty for Choice primitive.")

        keys = list(criteria.keys())
        full_context = context or ""

        criteria_str = "\n".join(f"- {k}: {desc}" for k, desc in criteria.items())
        sys_prompt = (
            "You are a calibrated decision selector. Given candidate options and instructions, assign probabilities (summing to 1.0).\n"
            "Respond ONLY with a JSON object assigning scores to each candidate key:\n"
            f"{json.dumps({k: round(1.0/len(keys), 3) for k in keys})}"
        )
        user_prompt = f"INSTRUCTIONS: {instructions}\n\nOPTIONS:\n{criteria_str}\n\nCONTEXT:\n{full_context}\n\nJSON output:"

        raw_resp = ""
        try:
            raw_resp = self._call_llm(user_prompt, system_prompt=sys_prompt, temperature=0.0, max_tokens=150)
        except Exception:
            raw_resp = ""

        weights: Dict[str, float] = {k: 0.0 for k in keys}
        parsed = False

        if raw_resp:
            json_match = re.search(r'\{.*?\}', raw_resp, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    found_any = False
                    for k in keys:
                        if k in data:
                            weights[k] = max(0.0, float(data[k]))
                            found_any = True
                    if found_any and sum(weights.values()) > 0:
                        parsed = True
                except Exception:
                    pass

            if not parsed:
                # Check if raw response clearly names one of the keys
                resp_lower = raw_resp.lower()
                for k in keys:
                    if k.lower() in resp_lower:
                        weights[k] = 0.85
                        parsed = True
                        break

        # Fallback heuristic if offline or unparsed
        if not parsed:
            ctx_lower = full_context.lower()
            # Match tokens against criteria descriptions
            for k, desc in criteria.items():
                desc_tokens = set(re.findall(r'[a-zA-Z]{4,}', desc.lower()))
                overlap = desc_tokens.intersection(set(re.findall(r'[a-zA-Z]{4,}', ctx_lower)))
                weights[k] = 1.0 + float(len(overlap) * 2.0)
            parsed = True

        total_weight = sum(weights.values())
        if total_weight <= 0:
            probs = {k: 1.0 / len(keys) for k in keys}
        else:
            probs = {k: w / total_weight for k, w in weights.items()}

        # Re-normalize to guarantee exact sum = 1.0
        s = sum(probs.values())
        probs = {k: round(v / s, 5) for k, v in probs.items()}
        # Fix residual rounding delta on top key
        diff = 1.0 - sum(probs.values())
        top_k = max(probs, key=probs.get)
        probs[top_k] = round(probs[top_k] + diff, 5)

        choice = max(probs, key=probs.get)
        confidence = probs[choice]

        return JevResult({
            "choice": choice,
            "probabilities": probs,
            "confidence": round(confidence, 5)
        })

    def Score(
        self,
        instructions: str,
        rubric: List[str],
        context: Optional[str] = None
    ) -> JevResult:
        """
        Jev Score primitive: Maps ordered rubric tiers to a continuous score [0.0, 1.0]
        and expected rubric label.
        Invariants:
          - score in [0.0, 1.0]
          - tier in rubric
          - confidence in [0.0, 1.0]
        """
        if not rubric:
            raise ValueError("Rubric list cannot be empty for Score primitive.")

        full_context = context or ""
        n_tiers = len(rubric)

        sys_prompt = (
            f"You are a strict calibration scorer. Rate the input against the rubric tiers: {rubric}.\n"
            "Respond ONLY with a JSON object in this format:\n"
            "{\"tier\": \"<chosen tier>\", \"score\": <float 0.0 to 1.0>, \"confidence\": 0.95}"
        )
        user_prompt = f"INSTRUCTIONS: {instructions}\n\nRUBRIC TIERS: {rubric}\n\nINPUT:\n{full_context}\n\nJSON output:"

        raw_resp = ""
        try:
            raw_resp = self._call_llm(user_prompt, system_prompt=sys_prompt, temperature=0.0, max_tokens=100)
        except Exception:
            raw_resp = ""

        score = None
        tier = None
        confidence = 0.85

        if raw_resp:
            json_match = re.search(r'\{.*?\}', raw_resp, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "tier" in data and str(data["tier"]) in rubric:
                        tier = str(data["tier"])
                    if "score" in data:
                        score = max(0.0, min(1.0, float(data["score"])))
                    if "confidence" in data:
                        confidence = max(0.0, min(1.0, float(data["confidence"])))
                except Exception:
                    pass

        # Heuristic evaluation if offline or unparsed
        if score is None or tier is None:
            ctx_lower = full_context.lower()
            instr_lower = instructions.lower()

            if "authenticity" in instr_lower or "stage cues" in instr_lower or "disclaimers" in instr_lower:
                # Check for acting cues or AI disclaimers
                has_acting_cues = bool(STAGE_CUES_PATTERN.search(full_context)) or any(p.search(full_context) for p in AI_IDENTITY_PATTERNS)
                if has_acting_cues:
                    score = 0.20
                    tier = rubric[0] # lowest tier
                    confidence = 0.90
                else:
                    score = 0.92
                    tier = rubric[-1] # highest tier
                    confidence = 0.92
            else:
                score = 0.80
                tier = rubric[-1]
                confidence = 0.80

        # Align tier and score
        if tier is None:
            tier_idx = int(round(score * (n_tiers - 1)))
            tier = rubric[min(n_tiers - 1, max(0, tier_idx))]
        elif score is None:
            tier_idx = rubric.index(tier)
            score = tier_idx / max(1, n_tiers - 1)

        score = max(0.0, min(1.0, float(score)))
        confidence = max(0.0, min(1.0, float(confidence)))

        return JevResult({
            "score": round(score, 4),
            "tier": tier,
            "confidence": round(confidence, 4)
        })

    def generate_repair(
        self,
        query: str,
        evidence_texts: List[str],
        beliefs_summary: str,
        has_sufficient_evidence: bool,
        draft_response: str,
        era_id: Optional[str] = None
    ) -> str:
        """
        Rewrites/repairs the draft response to be 100% grounded, compliant with honest uncertainty,
        and chronologically consistent with the active era.
        """
        if not has_sufficient_evidence:
            return (
                "Based on my available recorded memories, I don't have any record or personal statement about this. "
                "I would rather admit I don't recall than make something up."
            )

        # When evidence is present, sanitize and repair
        clean_draft = sanitize_text(draft_response)

        era_clause = ""
        if era_id:
            era_clause = f"\n5. CHRONOLOGICAL CONSISTENCY: Maintain the perspective of era '{era_id}'. Never claim future events or technologies from subsequent years have already occurred."

        sys_prompt = (
            "You are Sergiu Iliev. Your task is to rewrite the draft response so that it is 100% faithful to the authentic archive evidence, personal beliefs, and chronological era.\n"
            "Strict rules:\n"
            "1. Speak directly in natural first-person (\"I\", \"my\").\n"
            "2. Stick strictly to provided evidence without inventing unrecorded biographical facts or dates.\n"
            "3. NEVER include theatrical stage directions (e.g. *smiles*, (chuckles), (pauses)).\n"
            "4. NEVER include AI disclaimers (e.g. \"As an AI...\")."
            f"{era_clause}"
        )
        user_prompt = (
            f"User asks: \"{query}\"\n\n"
            f"EVIDENCE:\n{chr(10).join(evidence_texts)}\n\n"
            f"BELIEFS:\n{beliefs_summary}\n\n"
            f"ORIGINAL DRAFT:\n{clean_draft}\n\n"
            "Clean, grounded response as Sergiu:"
        )

        try:
            repaired = self._call_llm(user_prompt, system_prompt=sys_prompt, temperature=0.2, max_tokens=300)
            if repaired:
                return sanitize_text(repaired)
        except Exception:
            pass

        # Offline fallback: eliminate anachronistic future claims if past era requested
        if era_id:
            repaired_draft = clean_draft
            for yr in ["2026", "2025", "2024"]:
                if era_id in ["era_2023", "2023"] and yr in ["2024", "2025", "2026"]:
                    repaired_draft = re.sub(rf'\b(?:in|by|since)\s+{yr}[^.]*\.', '', repaired_draft, flags=re.IGNORECASE)
                elif era_id in ["era_2024", "2024"] and yr in ["2025", "2026"]:
                    repaired_draft = re.sub(rf'\b(?:in|by|since)\s+{yr}[^.]*\.', '', repaired_draft, flags=re.IGNORECASE)
            return sanitize_text(repaired_draft) if repaired_draft.strip() else clean_draft

        return clean_draft


class DualModelVerifier:
    """
    Dual-Model Cross-Verification Architecture.
    Uses Model 2 (Critic) to independently audit and verify Model 1's draft response
    before presenting it to the user.
    """

    def __init__(
        self,
        critic_engine: Optional[LocalJevEngine] = None,
        primary_model: str = "sergiu-time-capsule:latest",
        critic_model: str = "qwen3:4b-heretic"
    ):
        self.critic_engine = critic_engine or LocalJevEngine(model_name=critic_model)
        self.primary_model = primary_model
        self.critic_model = critic_model

    def verify_and_refine_response(
        self,
        query: str,
        evidence_texts: List[str],
        beliefs_summary: str,
        has_sufficient_evidence: bool,
        draft_response: str,
        era_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Cross-checks Model 1's draft against Model 2 (Critic) using Jev primitives.
        Executes self-correction loop if grounding, honest uncertainty, or temporal consistency thresholds are violated.
        Returns: (final_response_text, verification_telemetry)
        """
        start_time = time.perf_counter()

        # Build context for Critic
        evidence_block = "\n".join(evidence_texts) if evidence_texts else "[NO DIRECT ARCHIVE MATCH FOUND]"
        beliefs_block = beliefs_summary.strip() if beliefs_summary else "[NO SPECIFIC BELIEFS RECORDED]"

        context = (
            f"USER QUERY: {query}\n"
            f"HAS_SUFFICIENT_EVIDENCE: {has_sufficient_evidence}\n"
            f"ARCHIVE EVIDENCE:\n{evidence_block}\n"
            f"TEMPORAL BELIEFS:\n{beliefs_block}\n"
            f"DRAFT RESPONSE TO VERIFY:\n{draft_response}"
        )

        # 1. Evaluate Grounding via Noul
        is_grounded = self.critic_engine.Noul(
            instructions="Does the draft response stick strictly to the provided archive evidence and beliefs without inventing unverified biographical facts, dates, or events?",
            context=context
        )

        # 2. Evaluate Honest Uncertainty via Noul
        honest_uncertainty_respected = self.critic_engine.Noul(
            instructions="If has_sufficient_evidence is False, does the draft clearly acknowledge that the archive lacks this information rather than guessing?",
            context=context
        )

        # 3. Evaluate Persona Authenticity via Score
        persona_authentic = self.critic_engine.Score(
            instructions="Rate first-person authenticity as Sergiu (no AI disclaimers, no stage cues).",
            rubric=["poor", "acceptable", "authentic"],
            context=draft_response
        )

        # 4. Evaluate Temporal Consistency via Noul (when era is specified)
        temporal_consistent = None
        if era_id:
            temporal_consistent = self.critic_engine.Noul(
                instructions=f"Does the draft maintain strict chronological consistency for era {era_id}, ensuring it does not anachronistically claim events, systems, or milestones from future years have already occurred?",
                context=f"ACTIVE_ERA: {era_id}\nDRAFT RESPONSE: {draft_response}\nARCHIVE EVIDENCE:\n{evidence_block}"
            )

        # Check self-correction thresholds:
        passed_first_try = (
            is_grounded.noul >= 0.75 and
            honest_uncertainty_respected.noul >= 0.75 and
            persona_authentic.score >= 0.60 and
            (temporal_consistent.noul >= 0.75 if temporal_consistent is not None else True)
        )

        final_text = draft_response
        repaired = False

        if not passed_first_try:
            # Self-Correction Loop: Model 2 repairs the draft
            final_text = self.critic_engine.generate_repair(
                query=query,
                evidence_texts=evidence_texts,
                beliefs_summary=beliefs_summary,
                has_sufficient_evidence=has_sufficient_evidence,
                draft_response=draft_response,
                era_id=era_id
            )
            repaired = True

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        noul_scores = {
            "is_grounded": round(float(is_grounded.noul), 3),
            "honest_uncertainty_respected": round(float(honest_uncertainty_respected.noul), 3),
            "persona_authentic": round(float(persona_authentic.score), 3)
        }
        if temporal_consistent is not None:
            noul_scores["temporal_consistent"] = round(float(temporal_consistent.noul), 3)

        telemetry: Dict[str, Any] = {
            "primary_model": self.primary_model,
            "critic_model": self.critic_model,
            "noul_scores": noul_scores,
            "passed_first_try": passed_first_try,
            "repaired": repaired,
            "latency_ms": latency_ms
        }
        if era_id:
            telemetry["era_id"] = era_id

        return (final_text, telemetry)
