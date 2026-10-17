"""
Ollama Local LLM Provider.
Integrates with locally running Ollama engine (e.g. llama3.2:3b, gemma2:2b) via loopback HTTP API.
Zero external cloud communication.
"""

from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from core.config import LLMConfig
from providers.llm.base import LLMProvider


import re

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


def clean_acting_cues(text: str) -> str:
    """Removes parenthetical/asterisk acting cues and AI self-referrals."""
    cleaned = STAGE_CUES_PATTERN.sub(' ', text)
    cleaned = re.sub(r'^\s*[\(\*][^\)\*]+[\)\*]\s*', '', cleaned)
    for p in AI_IDENTITY_PATTERNS:
        cleaned = p.sub('', cleaned)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    return cleaned


class OllamaLLMProvider(LLMProvider):
    """Local inference provider communicating with Ollama over loopback 127.0.0.1."""

    def __init__(self, config: LLMConfig):
        self.endpoint = config.endpoint.rstrip("/")
        self.model_name = config.model_name
        self.fallback_model = config.fallback_model
        self.timeout = config.timeout_seconds
        self.temperature = config.temperature

    def is_healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        try:
            payload = json.dumps({"name": self.model_name}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.endpoint}/api/show",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "provider": "ollama",
                    "model_name": self.model_name,
                    "details": data.get("details", {}),
                    "parameters": data.get("parameters", ""),
                    "online": True
                }
        except Exception as e:
            return {
                "provider": "ollama",
                "model_name": self.model_name,
                "online": False,
                "error": str(e)
            }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024
    ) -> str:
        temp = temperature if temperature is not None else self.temperature

        # Attempt with primary model, then fallback
        models_to_try = [self.model_name]
        if self.fallback_model and self.fallback_model != self.model_name:
            models_to_try.append(self.fallback_model)

        stop_tokens = [
            "<|eot_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "(smirks)",
            "(chuckles)",
            "(pauses",
            "(sighs)",
            "(laughs)",
            "*smiles*",
            "*chuckles*",
            "*sighs*",
            "*pauses*"
        ]

        last_error = None
        for model in models_to_try:
            try:
                body: Dict[str, Any] = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tokens,
                        "stop": stop_tokens
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
                    res_json = json.loads(resp.read().decode("utf-8"))
                    response_text = res_json.get("response", "").strip()
                    if response_text:
                        return clean_acting_cues(response_text)
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"Ollama inference failed across models {models_to_try}: {last_error}")

