"""
Mock LLM Provider for unit testing, offline verification, and deterministic persona evaluation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from providers.llm.base import LLMProvider


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider that simulates grounded responses without network or GPU calls."""

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response

    def is_healthy(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "model_name": "mock-deterministic-v1",
            "online": True
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024
    ) -> str:
        if self.canned_response:
            return self.canned_response

        # Inspect prompt to simulate grounded behavior
        sys_lower = (system_prompt or "").lower()
        prompt_lower = prompt.lower()
        if (
            "no direct match" in prompt_lower or
            "no sufficient recorded memory" in sys_lower or
            "insufficient" in sys_lower or
            "honest uncertainty" in prompt_lower
        ):
            return "Based on my available recorded memories, I don't have any record or personal statement about this. I would rather admit I don't recall than make something up."

        if "[ERA:" in prompt or "worldview" in prompt.lower():
            return f"Thinking back to that period, my perspective was shaped by what I valued at the time. Grounded in my notes: {prompt[:100]}..."

        return f"Reflecting on what is recorded: {prompt[:120]}... That experience stays with me."
