"""
Abstract Base Class for Local Language Model Providers.
Enforces modular provider swappability (Ollama, MLX, vLLM, Mock).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract interface for local language model inference."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024
    ) -> str:
        """Generates a text completion given user prompt and system constraints."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Returns True if the underlying model engine is running and responsive."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active model (name, quantization, parameter size)."""
        pass
