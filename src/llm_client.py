from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

class EchoLLM(LLMClient):
    """Fallback dummy model: echos prompt for testing."""
    def generate(self, prompt: str, **kwargs) -> str:
        return f"[ECHO MODEL] {prompt}"

# Example OpenAI client (pseudo; implement real call when you have a key):
class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
    def generate(self, prompt: str, **kwargs) -> str:
        # Use openai SDK here. This is a placeholder to keep starter offline.
        return f"[OPENAI:{self.model}] (stubbed) -> {prompt[:120]}..."
