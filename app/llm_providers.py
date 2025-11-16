from __future__ import annotations

"""Central LLM provider configuration for the Syllabye backend.

This module defines a single helper, `get_llm`, which returns the active
LangChain chat model instance. You can easily switch providers (OpenAI, Groq,
etc.) by editing the configuration below instead of touching the rest of the
codebase.

Best practices followed here:
- No API keys are hardcoded; they come from environment variables expected by
  the respective LangChain integrations (e.g. OPENAI_API_KEY, GROQ_API_KEY).
- Providers are configured but only the *active* one is instantiated.
- Optional dependencies (like Groq) are imported defensively.
"""

from typing import Literal

from langchain_openai import ChatOpenAI

try:  # Optional: only needed if you actually use Groq
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency
    ChatGroq = None  # type: ignore


ProviderName = Literal["openai", "groq"]


# Single place to configure which provider/model is active for the agents.
# Flip "active_provider" or adjust the model names as needed.
active_provider: ProviderName = "openai"

provider_models = {
    "openai": {
        "default_model": "gpt-4.1-mini",
    },
    "groq": {
        # You already set this in agents.py; keep it here as the default.
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
}


def get_llm(temperature: float = 0.0):
    """Return the currently active LangChain chat model.

    - Uses `active_provider` and `provider_models` above.
    - Reads API keys from environment variables expected by each integration
      (e.g. OPENAI_API_KEY, GROQ_API_KEY); do not hardcode keys in code.
    - Only instantiates one provider per call (no serving all APIs at once).
    """

    if active_provider == "openai":
        cfg = provider_models["openai"]
        return ChatOpenAI(model=cfg["default_model"], temperature=temperature)

    if active_provider == "groq":
        if ChatGroq is None:
            raise RuntimeError(
                "Groq provider is active but langchain_groq is not installed. "
                "Install it (`pip install langchain-groq`) and set GROQ_API_KEY."
            )
        cfg = provider_models["groq"]
        return ChatGroq(model=cfg["default_model"], temperature=temperature)

    raise RuntimeError(f"Unsupported active_provider: {active_provider}")
