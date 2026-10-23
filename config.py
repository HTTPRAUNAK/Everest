"""All env-driven switches in one place."""
import os


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


# LLM
MOCK_LLM = _flag("MOCK_LLM")                       # test the whole pipeline without a model
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "90"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))   # extra attempts on bad JSON / errors

# Cloud fallback (must be disclosed in README). OpenAI-compatible endpoint.
CLOUD_FALLBACK = _flag("CLOUD_FALLBACK")
CLOUD_URL = os.getenv("CLOUD_URL", "")
CLOUD_KEY = os.getenv("CLOUD_KEY", "")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "")

# Retrieval: cosine similarity below this = treat chunk as irrelevant (drives not_found). Tune on eval set.
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))
