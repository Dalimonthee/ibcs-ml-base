"""Single-image IBCS LLM audit wrapper for the web app."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
_LLM_SCRIPT = _REPO_ROOT / "llm-test-openai.py"


@lru_cache(maxsize=1)
def _load_llm_module():
    if not _LLM_SCRIPT.is_file():
        raise RuntimeError(f"LLM audit script not found: {_LLM_SCRIPT}")
    spec = importlib.util.spec_from_file_location("llm_test_openai", _LLM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load LLM audit script: {_LLM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def llm_audit_available() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return _LLM_SCRIPT.is_file()


def audit_dashboard_image(
    image_path: Path,
    api_key: str,
    *,
    model: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Run a live OpenAI IBCS compliance audit on one dashboard image."""
    key = (api_key or "").strip()
    if not key:
        raise ValueError("OpenAI API key is required for LLM analysis.")

    mod = _load_llm_module()
    AuditConfig = mod.AuditConfig
    RateLimiter = mod.RateLimiter
    OpenAI = mod.OpenAI

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    config = AuditConfig(
        api_key=key,
        model=(model or "").strip() or mod.DEFAULT_OPENAI_MODEL,
        image_dir=image_path.parent,
        audit_runs_per_image=1,
    )
    client = OpenAI(api_key=key, timeout=config.http_timeout_sec)
    rate_limiter = RateLimiter(
        config.min_interval_sec,
        config.max_requests_per_minute,
    )

    result = mod.generate_audit(image_path, config, rate_limiter, client)
    if filename:
        result["filename"] = filename
    if "error" in result and result.get("compliant") is None:
        raise RuntimeError(result.get("error", "LLM audit failed."))
    return result
