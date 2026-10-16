from __future__ import annotations

import json
import os
from typing import Any

from .albert_reasoner import short_detail
from .azure_openai_reasoner import normalize_azure_endpoint, resolve_openai_v1_credential


def run_albert_python_cue_reasoner(
    *,
    cue_batch: dict[str, Any],
    model: str,
    system_prompt: str,
    timeout_seconds: float = 600.0
) -> dict[str, Any]:
    api_key = os.environ.get("ALBERT_API_KEY")
    if not api_key:
        return {
            "available": False,
            "error": "ALBERT_API_KEY is not set."
        }

    try:
        from openai import OpenAI
    except ImportError as exc:
        return {
            "available": False,
            "error": "openai package is not installed. Use `python -m pip install -r requirements.txt` inside your virtual environment.",
            "detail": str(exc)
        }

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1"),
        timeout=timeout_seconds,
    )
    user_prompt = (
        "Reason over these Python-calculated clue results and write the report.\n\n"
        + json.dumps(cue_batch, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Albert Python-clue reasoner call failed.",
            "detail": str(exc),
        }

    return {
        "available": True,
        "provider": "albert",
        "model": model,
        "content": response.choices[0].message.content or "",
    }


def run_azure_openai_python_cue_reasoner(
    *,
    cue_batch: dict[str, Any],
    deployment: str | None = None,
    api_version: str | None = None,
    system_prompt: str,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    configured_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    endpoint = normalize_azure_endpoint(configured_endpoint)
    auth_mode = os.environ.get("AZURE_OPENAI_AUTH", "api_key").strip().lower()
    resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    resolved_api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    is_openai_v1_endpoint = bool(endpoint and "/openai/v1" in endpoint.rstrip("/"))

    missing = [
        name for name, value in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", resolved_deployment),
        ]
        if not value
    ]
    if auth_mode != "azure_identity" and not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if missing:
        return {
            "available": False,
            "provider": "azure_openai",
            "error": "Missing Azure OpenAI environment config.",
            "detail": ", ".join(missing),
        }

    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        return {
            "available": False,
            "provider": "azure_openai",
            "error": "openai package is not installed. Use `python -m pip install -r requirements.txt` inside your virtual environment.",
            "detail": str(exc),
        }

    try:
        if is_openai_v1_endpoint:
            resolved_api_version = None
            credential = resolve_openai_v1_credential(api_key=api_key, auth_mode=auth_mode)
            client = OpenAI(
                base_url=endpoint.rstrip("/"),
                api_key=credential,
                timeout=timeout_seconds,
            )
            client_mode = "openai_v1"
        else:
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=resolved_api_version,
                timeout=timeout_seconds,
            )
            client_mode = "azure_openai_legacy"
    except Exception as exc:
        return {
            "available": False,
            "provider": "azure_openai",
            "error": "Could not configure Azure OpenAI Python-clue reasoner.",
            "detail": str(exc),
        }

    user_prompt = (
        "Reason over these Python-calculated clue results and write the report.\n\n"
        + json.dumps(cue_batch, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=resolved_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    except Exception as exc:
        return {
            "available": False,
            "provider": "azure_openai",
            "deployment": resolved_deployment,
            "endpoint": endpoint,
            "configured_endpoint": configured_endpoint,
            "client_mode": client_mode,
            "api_version": resolved_api_version,
            "error": "Azure OpenAI Python-clue reasoner call failed.",
            "detail": str(exc),
            "input_characters": len(user_prompt),
        }

    return {
        "available": True,
        "provider": "azure_openai",
        "deployment": resolved_deployment,
        "endpoint": endpoint,
        "configured_endpoint": configured_endpoint,
        "client_mode": client_mode,
        "api_version": resolved_api_version,
        "content": response.choices[0].message.content or "",
        "usage": response.usage.model_dump() if response.usage else {},
        "input_characters": len(user_prompt),
    }


def python_cue_report_markdown(result: dict[str, Any]) -> str:
    provider = result.get("provider", "albert")
    if provider == "azure_openai":
        lines = ["# Azure OpenAI Reasoning Over Python-Calculated Clues", ""]
        lines.extend([
            f"Provider: `{provider}`",
            f"Client mode: `{result.get('client_mode', 'unknown')}`",
            f"Deployment: `{result.get('deployment', 'unknown')}`",
            f"API version: `{result.get('api_version') or 'not used for /openai/v1 endpoint'}`",
            "",
        ])
    else:
        lines = ["# Albert Reasoning Over Python-Calculated Clues", ""]
    if not result.get("available"):
        lines.extend([
            f"{'Azure OpenAI' if provider == 'azure_openai' else 'Albert'} did not run over Python-calculated clues.",
            "",
            f"Reason: {result.get('error', 'unknown error')}",
            "",
        ])
        detail = result.get("detail")
        if detail:
            lines.extend([f"Detail: {short_detail(str(detail))}", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend([str(result.get("content", "")).strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def combined_python_evidence_reasoning_markdown(
    *,
    cue_evidence_markdown: str,
    reasoner_result: dict[str, Any],
) -> str:
    provider = reasoner_result.get("provider", "azure_openai")
    provider_label = "Azure OpenAI" if provider == "azure_openai" else "Albert"
    lines = [
        "# Approach 1: Python Evidence With LLM Reasoning",
        "",
        "Python calculated the structured clue evidence from the raw logs and trajectory data. "
        f"{provider_label} then used that evidence to write the final reasoning.",
        "",
        "## Python-Calculated Evidence",
        "",
        strip_top_heading(cue_evidence_markdown).strip(),
        "",
        f"## {provider_label} Final Reasoning",
        "",
        python_cue_report_markdown(reasoner_result).strip(),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def strip_top_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)
