from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .albert_reasoner import short_detail


def run_gemini_raw_reasoner(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    model: str,
    system_prompt: str,
    sleep_seconds: float = 0.0,
    timeout_seconds: float = 600.0
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "available": False,
            "error": "GEMINI_API_KEY is not set."
        }

    participant_reports = []
    participants = raw_batch.get("participants", [])
    for index, participant in enumerate(participants):
        result = call_gemini_for_participant(
            api_key=api_key,
            model=model,
            participant=participant,
            tasks_config=tasks_config,
            clues_config=clues_config,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds
        )
        participant_reports.append({
            "participant_id": participant.get("participant_id"),
            "result": result
        })

        if sleep_seconds > 0 and index < len(participants) - 1:
            time.sleep(sleep_seconds)

    return {
        "available": True,
        "provider": "gemini",
        "model": model,
        "participant_reports": participant_reports
    }


def call_gemini_for_participant(
    *,
    api_key: str,
    model: str,
    participant: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
    timeout_seconds: float
) -> dict[str, Any]:
    payload = {
        "participant": participant,
        "config": {
            "tasks": tasks_config,
            "clues": clues_config
        }
    }
    user_prompt = (
        "Analyze this participant using the system instructions. "
        "Use the raw rows directly. Do not ask for cleaned or preprocessed events.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    request_payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "available": False,
            "error": "Gemini raw-log reasoner call failed.",
            "detail": f"HTTP {exc.code}: {detail}"
        }
    except Exception as exc:
        return {
            "available": False,
            "error": "Gemini raw-log reasoner call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": extract_gemini_text(response_data),
        "usage": response_data.get("usageMetadata", {})
    }


def extract_gemini_text(response_data: dict[str, Any]) -> str:
    texts = []
    for candidate in response_data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def gemini_report_markdown(gemini_result: dict[str, Any]) -> str:
    lines = [
        "# Gemini Prospective Memory Report",
        ""
    ]

    if not gemini_result.get("available"):
        lines.extend([
            "Gemini did not run.",
            "",
            f"Reason: {gemini_result.get('error', 'unknown error')}",
            ""
        ])
        detail = gemini_result.get("detail")
        if detail:
            lines.extend([f"Detail: {short_detail(str(detail))}", ""])
        return "\n".join(lines)

    for item in gemini_result.get("participant_reports", []):
        participant_id = item.get("participant_id", "unknown participant")
        result = item.get("result", {})
        lines.extend([f"## {participant_id}", ""])
        if result.get("available") is False or result.get("error"):
            lines.extend([
                "Gemini could not analyze this participant.",
                "",
                f"Reason: {result.get('error', 'unknown error')}",
                "",
                f"Detail: {short_detail(str(result.get('detail', '')))}",
                ""
            ])
            continue
        content = str(result.get("content", "")).strip()
        if content:
            lines.extend([content, ""])
        usage = result.get("usage") or {}
        if usage:
            lines.extend([
                "Usage:",
                f"- promptTokenCount: {usage.get('promptTokenCount', 'unknown')}",
                f"- candidatesTokenCount: {usage.get('candidatesTokenCount', 'unknown')}",
                f"- totalTokenCount: {usage.get('totalTokenCount', 'unknown')}",
                ""
            ])

    return "\n".join(lines).rstrip() + "\n"
