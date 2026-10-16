from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

from .prompt_loader import load_config_prompt


def run_albert_raw_reasoner(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    model: str,
    sleep_seconds: float = 0.0,
    chunk_rows: int = 0,
    chunk_overlap_rows: int = 0,
    max_chunks: int = 0,
    retry_count: int = 0,
    retry_sleep_seconds: float = 0.0,
    system_prompt: str = "",
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

    base_url = os.environ.get("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    participant_reports = []
    participants = raw_batch.get("participants", [])
    for index, participant in enumerate(participants):
        if chunk_rows > 0:
            result = call_albert_for_participant_chunks(
                client=client,
                model=model,
                participant=participant,
                tasks_config=tasks_config,
                clues_config=clues_config,
                rows_per_chunk=chunk_rows,
                overlap_rows=chunk_overlap_rows,
                max_chunks=max_chunks,
                sleep_seconds=sleep_seconds,
                retry_count=retry_count,
                retry_sleep_seconds=retry_sleep_seconds,
                system_prompt=system_prompt
            )
        else:
            result = call_albert_for_participant(
                client=client,
                model=model,
                participant=participant,
                tasks_config=tasks_config,
                clues_config=clues_config,
                system_prompt=system_prompt,
            )
        participant_reports.append({
            "participant_id": participant.get("participant_id"),
            "result": result
        })

        if is_quota_or_rate_error(result):
            return {
                "available": True,
                "provider": "albert",
                "model": model,
                "base_url": base_url,
                "stopped_early": "quota_or_rate_error",
                "participant_reports": participant_reports
            }

        if sleep_seconds > 0 and index < len(participants) - 1:
            time.sleep(sleep_seconds)

    return {
        "available": True,
        "provider": "albert",
        "model": model,
        "base_url": base_url,
        "participant_reports": participant_reports
    }


def call_albert_for_participant(
    *,
    client: Any,
    model: str,
    participant: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Albert raw-log reasoner call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def call_albert_for_participant_chunks(
    *,
    client: Any,
    model: str,
    participant: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    rows_per_chunk: int,
    overlap_rows: int,
    max_chunks: int,
    sleep_seconds: float,
    retry_count: int,
    retry_sleep_seconds: float,
    system_prompt: str
) -> dict[str, Any]:
    all_chunks = build_participant_chunks(
        participant,
        rows_per_chunk=rows_per_chunk,
        overlap_rows=overlap_rows,
        max_chunks=0
    )
    source_complete = not max_chunks or len(all_chunks) <= max_chunks
    chunks = all_chunks[:max_chunks] if max_chunks else all_chunks
    chunk_results = []

    for index, chunk in enumerate(chunks):
        result = call_with_quota_retries(
            lambda: call_albert_for_chunk(
                client=client,
                model=model,
                participant_id=participant.get("participant_id", "unknown"),
                chunk=chunk,
                total_chunks=len(all_chunks),
                tasks_config=tasks_config,
                clues_config=clues_config,
                system_prompt=system_prompt
            ),
            retry_count=retry_count,
            retry_sleep_seconds=retry_sleep_seconds
        )
        chunk_results.append({
            "chunk_number": chunk["chunk_number"],
            "environment": chunk["environment"],
            "data_kind": chunk["data_kind"],
            "file_name": chunk["file_name"],
            "data_row_start": chunk["data_row_start"],
            "data_row_end": chunk["data_row_end"],
            "result": result
        })

        if is_quota_or_rate_error(result):
            return {
                "available": False,
                "error": "Albert chunk evidence extraction stopped because of quota or rate limit.",
                "detail": result.get("detail", result.get("error", "")),
                "chunked": True,
                "source_complete": False,
                "chunks_attempted": len(chunk_results),
                "chunks_total": len(all_chunks),
                "chunk_results": chunk_results
            }

        if sleep_seconds > 0 and index < len(chunks) - 1:
            time.sleep(sleep_seconds)

    if not source_complete:
        return {
            "available": False,
            "error": "Final report was not produced because only part of the participant data was analyzed.",
            "detail": f"Remove --albert-max-chunks to process all {len(all_chunks)} data segments.",
            "chunked": True,
            "source_complete": False,
            "chunks_total": len(all_chunks),
            "chunks_attempted": len(chunk_results),
            "chunk_results": chunk_results
        }

    if sleep_seconds > 0 and chunks:
        time.sleep(sleep_seconds)

    synthesis = call_with_quota_retries(
        lambda: call_albert_for_chunk_synthesis(
            client=client,
            model=model,
            participant_id=participant.get("participant_id", "unknown"),
            chunk_results=chunk_results,
            tasks_config=tasks_config,
            clues_config=clues_config,
            system_prompt=system_prompt
        ),
        retry_count=retry_count,
        retry_sleep_seconds=retry_sleep_seconds
    )

    return {
        "available": synthesis.get("available", False),
        "error": synthesis.get("error"),
        "detail": synthesis.get("detail"),
        "content": synthesis.get("content", ""),
        "chunked": True,
        "source_complete": source_complete,
        "chunks_total": len(all_chunks),
        "chunks_attempted": len(chunk_results),
        "chunk_results": chunk_results
    }


def call_with_quota_retries(
    call: Callable[[], dict[str, Any]],
    *,
    retry_count: int,
    retry_sleep_seconds: float
) -> dict[str, Any]:
    result = call()
    attempts = 0
    while is_quota_or_rate_error(result) and attempts < retry_count:
        attempts += 1
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
        result = call()

    if attempts:
        result = dict(result)
        result["retry_attempts"] = attempts
    return result


def call_albert_for_chunk(
    *,
    client: Any,
    model: str,
    participant_id: str,
    chunk: dict[str, Any],
    total_chunks: int,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str
) -> dict[str, Any]:
    payload = {
        "participant_id": participant_id,
        "chunk": chunk,
        "chunk_instruction": load_config_prompt("chunk_evidence_instruction.md"),
        "config": {
            "tasks": tasks_config,
            "clues": clues_config
        }
    }
    user_prompt = (
        f"Extract possible evidence from chunk {chunk['chunk_number']} of {total_chunks}. "
        "This is an evidence-extraction pass only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Albert chunk evidence extraction call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def call_albert_for_chunk_synthesis(
    *,
    client: Any,
    model: str,
    participant_id: str,
    chunk_results: list[dict[str, Any]],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str
) -> dict[str, Any]:
    evidence = []
    for item in chunk_results:
        result = item.get("result", {})
        evidence.append({
            "environment": item.get("environment"),
            "data_kind": item.get("data_kind"),
            "file_name": item.get("file_name"),
            "data_row_start": item.get("data_row_start"),
            "data_row_end": item.get("data_row_end"),
            "evidence_notes": result.get("content", ""),
            "error": result.get("error"),
            "detail": result.get("detail")
        })

    payload = {
        "participant_id": participant_id,
        "synthesis_instruction": load_config_prompt("chunk_synthesis_instruction.md"),
        "config": {
            "tasks": tasks_config,
            "clues": clues_config
        },
        "extracted_chunk_evidence": evidence
    }
    user_prompt = (
        "Create the final participant-level report from the extracted chunk evidence. "
        "Do not treat any single chunk as the full record. "
        "Return only the standard report, starting with the participant and task results.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Albert chunk synthesis call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def build_participant_chunks(
    participant: dict[str, Any],
    *,
    rows_per_chunk: int,
    overlap_rows: int,
    max_chunks: int
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for env_name, environment in participant.get("environments", {}).items():
        for data_kind in ["events", "trajectory"]:
            for file_payload in environment.get(data_kind, []):
                raw_text = file_payload.get("raw_csv_with_row_numbers", "")
                lines = raw_text.splitlines()
                if not lines:
                    continue

                header = lines[0]
                rows = lines[1:]
                for start in range(0, len(rows), rows_per_chunk):
                    end = min(len(rows), start + rows_per_chunk)
                    overlap_start = max(0, start - max(0, overlap_rows))
                    chunk_lines = [header, *rows[overlap_start:end]]
                    chunks.append({
                        "chunk_number": chunk_number,
                        "participant_id": participant.get("participant_id"),
                        "environment": env_name,
                        "data_kind": data_kind,
                        "file_name": file_payload.get("file_name"),
                        "file_path": file_payload.get("file_path"),
                        "columns": file_payload.get("columns", []),
                        "data_row_start": start + 1,
                        "data_row_end": end,
                        "overlap_data_row_start": overlap_start + 1,
                        "included_data_row_count": end - overlap_start,
                        "raw_csv_with_row_numbers": "\n".join(chunk_lines)
                    })
                    chunk_number += 1

                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks

    return chunks


def albert_report_markdown(albert_result: dict[str, Any]) -> str:
    lines = [
        "# Prospective Memory Report",
        ""
    ]

    if not albert_result.get("available"):
        lines.extend([
            "Albert did not run.",
            "",
            f"Reason: {albert_result.get('error', 'unknown error')}",
            ""
        ])
        detail = albert_result.get("detail")
        if detail:
            lines.extend([f"Detail: {short_detail(str(detail))}", ""])
        return "\n".join(lines)

    stopped = albert_result.get("stopped_early")
    if stopped:
        if stopped == "quota_or_rate_error":
            lines.extend(["Stopped early because Albert returned a quota or rate-limit error.", ""])
        else:
            lines.extend([f"Stopped early: {stopped}", ""])

    for item in albert_result.get("participant_reports", []):
        participant_id = item.get("participant_id", "unknown participant")
        result = item.get("result", {})
        lines.extend([f"## {participant_id}", ""])
        if result.get("available") is False or result.get("error"):
            if is_daily_quota_error(result):
                lines.extend([
                    "Albert could not analyze this participant because the daily input-token quota is exhausted.",
                    "",
                    f"Detail: {short_detail(str(result.get('detail', '')))}",
                    ""
                ])
                continue
            if result.get("chunked") and result.get("source_complete") is False:
                lines.extend([
                    "Final report was not produced because the run did not process the complete participant data.",
                    "",
                    f"Detail: {short_detail(str(result.get('detail', '')))}",
                    ""
                ])
                continue
            lines.extend([
                "Albert could not analyze this participant.",
                "",
                f"Reason: {result.get('error', 'unknown error')}",
                "",
                f"Detail: {short_detail(str(result.get('detail', '')))}",
                ""
            ])
            continue
        lines.extend([str(result.get("content", "")).strip(), ""])

    return "\n".join(lines).rstrip() + "\n"


def is_quota_or_rate_error(result: dict[str, Any]) -> bool:
    text = f"{result.get('error', '')} {result.get('detail', '')}".casefold()
    return any(marker in text for marker in ["429", "quota", "rate", "too many requests", "token"])


def is_daily_quota_error(result: dict[str, Any]) -> bool:
    text = f"{result.get('error', '')} {result.get('detail', '')}".casefold()
    return "per day exceeded" in text or "daily" in text


def short_detail(detail: str, limit: int = 700) -> str:
    detail = re.sub(r"\s+", " ", detail).strip()
    if len(detail) <= limit:
        return detail
    return detail[:limit].rstrip() + "..."
