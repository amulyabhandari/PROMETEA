from __future__ import annotations

import json
import os
import time
from typing import Any

from .albert_reasoner import (
    build_participant_chunks,
    short_detail,
)
from .prompt_loader import load_config_prompt


def run_local_raw_reasoner(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    model: str,
    base_url: str,
    sleep_seconds: float = 0.0,
    chunk_rows: int = 300,
    chunk_overlap_rows: int = 25,
    max_chunks: int = 0,
    timeout_seconds: float = 600.0,
    system_prompt: str = "",
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        return {
            "available": False,
            "error": "openai package is not installed. Use `python -m pip install -r requirements.txt` inside your virtual environment.",
            "detail": str(exc)
        }

    client = OpenAI(
        api_key=os.environ.get("LOCAL_LLM_API_KEY", "ollama"),
        base_url=base_url,
        timeout=timeout_seconds
    )

    participant_reports = []
    participants = raw_batch.get("participants", [])
    for index, participant in enumerate(participants):
        if chunk_rows > 0:
            result = call_local_for_participant_chunks(
                client=client,
                model=model,
                participant=participant,
                tasks_config=tasks_config,
                clues_config=clues_config,
                rows_per_chunk=chunk_rows,
                overlap_rows=chunk_overlap_rows,
                max_chunks=max_chunks,
                sleep_seconds=sleep_seconds,
                system_prompt=system_prompt,
            )
        else:
            result = call_local_for_participant(
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

        if sleep_seconds > 0 and index < len(participants) - 1:
            time.sleep(sleep_seconds)

    return {
        "available": True,
        "provider": "local",
        "model": model,
        "base_url": base_url,
        "participant_reports": participant_reports
    }


def call_local_for_participant(
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
            "error": "Local raw-log reasoner call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def call_local_for_participant_chunks(
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
    system_prompt: str,
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
        result = call_local_for_chunk(
            client=client,
            model=model,
            participant_id=participant.get("participant_id", "unknown"),
            chunk=chunk,
            total_chunks=len(all_chunks),
            tasks_config=tasks_config,
            clues_config=clues_config,
            system_prompt=system_prompt,
        )
        chunk_results.append({
            "environment": chunk["environment"],
            "data_kind": chunk["data_kind"],
            "file_name": chunk["file_name"],
            "data_row_start": chunk["data_row_start"],
            "data_row_end": chunk["data_row_end"],
            "result": result
        })

        if result.get("available") is False or result.get("error"):
            return {
                "available": False,
                "error": "Local chunk evidence extraction stopped because a chunk call failed.",
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
            "detail": f"Remove --local-max-chunks to process all {len(all_chunks)} data segments.",
            "chunked": True,
            "source_complete": False,
            "chunks_total": len(all_chunks),
            "chunks_attempted": len(chunk_results),
            "chunk_results": chunk_results
        }

    synthesis = call_local_for_chunk_synthesis(
        client=client,
        model=model,
        participant_id=participant.get("participant_id", "unknown"),
        chunk_results=chunk_results,
        tasks_config=tasks_config,
        clues_config=clues_config,
        system_prompt=system_prompt,
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


def call_local_for_chunk(
    *,
    client: Any,
    model: str,
    participant_id: str,
    chunk: dict[str, Any],
    total_chunks: int,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
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
            "error": "Local chunk evidence extraction call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def call_local_for_chunk_synthesis(
    *,
    client: Any,
    model: str,
    participant_id: str,
    chunk_results: list[dict[str, Any]],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
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
            "error": "Local chunk synthesis call failed.",
            "detail": str(exc)
        }

    return {
        "available": True,
        "content": response.choices[0].message.content or ""
    }


def local_report_markdown(local_result: dict[str, Any]) -> str:
    lines = [
        "# Local Prospective Memory Report",
        ""
    ]

    if not local_result.get("available"):
        lines.extend([
            "Local model did not run.",
            "",
            f"Reason: {local_result.get('error', 'unknown error')}",
            ""
        ])
        detail = local_result.get("detail")
        if detail:
            lines.extend([f"Detail: {short_detail(str(detail))}", ""])
        return "\n".join(lines)

    for item in local_result.get("participant_reports", []):
        participant_id = item.get("participant_id", "unknown participant")
        result = item.get("result", {})
        lines.extend([f"## {participant_id}", ""])
        if result.get("available") is False or result.get("error"):
            lines.extend([
                "Local model could not analyze this participant.",
                "",
                f"Reason: {result.get('error', 'unknown error')}",
                "",
                f"Detail: {short_detail(str(result.get('detail', '')))}",
                ""
            ])
            continue
        lines.extend([str(result.get("content", "")).strip(), ""])

    return "\n".join(lines).rstrip() + "\n"
