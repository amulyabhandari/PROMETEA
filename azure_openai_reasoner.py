from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

from .albert_reasoner import build_participant_chunks, short_detail
from .cue_engine import calculate_python_cues
from .prompt_loader import load_config_prompt


DEFAULT_AUTO_CHUNK_ROWS = 1000
DEFAULT_MAX_OUTPUT_TOKENS = 12000
EVIDENCE_TABLE_LABELS = [
    "Completed tasks",
    "Decathlon entries",
    "Decathlon visual rows",
    "Decathlon Ask target item",
    "Decathlon Ask wrong item",
    "Decathlon Pay target item",
    "Decathlon Pay wrong item",
    "Decathlon validation",
    "Fnac entries",
    "Fnac visual rows",
    "Fnac Ask target item",
    "Fnac Ask wrong item",
    "Fnac Pay target item",
    "Fnac Pay wrong item",
    "Fnac validation",
    "Zara entries",
    "Zara visual rows",
    "Zara Ask target item",
    "Zara Ask wrong item",
    "Zara Pay target item",
    "Zara Pay wrong item",
    "Zara validation",
    "House confirmed tasks",
    "House confused/unclear tasks",
    "Actual beacon taps",
    "Beacon taps at target shops",
]


def run_azure_openai_raw_reasoner(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    deployment: str | None = None,
    api_version: str | None = None,
    system_prompt: str = "",
    sleep_seconds: float = 0.0,
    timeout_seconds: float = 900.0,
    chunk_rows: int = 0,
    chunk_overlap_rows: int = 25,
    max_chunks: int = 0,
) -> dict[str, Any]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    configured_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    endpoint = normalize_azure_endpoint(configured_endpoint)
    auth_mode = os.environ.get("AZURE_OPENAI_AUTH", "api_key").strip().lower()
    resolved_deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    resolved_api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    max_output_tokens = int(os.environ.get("AZURE_OPENAI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
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
            "error": "Missing Azure OpenAI environment config.",
            "detail": ", ".join(missing),
        }

    try:
        from openai import AzureOpenAI, OpenAI
    except ImportError as exc:
        return {
            "available": False,
            "error": "openai package is not installed. Use `python -m pip install -r requirements.txt` inside your virtual environment.",
            "detail": str(exc),
        }

    if is_openai_v1_endpoint:
        resolved_api_version = None
        try:
            credential = resolve_openai_v1_credential(api_key=api_key, auth_mode=auth_mode)
        except Exception as exc:
            return {
                "available": False,
                "error": "Could not configure Azure OpenAI v1 authentication.",
                "detail": str(exc),
            }
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

    participant_reports = []
    participants = raw_batch.get("participants", [])
    verified_evidence_by_participant = build_verified_evidence_by_participant(
        raw_batch=raw_batch,
        tasks_config=tasks_config,
        clues_config=clues_config,
    )
    for index, participant in enumerate(participants):
        participant_id = participant.get("participant_id")
        if chunk_rows > 0:
            result = call_azure_openai_for_participant_chunks(
                client=client,
                deployment=resolved_deployment,
                participant=participant,
                tasks_config=tasks_config,
                clues_config=clues_config,
                rows_per_chunk=chunk_rows,
                overlap_rows=chunk_overlap_rows,
                max_chunks=max_chunks,
                sleep_seconds=sleep_seconds,
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
            )
        else:
            result = call_azure_openai_for_participant(
                client=client,
                deployment=resolved_deployment,
                participant=participant,
                tasks_config=tasks_config,
                clues_config=clues_config,
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
            )
            if is_context_length_error(result):
                fallback_result = call_azure_openai_for_participant_chunks(
                    client=client,
                    deployment=resolved_deployment,
                    participant=participant,
                    tasks_config=tasks_config,
                    clues_config=clues_config,
                    rows_per_chunk=DEFAULT_AUTO_CHUNK_ROWS,
                    overlap_rows=chunk_overlap_rows,
                    max_chunks=max_chunks,
                    sleep_seconds=sleep_seconds,
                    system_prompt=system_prompt,
                    max_output_tokens=max_output_tokens,
                )
                fallback_result["auto_chunked_after_context_error"] = True
                fallback_result["full_request_error"] = {
                    "error": result.get("error"),
                    "detail": result.get("detail"),
                    "input_characters": result.get("input_characters"),
                }
                result = fallback_result
            elif report_missing_task_sections(result, tasks_config):
                fallback_result = call_azure_openai_for_participant_chunks(
                    client=client,
                    deployment=resolved_deployment,
                    participant=participant,
                    tasks_config=tasks_config,
                    clues_config=clues_config,
                    rows_per_chunk=DEFAULT_AUTO_CHUNK_ROWS,
                    overlap_rows=chunk_overlap_rows,
                    max_chunks=max_chunks,
                    sleep_seconds=sleep_seconds,
                    system_prompt=system_prompt,
                    max_output_tokens=max_output_tokens,
                )
                fallback_result["auto_chunked_after_incomplete_report"] = True
                fallback_result["full_request_usage"] = result.get("usage", {})
                result = fallback_result
            elif report_self_contradiction(result, tasks_config):
                fallback_result = call_azure_openai_for_participant_chunks(
                    client=client,
                    deployment=resolved_deployment,
                    participant=participant,
                    tasks_config=tasks_config,
                    clues_config=clues_config,
                    rows_per_chunk=DEFAULT_AUTO_CHUNK_ROWS,
                    overlap_rows=chunk_overlap_rows,
                    max_chunks=max_chunks,
                    sleep_seconds=sleep_seconds,
                    system_prompt=system_prompt,
                    max_output_tokens=max_output_tokens,
                )
                fallback_result["auto_chunked_after_self_contradiction"] = True
                fallback_result["full_request_usage"] = result.get("usage", {})
                result = fallback_result
        participant_reports.append({
            "participant_id": participant_id,
            "result": audit_azure_evidence_counts(
                audit_azure_report_if_possible(
                    client=client,
                    deployment=resolved_deployment,
                    result=result,
                    tasks_config=tasks_config,
                    clues_config=clues_config,
                    max_output_tokens=max_output_tokens,
                ),
                verified_evidence_by_participant.get(str(participant_id), {}),
            ),
        })

        if sleep_seconds > 0 and index < len(participants) - 1:
            time.sleep(sleep_seconds)

    return {
        "available": True,
        "provider": "azure_openai",
        "endpoint": endpoint,
        "configured_endpoint": configured_endpoint,
        "deployment": resolved_deployment,
        "api_version": resolved_api_version,
        "client_mode": client_mode,
        "participant_reports": participant_reports,
    }


def build_verified_evidence_by_participant(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
) -> dict[str, dict[str, int]]:
    try:
        cue_batch = calculate_python_cues(
            raw_batch=raw_batch,
            tasks_config=tasks_config,
            clues_config=clues_config,
            reference_batch=raw_batch,
            shop_locations_config=tasks_config.get("shop_locations"),
        )
    except Exception:
        return {}
    summaries = {}
    for participant in cue_batch.get("participants", []):
        participant_id = str(participant.get("participant_id", ""))
        summary = participant.get("evidence_summary") or {}
        summaries[participant_id] = {
            str(label): int(value)
            for label, value in summary.items()
            if isinstance(value, int)
        }
    return summaries


def audit_azure_evidence_counts(
    result: dict[str, Any],
    verified_evidence: dict[str, int],
) -> dict[str, Any]:
    content = str(result.get("content", ""))
    if result.get("available") is False or not content or not verified_evidence:
        return result
    reported = extract_evidence_table_counts(content)
    if not reported:
        return result
    mismatches = []
    for label, expected in verified_evidence.items():
        observed = reported.get(label)
        if observed != expected:
            mismatches.append((label, expected, observed))
    if not mismatches:
        return result

    updated = dict(result)
    updated["raw_evidence_audit_failed"] = True
    updated["raw_evidence_mismatches"] = mismatches
    updated["pre_raw_evidence_audit_content"] = content
    provider_label = str(result.get("provider") or "LLM")
    updated["content"] = raw_evidence_audit_failed_markdown(
        verified_evidence,
        mismatches,
        provider_label,
        generated_content=content,
    )
    return updated


def extract_evidence_table_counts(content: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    expected_labels = set(EVIDENCE_TABLE_LABELS)
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = cells[0]
        if label not in expected_labels:
            continue
        match = re.search(r"-?\d+", cells[1])
        if match:
            counts[label] = int(match.group(0))
    return counts


def raw_evidence_audit_failed_markdown(
    verified_evidence: dict[str, int],
    mismatches: list[tuple[str, int, int | None]],
    provider_label: str = "LLM",
    generated_content: str = "",
) -> str:
    lines = [
        "## Evidence Verification",
        "",
        f"The generated {provider_label} evidence table did not fully match the raw CSV verification.",
        "To avoid presenting unsupported interpretation, this report shows the verified evidence counts and the mismatches that need analyst review.",
        "",
        "Verified evidence from raw CSV:",
        "",
        "| Evidence | Count / value |",
        "|---|---:|",
    ]
    for label in EVIDENCE_TABLE_LABELS:
        lines.append(f"| {label} | {verified_evidence.get(label, 0)} |")
    lines.extend(["", "Mismatches:", ""])
    for label, expected, observed in mismatches:
        lines.append(f"- {label}: raw CSV count = {expected}, LLM report count = {observed}")
    lines.extend([
        "",
        "Use in comparison:",
        "Use the verified evidence counts above for the comparison table. Treat the generated narrative as requiring analyst review before it is used as an interpretation.",
    ])
    if generated_content.strip():
        lines.extend([
            "",
            "## Generated Raw LLM Reasoning",
            "",
            "The section below is the raw LLM final reasoning. It is kept for qualitative comparison, but any evidence-count mismatch listed above should be considered during analyst review.",
            "",
            generated_content.strip(),
        ])
    return "\n".join(lines)


def normalize_azure_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return endpoint
    cleaned = endpoint.rstrip("/")
    if "/openai/v1" in cleaned:
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.netloc.endswith(".services.ai.azure.com"):
        return f"{parsed.scheme}://{parsed.netloc}/openai/v1"
    return cleaned


def resolve_openai_v1_credential(*, api_key: str | None, auth_mode: str) -> Any:
    if auth_mode == "azure_identity":
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is not installed. Install requirements.txt or use AZURE_OPENAI_AUTH=api_key."
            ) from exc
        return get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )
    return api_key


def is_context_length_error(result: dict[str, Any]) -> bool:
    if result.get("available", True):
        return False
    text = " ".join(
        str(result.get(key, ""))
        for key in ["error", "detail"]
    ).lower()
    markers = [
        "context_length_exceeded",
        "maximum context length",
        "reduce the length",
        "messages resulted in",
        "token limit",
        "too many tokens",
        "input tokens",
    ]
    if any(marker in text for marker in markers):
        return True
    return "400" in text and "token" in text


def report_missing_task_sections(result: dict[str, Any], tasks_config: dict[str, Any]) -> bool:
    if result.get("available") is False:
        return False
    content = str(result.get("content", ""))
    if not content.strip():
        return True
    for task in tasks_config.get("event_based_tasks", []):
        if not task.get("enabled", True):
            continue
        label = str(task.get("example_task", "")).strip()
        if label and re.search(rf"Task:\s*{re.escape(label)}\b", content, flags=re.IGNORECASE) is None:
            return True
    return False


def report_self_contradiction(result: dict[str, Any], tasks_config: dict[str, Any]) -> bool:
    if result.get("available") is False:
        return False
    content = str(result.get("content", ""))
    if not content.strip():
        return False
    task_sections = split_task_sections(content, tasks_config)
    contradiction_markers = [
        "watching, not entry",
        "not entry",
        "no enter verb",
        "no ask verb",
        "no pay verb",
        "no place verb",
        "no validate verb",
        "no direct purchase sequence",
        "payment not explicit",
        "no matching row found",
    ]
    for section in task_sections:
        lower = section.lower()
        if "outcome: completed" not in lower:
            continue
        if any(marker in lower for marker in contradiction_markers):
            return True
    return False


def split_task_sections(content: str, tasks_config: dict[str, Any]) -> list[str]:
    labels = [
        str(task.get("example_task", "")).strip()
        for task in tasks_config.get("event_based_tasks", [])
        if task.get("enabled", True) and task.get("example_task")
    ]
    starts: list[tuple[int, str]] = []
    for label in labels:
        for match in re.finditer(rf"Task:\s*{re.escape(label)}\b", content, flags=re.IGNORECASE):
            starts.append((match.start(), label))
    starts.sort()
    sections = []
    for index, (start, _label) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(content)
        sections.append(content[start:end])
    return sections


def audit_azure_report_if_possible(
    *,
    client: Any,
    deployment: str,
    result: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    if result.get("available") is False or not result.get("content"):
        return result
    if not report_needs_audit(str(result.get("content", "")), tasks_config):
        return result
    audited = call_azure_openai_report_auditor(
        client=client,
        deployment=deployment,
        report_content=str(result.get("content", "")),
        tasks_config=tasks_config,
        clues_config=clues_config,
        max_output_tokens=max_output_tokens,
    )
    if audited.get("available") is False or not str(audited.get("content", "")).strip():
        return result
    updated = dict(result)
    updated["content"] = audited.get("content", "")
    updated["audit_pass_applied"] = True
    updated["pre_audit_usage"] = result.get("usage", {})
    updated["usage"] = audited.get("usage", {})
    return updated


def report_needs_audit(content: str, tasks_config: dict[str, Any]) -> bool:
    lower = content.lower()
    if report_missing_task_sections({"available": True, "content": content}, tasks_config):
        return True
    if report_self_contradiction({"available": True, "content": content}, tasks_config):
        return True
    markers = [
        "not strongly supported",
        "not a long delay",
        "short delay",
        "tag scan occurred before the first task attempt",
        "tag scan events occurred before",
        "watching, not entry",
        "no explicit pay",
        "no explicit place",
        "no explicit validate",
    ]
    return any(marker in lower for marker in markers)


def call_azure_openai_report_auditor(
    *,
    client: Any,
    deployment: str,
    report_content: str,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    audit_prompt = (
        load_config_prompt("azure_report_audit_prompt.md")
        + "\n\n"
        + json.dumps(
            {
                "config": {"tasks": tasks_config, "clues": clues_config},
                "report_to_audit": report_content,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": audit_prompt},
            ],
            temperature=0,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Azure OpenAI report-audit call failed.",
            "detail": str(exc),
            "input_characters": len(audit_prompt),
        }
    usage = getattr(response, "usage", None)
    return {
        "available": True,
        "content": response.choices[0].message.content or "",
        "input_characters": len(audit_prompt),
        "usage": response_usage(usage),
    }


def call_azure_openai_for_participant(
    *,
    client: Any,
    deployment: str,
    participant: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    payload = {
        "participant": participant,
        "config": {
            "tasks": tasks_config,
            "clues": clues_config,
        },
    }
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    user_prompt = (
        "Analyze this participant using the system instructions. "
        "Use the raw rows directly. Do not ask for cleaned or preprocessed events.\n\n"
        + payload_text
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Azure OpenAI raw-log reasoner call failed.",
            "detail": str(exc),
            "input_characters": len(user_prompt) + len(system_prompt),
        }

    usage = getattr(response, "usage", None)
    return {
        "available": True,
        "content": response.choices[0].message.content or "",
        "input_characters": len(user_prompt) + len(system_prompt),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        } if usage else {},
    }


def call_azure_openai_for_participant_chunks(
    *,
    client: Any,
    deployment: str,
    participant: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    rows_per_chunk: int,
    overlap_rows: int,
    max_chunks: int,
    sleep_seconds: float,
    system_prompt: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    all_chunks = build_participant_chunks(
        participant,
        rows_per_chunk=rows_per_chunk,
        overlap_rows=overlap_rows,
        max_chunks=0,
    )
    source_complete = not max_chunks or len(all_chunks) <= max_chunks
    chunks = all_chunks[:max_chunks] if max_chunks else all_chunks
    chunk_results = []

    for index, chunk in enumerate(chunks):
        result = call_azure_openai_for_chunk(
            client=client,
            deployment=deployment,
            participant_id=participant.get("participant_id", "unknown"),
            chunk=chunk,
            total_chunks=len(all_chunks),
            tasks_config=tasks_config,
            clues_config=clues_config,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
        )
        chunk_results.append({
            "chunk_number": chunk["chunk_number"],
            "environment": chunk["environment"],
            "data_kind": chunk["data_kind"],
            "file_name": chunk["file_name"],
            "data_row_start": chunk["data_row_start"],
            "data_row_end": chunk["data_row_end"],
            "result": result,
        })

        if result.get("available") is False or result.get("error"):
            return {
                "available": False,
                "error": "Azure OpenAI chunk evidence extraction stopped because a chunk call failed.",
                "detail": result.get("detail", result.get("error", "")),
                "input_characters": result.get("input_characters"),
                "chunked": True,
                "source_complete": False,
                "chunks_attempted": len(chunk_results),
                "chunks_total": len(all_chunks),
                "chunk_results": chunk_results,
            }

        if sleep_seconds > 0 and index < len(chunks) - 1:
            time.sleep(sleep_seconds)

    if not source_complete:
        return {
            "available": False,
            "error": "Final report was not produced because only part of the participant data was analyzed.",
            "detail": f"Remove --azure-openai-max-chunks to process all {len(all_chunks)} data segments.",
            "chunked": True,
            "source_complete": False,
            "chunks_total": len(all_chunks),
            "chunks_attempted": len(chunk_results),
            "chunk_results": chunk_results,
        }

    if sleep_seconds > 0 and chunks:
        time.sleep(sleep_seconds)

    synthesis = call_azure_openai_for_chunk_synthesis(
        client=client,
        deployment=deployment,
        participant_id=participant.get("participant_id", "unknown"),
        chunk_results=chunk_results,
        tasks_config=tasks_config,
        clues_config=clues_config,
        system_prompt=system_prompt,
        max_output_tokens=max_output_tokens,
    )

    return {
        "available": synthesis.get("available", False),
        "error": synthesis.get("error"),
        "detail": synthesis.get("detail"),
        "content": synthesis.get("content", ""),
        "input_characters": synthesis.get("input_characters"),
        "usage": synthesis.get("usage", {}),
        "chunked": True,
        "source_complete": source_complete,
        "chunks_total": len(all_chunks),
        "chunks_attempted": len(chunk_results),
        "chunk_results": chunk_results,
    }


def call_azure_openai_for_chunk(
    *,
    client: Any,
    deployment: str,
    participant_id: str,
    chunk: dict[str, Any],
    total_chunks: int,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    payload = {
        "participant_id": participant_id,
        "chunk": chunk,
        "chunk_instruction": load_config_prompt("azure_chunk_evidence_instruction.md"),
        "config": {
            "tasks": tasks_config,
            "clues": clues_config,
        },
    }
    user_prompt = (
        f"Extract possible evidence from chunk {chunk['chunk_number']} of {total_chunks}. "
        "This is an evidence-extraction pass only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Azure OpenAI chunk evidence extraction call failed.",
            "detail": str(exc),
            "input_characters": len(user_prompt) + len(system_prompt),
        }

    usage = getattr(response, "usage", None)
    content = response.choices[0].message.content or ""
    return {
        "available": True,
        "content": content,
        "structured_evidence": parse_chunk_evidence(content),
        "input_characters": len(user_prompt) + len(system_prompt),
        "usage": response_usage(usage),
    }


def call_azure_openai_for_chunk_synthesis(
    *,
    client: Any,
    deployment: str,
    participant_id: str,
    chunk_results: list[dict[str, Any]],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    system_prompt: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    evidence = merge_structured_chunk_evidence(
        chunk_results=chunk_results,
        tasks_config=tasks_config,
    )

    payload = {
        "participant_id": participant_id,
        "synthesis_instruction": load_config_prompt("chunk_synthesis_instruction.md"),
        "config": {
            "tasks": tasks_config,
            "clues": clues_config,
        },
        "extracted_chunk_evidence": evidence,
    }
    user_prompt = (
        "Create the final participant-level report from the extracted evidence. "
        "Do not treat any single evidence section as the full record. "
        "Return only the standard report, starting with the participant and task results.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": "Azure OpenAI chunk synthesis call failed.",
            "detail": str(exc),
            "input_characters": len(user_prompt) + len(system_prompt),
        }

    usage = getattr(response, "usage", None)
    return {
        "available": True,
        "content": response.choices[0].message.content or "",
        "input_characters": len(user_prompt) + len(system_prompt),
        "usage": response_usage(usage),
    }


def parse_chunk_evidence(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def merge_structured_chunk_evidence(
    *,
    chunk_results: list[dict[str, Any]],
    tasks_config: dict[str, Any],
) -> dict[str, Any]:
    evidence_types = [
        "shop_entry",
        "npc_or_ask",
        "payment",
        "bag_placement",
        "validation",
        "wrong_item",
        "secondary",
        "watch_phone",
        "house_recall",
        "trajectory",
        "other_relevant",
    ]
    task_labels = [
        task.get("example_task", "")
        for task in tasks_config.get("event_based_tasks", [])
        if task.get("enabled", True)
    ]
    merged = {
        label: {evidence_type: [] for evidence_type in evidence_types}
        for label in task_labels
        if label
    }
    unparsed_notes = []

    for chunk_result in chunk_results:
        result = chunk_result.get("result", {})
        structured = result.get("structured_evidence") or {}
        chunk_meta = {
            "chunk_number": chunk_result.get("chunk_number"),
            "environment": chunk_result.get("environment"),
            "data_kind": chunk_result.get("data_kind"),
            "file_name": chunk_result.get("file_name"),
            "data_row_start": chunk_result.get("data_row_start"),
            "data_row_end": chunk_result.get("data_row_end"),
        }
        tasks = structured.get("tasks") if isinstance(structured, dict) else {}
        if isinstance(tasks, dict):
            for task_label, task_evidence in tasks.items():
                canonical_label = canonical_task_label(str(task_label), task_labels)
                if not canonical_label:
                    continue
                if canonical_label not in merged:
                    merged[canonical_label] = {evidence_type: [] for evidence_type in evidence_types}
                if not isinstance(task_evidence, dict):
                    continue
                for evidence_type in evidence_types:
                    items = task_evidence.get(evidence_type, [])
                    if not isinstance(items, list):
                        continue
                    for evidence_item in items:
                        if not evidence_item:
                            continue
                        merged[canonical_label][evidence_type].append({
                            **chunk_meta,
                            "evidence": evidence_item,
                        })
        else:
            content = str(result.get("content", "")).strip()
            if content:
                unparsed_notes.append({**chunk_meta, "content": content})

    return {
        "tasks": merged,
        "unparsed_notes": unparsed_notes,
    }


def canonical_task_label(label: str, task_labels: list[str]) -> str | None:
    normalized = normalize_label(label)
    for task_label in task_labels:
        if normalize_label(task_label) == normalized:
            return task_label
    for task_label in task_labels:
        task_norm = normalize_label(task_label)
        if task_norm in normalized or normalized in task_norm:
            return task_label
    return None


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def response_usage(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def azure_openai_report_markdown(result: dict[str, Any]) -> str:
    lines = ["# Azure OpenAI Raw-Log Prospective Memory Report", ""]
    if not result.get("available"):
        lines.extend([
            "Azure OpenAI did not run.",
            "",
            f"Reason: {result.get('error', 'unknown error')}",
            "",
        ])
        detail = result.get("detail")
        if detail:
            lines.extend([f"Detail: {short_detail(str(detail))}", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend([
        f"Provider: `{result.get('provider')}`",
        f"Client mode: `{result.get('client_mode')}`",
        f"Deployment: `{result.get('deployment')}`",
        f"API version: `{result.get('api_version') or 'not used for /openai/v1 endpoint'}`",
        "",
    ])

    for item in result.get("participant_reports", []):
        participant_id = item.get("participant_id", "unknown participant")
        participant_result = item.get("result", {})
        lines.extend([f"## {participant_id}", ""])
        if participant_result.get("available") is False or participant_result.get("error"):
            lines.extend([
                "Azure OpenAI could not analyze this participant.",
                "",
                f"Reason: {participant_result.get('error', 'unknown error')}",
                "",
            ])
            detail = participant_result.get("detail")
            if detail:
                lines.extend([f"Detail: {short_detail(str(detail))}", ""])
            input_characters = participant_result.get("input_characters")
            if input_characters:
                lines.extend([f"Input characters attempted: `{input_characters}`", ""])
            continue

        content = str(participant_result.get("content", "")).strip()
        if participant_result.get("auto_chunked_after_context_error"):
            lines.extend([
                "Technical note: Azure first tried one full raw-data request, then automatically used chunked evidence extraction because the full request exceeded the model context length.",
                "",
            ])
        if participant_result.get("auto_chunked_after_incomplete_report"):
            lines.extend([
                "Technical note: Azure first tried one full raw-data request, but that answer did not include every enabled event-based task, so the system automatically reran Azure using chunked evidence extraction.",
                "",
            ])
        if participant_result.get("auto_chunked_after_self_contradiction"):
            lines.extend([
                "Technical note: Azure first tried one full raw-data request, but that answer contradicted its own cited evidence, so the system automatically reran Azure using chunked evidence extraction.",
                "",
            ])
        if participant_result.get("audit_pass_applied"):
            lines.extend([
                "Technical note: Azure applied a second audit pass to remove contradictions in its own report. This audit used the report text only, not Python clue results.",
                "",
            ])
        if content:
            lines.extend([content, ""])

        usage = participant_result.get("usage") or {}
        if usage:
            lines.extend([
                "Usage:",
                f"- prompt_tokens: {usage.get('prompt_tokens')}",
                f"- completion_tokens: {usage.get('completion_tokens')}",
                f"- total_tokens: {usage.get('total_tokens')}",
                "",
            ])

    return "\n".join(lines).rstrip() + "\n"
