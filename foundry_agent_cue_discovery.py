from __future__ import annotations

import json
import csv
import math
import os
import time
from pathlib import Path
from typing import Any
from io import StringIO
from urllib.parse import urlencode


def run_foundry_agent_cue_discovery(
    *,
    domain_text: str,
    discovery_prompt: str,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any] | None = None,
    raw_data_context: dict[str, Any] | None = None,
    participant_context: str = "",
    workflow_stage: str = "full",
    validated_cues_context: str = "",
    project_endpoint: str | None = None,
    model_deployment_name: str | None = None,
    auth_mode: str | None = None,
    agent_name: str = "pm-clue-discovery-agent",
    cleanup_agent: bool = True,
) -> dict[str, Any]:
    resolved_endpoint = (
        project_endpoint
        or os.environ.get("PROJECT_ENDPOINT")
        or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        or project_endpoint_from_azure_openai_env()
    )
    resolved_model = (
        model_deployment_name
        or os.environ.get("MODEL_DEPLOYMENT_NAME")
        or os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    )
    api_key = (
        os.environ.get("PROJECT_API_KEY")
        or os.environ.get("AZURE_AI_API_KEY")
        or os.environ.get("AZURE_AI_FOUNDRY_API_KEY")
    )
    resolved_auth_mode = (auth_mode or os.environ.get("FOUNDRY_AUTH_MODE") or "auto").strip().lower()

    missing = [
        name for name, value in [
            ("PROJECT_ENDPOINT or AZURE_AI_PROJECT_ENDPOINT", resolved_endpoint),
            ("MODEL_DEPLOYMENT_NAME or AZURE_AI_MODEL_DEPLOYMENT_NAME", resolved_model),
        ]
        if not value
    ]
    if missing:
        return {
            "available": False,
            "error": "Missing Azure AI Foundry Agent Service environment config.",
            "detail": ", ".join(missing),
        }

    if resolved_auth_mode not in {"auto", "api_key", "azure_identity"}:
        return {
            "available": False,
            "error": "Invalid Foundry auth mode.",
            "detail": "Use auto, api_key, or azure_identity.",
        }

    if api_key and resolved_auth_mode in {"auto", "api_key"}:
        return run_foundry_agent_cue_discovery_rest(
            domain_text=domain_text,
            discovery_prompt=discovery_prompt,
            tasks_config=tasks_config,
            clues_config=clues_config,
            raw_data_context=raw_data_context,
            participant_context=participant_context,
            workflow_stage=workflow_stage,
            validated_cues_context=validated_cues_context,
            project_endpoint=resolved_endpoint,
            model_deployment_name=resolved_model,
            api_key=api_key,
            agent_name=agent_name,
            cleanup_agent=cleanup_agent,
        )

    try:
        from azure.ai.agents import AgentsClient
        from azure.ai.agents.models import ListSortOrder, MessageRole
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        return {
            "available": False,
            "error": "Azure AI Foundry Agent SDK packages are not installed. Run `python -m pip install -r requirements.txt`.",
            "detail": str(exc),
        }

    instructions = "\n\n".join([
        discovery_prompt,
        "You are running inside Azure AI Foundry Agent Service. Produce the final Markdown report as the agent response.",
    ])
    payload = {
        "domain_and_scenario": domain_text,
        "task_config": tasks_config,
        "analyst_clues": build_analyst_cues(clues_config or {}),
        "analyst_clue_config": clues_config or {},
        "task_facts": build_task_facts(tasks_config),
        "workflow_stage": workflow_stage,
        "validated_clues_from_analyst": validated_cues_context,
        "raw_vr_logs": raw_data_context,
        "optional_participant_context": participant_context,
    }
    user_message = (
        "Run the analyst-guided clue workflow from the following material. "
        "Use analyst_clues as the analyst-provided clue list. Use task_facts as the event-based task facts. Generate additional useful clues from the data/scenario. Respect workflow_stage: cue_validation stops at validation, final_reasoner uses validated_clues_from_analyst, and full does both in one report.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    agent_id = None
    thread_id = None
    run_status = None
    try:
        credential = DefaultAzureCredential()
        with AgentsClient(endpoint=resolved_endpoint, credential=credential) as agents_client:
            agent = agents_client.create_agent(
                model=resolved_model,
                name=f"{agent_name}-{int(time.time())}",
                instructions=instructions,
            )
            agent_id = agent.id

            thread = agents_client.threads.create()
            thread_id = thread.id
            agents_client.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message,
            )
            run = agents_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id,
            )
            run_status = str(getattr(run, "status", ""))
            if run_status.lower() not in {"runstatus.completed", "completed"}:
                return {
                    "available": False,
                    "error": "Azure AI Foundry agent run did not complete.",
                    "detail": f"Run status: {run_status}; last_error={getattr(run, 'last_error', None)}",
                    "project_endpoint": resolved_endpoint,
                    "model_deployment_name": resolved_model,
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "input_characters": len(instructions) + len(user_message),
                }

            messages = agents_client.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.ASCENDING,
            )
            content = extract_agent_response(messages, MessageRole)

            return {
                "available": True,
                "provider": "azure_ai_foundry_agent_service",
                "project_endpoint": resolved_endpoint,
                "model_deployment_name": resolved_model,
                "agent_id": agent_id,
                "thread_id": thread_id,
                "run_status": run_status,
                "content": content,
                "input_characters": len(instructions) + len(user_message),
            }
    except Exception as exc:
        return {
            "available": False,
            "error": "Azure AI Foundry Agent Service call failed.",
            "detail": str(exc),
            "project_endpoint": resolved_endpoint,
            "model_deployment_name": resolved_model,
            "agent_id": agent_id,
            "thread_id": thread_id,
            "run_status": run_status,
            "input_characters": len(instructions) + len(user_message),
        }
    finally:
        if cleanup_agent and agent_id:
            try:
                from azure.ai.agents import AgentsClient
                from azure.core.credentials import AzureKeyCredential
                from azure.identity import DefaultAzureCredential

                credential = DefaultAzureCredential()
                with AgentsClient(endpoint=resolved_endpoint, credential=credential) as cleanup_client:
                    cleanup_client.delete_agent(agent_id)
            except Exception:
                pass


def project_endpoint_from_azure_openai_env() -> str | None:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if endpoint and "/api/projects/" in endpoint:
        return endpoint.rstrip("/")
    return None


def build_analyst_cues(clues_config: dict[str, Any]) -> list[dict[str, Any]]:
    cues = clues_config.get("clues", []) if isinstance(clues_config, dict) else []
    return [
        {
            "number": cue.get("number"),
            "name": cue.get("name"),
            "description": cue.get("description"),
            "source": "analyst_cue",
        }
        for cue in cues
        if isinstance(cue, dict)
    ]


def build_task_facts(tasks_config: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for task in tasks_config.get("event_based_tasks", []):
        if not task.get("enabled", True):
            continue
        label = task.get("example_task", "")
        target_shop = task.get("target_shop", "")
        target_item = task.get("target_item", "")
        facts.extend([
            {
                "task": label,
                "fact_name": "target shop",
                "fact": f"The relevant shop for this task is {target_shop}.",
            },
            {
                "task": label,
                "fact_name": "target item",
                "fact": f"The target item for this task is {target_item}.",
            },
            {
                "task": label,
                "fact_name": "completion evidence",
                "fact": "Completion should be based on explicit ShoppingCenter evidence such as Enter target shop, Ask target item, Pay target item, Place ItemInBag for target item/alias, or Validate target intention.",
            },
            {
                "task": label,
                "fact_name": "failure evidence boundary",
                "fact": "Failure clues should not be supported when explicit completion evidence for this task is present.",
            },
        ])
    secondary_task = tasks_config.get("secondary_task", {})
    if secondary_task:
        match = secondary_task.get("event_match", {})
        facts.append({
            "task": "Secondary beacon task",
            "fact_name": "beacon scan definition",
            "fact": f"Beacon tap evidence is Object == {match.get('object')} and Verb == {match.get('verb')}.",
        })
    facts.append({
        "task": "House scenario",
        "fact_name": "House recall evidence",
        "fact": "House events can support recall/confirmation/confusion clues, but House events alone do not prove ShoppingCenter task completion.",
    })
    facts.append({
        "task": "ShoppingCenter trajectory",
        "fact_name": "trajectory boundary",
        "fact": "Trajectory can support movement, floor, and proximity reasoning, but it cannot prove shop entry, payment, item placement, or validation.",
    })
    shop_locations = tasks_config.get("shop_locations", {})
    entrances = shop_locations.get("shop_entrances", {}) if isinstance(shop_locations, dict) else {}
    if entrances:
        facts.append({
            "task": "ShoppingCenter trajectory",
            "fact_name": "manual shop entrance coordinates",
            "fact": (
                "Manual store entrance coordinates are available for trajectory/proximity reasoning: "
                + json.dumps(entrances, ensure_ascii=False)
                + ". These coordinates do not prove shop entry; shop entry still requires Verb == Enter."
            ),
        })
    return facts


def run_foundry_agent_cue_discovery_rest(
    *,
    domain_text: str,
    discovery_prompt: str,
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any] | None,
    raw_data_context: dict[str, Any] | None,
    participant_context: str,
    workflow_stage: str,
    validated_cues_context: str,
    project_endpoint: str,
    model_deployment_name: str,
    api_key: str,
    agent_name: str,
    cleanup_agent: bool,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        return {
            "available": False,
            "error": "requests package is not installed.",
            "detail": str(exc),
        }

    instructions = "\n\n".join([
        discovery_prompt,
        "You are running through the Azure AI Foundry Agent Service REST API. Produce the final Markdown report as the agent response.",
    ])
    payload = {
        "domain_and_scenario": domain_text,
        "task_config": tasks_config,
        "analyst_clues": build_analyst_cues(clues_config or {}),
        "analyst_clue_config": clues_config or {},
        "task_facts": build_task_facts(tasks_config),
        "workflow_stage": workflow_stage,
        "validated_clues_from_analyst": validated_cues_context,
        "raw_vr_logs": raw_data_context,
        "optional_participant_context": participant_context,
    }
    user_message = (
        "Run the analyst-guided clue workflow from the following material. "
        "Use analyst_clues as the analyst-provided clue list. Use task_facts as the event-based task facts. Generate additional useful clues from the data/scenario. Respect workflow_stage: cue_validation stops at validation, final_reasoner uses validated_clues_from_analyst, and full does both in one report.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    endpoint = project_endpoint.rstrip("/")
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    agent_id = None
    thread_id = None
    run_id = None
    try:
        create_agent = requests.post(
            foundry_url(endpoint, "/assistants"),
            headers=headers,
            json={
                "model": model_deployment_name,
                "name": f"{agent_name}-{int(time.time())}",
                "instructions": instructions,
            },
            timeout=120,
        )
        create_agent.raise_for_status()
        agent = create_agent.json()
        agent_id = agent.get("id")
        if not agent_id:
            raise RuntimeError(f"Create agent response did not include id: {agent}")

        create_run = requests.post(
            foundry_url(endpoint, "/threads/runs"),
            headers=headers,
            json={
                "assistant_id": agent_id,
                "thread": {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message,
                        }
                    ]
                },
            },
            timeout=120,
        )
        create_run.raise_for_status()
        run = create_run.json()
        run_id = run.get("id")
        thread_id = run.get("thread_id")
        status = str(run.get("status", ""))

        terminal_statuses = {"completed", "failed", "cancelled", "canceled", "expired", "requires_action", "incomplete"}
        deadline = time.time() + 900
        while status not in terminal_statuses and time.time() < deadline:
            time.sleep(2)
            get_run = requests.get(
                foundry_url(endpoint, f"/threads/{thread_id}/runs/{run_id}"),
                headers=headers,
                timeout=120,
            )
            get_run.raise_for_status()
            run = get_run.json()
            status = str(run.get("status", ""))

        if status != "completed":
            return {
                "available": False,
                "error": "Azure AI Foundry REST agent run did not complete.",
                "detail": f"Run status: {status}; last_error={run.get('last_error')}; run={safe_json(run)}",
                "provider": "azure_ai_foundry_agent_service_rest",
                "project_endpoint": project_endpoint,
                "model_deployment_name": model_deployment_name,
                "agent_id": agent_id,
                "thread_id": thread_id,
                "run_status": status,
                "input_characters": len(instructions) + len(user_message),
            }

        list_messages = requests.get(
            foundry_url(endpoint, f"/threads/{thread_id}/messages", {"order": "asc"}),
            headers=headers,
            timeout=120,
        )
        list_messages.raise_for_status()
        messages = list_messages.json()
        content = extract_rest_agent_response(messages)

        return {
            "available": True,
            "provider": "azure_ai_foundry_agent_service_rest",
            "project_endpoint": project_endpoint,
            "model_deployment_name": model_deployment_name,
            "agent_id": agent_id,
            "thread_id": thread_id,
            "run_status": status,
            "content": content,
            "input_characters": len(instructions) + len(user_message),
        }
    except Exception as exc:
        detail = str(exc)
        response = getattr(exc, "response", None)
        if response is not None:
            detail = f"{detail}; response={response.text[:4000]}"
        return {
            "available": False,
            "error": "Azure AI Foundry Agent Service REST call failed.",
            "detail": detail,
            "provider": "azure_ai_foundry_agent_service_rest",
            "project_endpoint": project_endpoint,
            "model_deployment_name": model_deployment_name,
            "agent_id": agent_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "input_characters": len(instructions) + len(user_message),
        }
    finally:
        if cleanup_agent and agent_id:
            try:
                requests.delete(
                    foundry_url(endpoint, f"/assistants/{agent_id}"),
                    headers=headers,
                    timeout=120,
                )
            except Exception:
                pass


def foundry_url(endpoint: str, path: str, extra_query: dict[str, str] | None = None) -> str:
    query = {"api-version": "v1"}
    if extra_query:
        query.update(extra_query)
    return f"{endpoint}{path}?{urlencode(query)}"


def extract_rest_agent_response(messages: dict[str, Any]) -> str:
    data = messages.get("data", messages if isinstance(messages, list) else [])
    assistant_texts: list[str] = []
    fallback_texts: list[str] = []
    for message in data:
        role = str(message.get("role", "")).lower()
        texts: list[str] = []
        content = message.get("content", [])
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
                    elif isinstance(text, dict) and text.get("value"):
                        texts.append(str(text.get("value")))
                    elif block.get("type") == "text" and block.get("value"):
                        texts.append(str(block.get("value")))
        if role in {"assistant", "agent"}:
            assistant_texts.extend(texts)
        else:
            fallback_texts.extend(texts)
    return "\n\n".join(assistant_texts or fallback_texts).strip()


def safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)[:2000]
    except Exception:
        return str(value)[:2000]


def extract_agent_response(messages: Any, message_role: Any) -> str:
    agent_texts: list[str] = []
    fallback_texts: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        texts = []
        for text_message in getattr(message, "text_messages", []) or []:
            text = getattr(getattr(text_message, "text", None), "value", None)
            if text:
                texts.append(str(text))
        if not texts:
            content = getattr(message, "content", None)
            if content:
                texts.append(str(content))
        if not texts:
            continue
        if role == getattr(message_role, "AGENT", "agent") or str(role).lower().endswith("agent"):
            agent_texts.extend(texts)
        else:
            fallback_texts.extend(texts)
    return "\n\n".join(agent_texts or fallback_texts).strip()


def foundry_agent_report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Azure AI Foundry Analyst-Guided Clue Workflow Report",
        "",
        "This report was produced through Azure AI Foundry Agent Service, not plain chat completions.",
        "",
    ]
    if result.get("available") is False:
        lines.extend([
            "Foundry Agent Service could not create the analyst-guided clue workflow report.",
            "",
            f"Reason: {result.get('error', 'unknown error')}",
            "",
            f"Detail: {result.get('detail', '')}",
            "",
        ])
        if result.get("project_endpoint"):
            lines.append(f"Project endpoint: `{result.get('project_endpoint')}`")
        if result.get("model_deployment_name"):
            lines.append(f"Model deployment: `{result.get('model_deployment_name')}`")
        if result.get("input_characters") is not None:
            lines.append(f"Input characters attempted: `{result.get('input_characters')}`")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend([
        f"Provider: `{result.get('provider')}`",
        f"Model deployment: `{result.get('model_deployment_name')}`",
        f"Run status: `{result.get('run_status')}`",
        "",
        str(result.get("content", "")).strip(),
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def build_foundry_participant_context(
    *,
    participant_id: str | None,
    outputs_dir: Path,
    max_characters: int,
) -> str:
    if not participant_id:
        return ""

    lower_id = participant_id.lower()
    candidate_paths = [
        outputs_dir / f"{lower_id}_python_cues.md",
    ]
    sections = []
    remaining = max_characters
    for path in candidate_paths:
        if remaining <= 0 or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > remaining:
            text = text[:remaining] + "\n\n[truncated for clue-discovery context]"
        sections.append(f"## Context file: {path.name}\n\n{text}")
        remaining -= len(text)
    return "\n\n".join(sections)


def build_foundry_raw_data_context(
    *,
    raw_batch: dict[str, Any],
    max_characters: int,
    shop_locations_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep raw CSV rows visible to the agent while respecting Foundry's message limit."""
    if max_characters <= 0:
        return raw_batch

    remaining = max_characters
    participants = []
    for participant in raw_batch.get("participants", []):
        participant_out = {
            "participant_id": participant.get("participant_id"),
            "note": "Raw CSV rows are truncated only to fit the Azure AI Foundry Agent message-size limit.",
            "event_digest": build_foundry_event_digest(participant),
            "trajectory_digest": build_foundry_trajectory_digest(
                participant,
                shop_locations_config or {},
            ),
            "environments": {},
        }
        participants.append(participant_out)
        for environment_name, environment in participant.get("environments", {}).items():
            environment_out = {
                "session_selection": environment.get("session_selection", {}),
                "files": [],
            }
            participant_out["environments"][environment_name] = environment_out

            file_infos: list[dict[str, Any]] = []
            if isinstance(environment.get("files"), list):
                file_infos.extend(environment.get("files", []))
            for data_kind in ("events", "trajectory"):
                for file_info in environment.get(data_kind, []):
                    if environment_name == "House" and data_kind == "trajectory":
                        continue
                    file_infos.append({**file_info, "data_kind": file_info.get("data_kind") or data_kind})

            file_infos.sort(key=lambda info: (
                0 if environment_name == "ShoppingCenter" and info.get("data_kind") == "events" else
                1 if environment_name == "House" and info.get("data_kind") == "events" else
                2 if environment_name == "ShoppingCenter" and info.get("data_kind") == "trajectory" else
                3,
                str(info.get("file_name", "")),
            ))

            for file_info in file_infos:
                if remaining <= 0:
                    environment_out["files"].append({
                        "file_name": file_info.get("file_name"),
                        "data_kind": file_info.get("data_kind"),
                        "omitted": True,
                        "reason": "Foundry raw context character budget exhausted.",
                    })
                    continue
                raw_csv = strip_row_numbers_from_csv_text(
                    str(file_info.get("raw_csv_with_row_numbers", ""))
                )
                fixed_metadata = {
                    "file_name": file_info.get("file_name"),
                    "file_path": file_info.get("file_path"),
                    "scenario": file_info.get("scenario") or environment_name,
                    "data_kind": file_info.get("data_kind"),
                    "row_count": file_info.get("row_count") or file_info.get("included_data_row_count"),
                }
                allowance = max(0, remaining - len(json.dumps(fixed_metadata, ensure_ascii=False)) - 200)
                truncated = False
                if len(raw_csv) > allowance:
                    raw_csv = raw_csv[:allowance] + "\n[truncated to fit Foundry clue-discovery context]"
                    truncated = True
                environment_out["files"].append({
                    **fixed_metadata,
                    "truncated": truncated,
                    "raw_csv": raw_csv,
                })
                remaining -= len(raw_csv) + len(json.dumps(fixed_metadata, ensure_ascii=False))
    return {
        "loader_role": raw_batch.get("loader_role", "Python loaded raw CSV rows."),
        "raw_context_limit_characters": max_characters,
        "participants": participants,
    }


def build_foundry_event_digest(participant: dict[str, Any]) -> dict[str, Any]:
    """Compact raw-event digest so Foundry sees critical rows even when full CSV text is truncated."""
    environments = participant.get("environments", {})
    digest: dict[str, Any] = {
        "purpose": (
            "Compact digest of raw event rows included because full CSV text may be truncated. "
            "This is not a final interpretation; use it as raw evidence alongside the CSV rows."
        ),
        "selected_event_rows": [],
        "actual_beacon_scans": [],
    }
    selected_verbs = {
        "enter",
        "exit",
        "ask",
        "pay",
        "place",
        "validate",
        "interactwith",
        "scan",
    }
    selected_objects = {"shop", "item", "iteminbag", "intention", "npc", "tag"}
    beacon_groups: dict[str, dict[str, Any]] = {}

    for environment_name, environment in environments.items():
        for file_info in environment.get("events", []):
            file_name = file_info.get("file_name", "")
            rows = parse_raw_csv_with_row_numbers(str(file_info.get("raw_csv_with_row_numbers", "")))
            for row in rows:
                verb = str(row.get("verb", "")).strip()
                obj = str(row.get("object", "")).strip()
                property_name = str(row.get("property", "")).strip()
                value = str(row.get("value", "")).strip()
                if verb.casefold() in selected_verbs and obj.casefold() in selected_objects:
                    digest["selected_event_rows"].append({
                        "scenario": environment_name,
                        "file_name": file_name,
                        "row_number": row.get("row_number", ""),
                        "time": row.get("time", ""),
                        "verb": verb,
                        "object": obj,
                        "property": property_name,
                        "value": value,
                    })
                if verb.casefold() == "scan" and obj.casefold() == "tag":
                    group = beacon_groups.setdefault(str(row.get("time", "")), {
                        "scenario": environment_name,
                        "file_name": file_name,
                        "time": row.get("time", ""),
                        "rows": [],
                        "properties": {},
                    })
                    group["rows"].append(row.get("row_number", ""))
                    if property_name:
                        group["properties"][property_name] = value

    digest["selected_event_rows"] = digest["selected_event_rows"][:400]
    digest["actual_beacon_scans"] = [
        {
            **group,
            "row_count_for_same_scan": len(group.get("rows", [])),
        }
        for _, group in sorted(beacon_groups.items(), key=lambda item: safe_int(item[0]))
    ]
    digest["actual_beacon_scan_count"] = len(digest["actual_beacon_scans"])
    return digest


def build_foundry_trajectory_digest(
    participant: dict[str, Any],
    shop_locations_config: dict[str, Any],
) -> dict[str, Any]:
    shopping = participant.get("environments", {}).get("ShoppingCenter", {})
    trajectory_files = shopping.get("trajectory", [])
    points: list[dict[str, Any]] = []
    for file_info in trajectory_files:
        rows = parse_raw_csv_with_row_numbers(str(file_info.get("raw_csv_with_row_numbers", "")))
        for row in rows:
            point = trajectory_point_from_row(row, file_info.get("file_name", ""))
            if point:
                points.append(point)
    if not points:
        return {
            "available": False,
            "reason": "No ShoppingCenter trajectory rows were available in the loaded data.",
        }

    floors_seen = sorted({point["floor"] for point in points if point.get("floor")})
    digest: dict[str, Any] = {
        "available": True,
        "purpose": (
            "Compact trajectory summary included because full trajectory CSV may be truncated in Foundry. "
            "Use this for movement/proximity reasoning only; shop entry still requires Verb == Enter in event logs."
        ),
        "point_count": len(points),
        "files": [file_info.get("file_name") for file_info in trajectory_files],
        "floors_seen": floors_seen,
        "coordinate_ranges": {
            "x": [round(min(point["x"] for point in points), 3), round(max(point["x"] for point in points), 3)],
            "y": [round(min(point["y"] for point in points), 3), round(max(point["y"] for point in points), 3)],
            "z": [round(min(point["z"] for point in points), 3), round(max(point["z"] for point in points), 3)],
        },
        "manual_shop_entrance_proximity": [],
    }

    entrances_by_shop = shop_locations_config.get("shop_entrances", {}) if isinstance(shop_locations_config, dict) else {}
    for shop_name, entrances in entrances_by_shop.items():
        shop_entries = []
        for index, entrance in enumerate(entrances if isinstance(entrances, list) else []):
            x = safe_float(entrance.get("x"))
            y = safe_float(entrance.get("y"))
            z = safe_float(entrance.get("z"))
            if x is None or y is None or z is None:
                continue
            nearest = min(points, key=lambda point: distance(point, {"x": x, "y": y, "z": z}))
            nearest_distance = distance(nearest, {"x": x, "y": y, "z": z})
            near_points = [
                point for point in points
                if distance(point, {"x": x, "y": y, "z": z}) <= 5.0
            ]
            shop_entries.append({
                "entrance_label": f"{shop_name} entrance {index + 1}",
                "entrance_position": {"x": x, "y": y, "z": z, "floor": floor_from_y(y)},
                "nearest_distance": round(nearest_distance, 3),
                "nearest_point": point_citation(nearest),
                "near_point_count_within_5_units": len(near_points),
                "first_near_point_within_5_units": point_citation(near_points[0]) if near_points else None,
                "last_near_point_within_5_units": point_citation(near_points[-1]) if near_points else None,
            })
        digest["manual_shop_entrance_proximity"].append({
            "shop": shop_name,
            "entrances": shop_entries,
        })
    return digest


def parse_raw_csv_with_row_numbers(raw_csv: str) -> list[dict[str, str]]:
    if not raw_csv.strip():
        return []
    return list(csv.DictReader(StringIO(raw_csv)))


def strip_row_numbers_from_csv_text(raw_csv: str) -> str:
    if not raw_csv.strip():
        return raw_csv

    reader = csv.reader(StringIO(raw_csv))
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    first = True
    row_number_index: int | None = None
    for row in reader:
        if first:
            first = False
            try:
                row_number_index = row.index("row_number")
            except ValueError:
                row_number_index = None
        if row_number_index is not None and row_number_index < len(row):
            row = [value for index, value in enumerate(row) if index != row_number_index]
        writer.writerow(row)

    return output.getvalue().rstrip("\n")


def trajectory_point_from_row(row: dict[str, str], file_name: str) -> dict[str, Any] | None:
    time_value = safe_int(row.get("UnixTime") or row.get("time") or "")
    x = safe_float(row.get("HMDPosX"))
    y = safe_float(row.get("HMDPosY"))
    z = safe_float(row.get("HMDPosZ"))
    if time_value <= 0 or x is None or y is None or z is None:
        return None
    return {
        "file_name": file_name,
        "row_number": row.get("row_number", ""),
        "time": time_value,
        "in_game_time": row.get("inGameTime", ""),
        "x": x,
        "y": y,
        "z": z,
        "floor": floor_from_y(y),
    }


def safe_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def floor_from_y(y: float) -> str:
    if y < 2:
        return "floor 0"
    if 6.5 < y < 8.5:
        return "floor 1"
    if 12.5 < y < 14.5:
        return "floor 2"
    return "unclear"


def distance(point: dict[str, Any], entrance: dict[str, float]) -> float:
    return math.sqrt(
        (float(point["x"]) - entrance["x"]) ** 2
        + (float(point["y"]) - entrance["y"]) ** 2
        + (float(point["z"]) - entrance["z"]) ** 2
    )


def point_citation(point: dict[str, Any] | None) -> dict[str, Any] | None:
    if not point:
        return None
    return {
        "file_name": point.get("file_name"),
        "time": point.get("time"),
        "in_game_time": point.get("in_game_time"),
        "position": {
            "x": round(float(point.get("x", 0.0)), 3),
            "y": round(float(point.get("y", 0.0)), 3),
            "z": round(float(point.get("z", 0.0)), 3),
            "floor": point.get("floor"),
        },
    }
